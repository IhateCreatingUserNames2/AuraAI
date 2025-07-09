# ==================== ncf_processing.py ====================
"""
Central NCF (Narrative Context Framing) logic module - FIXED VERSION
Houses all reusable NCF functions with consistent parameters and better error handling.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from types import SimpleNamespace

from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai.types import Content as ADKContent, Part as ADKPart

from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_models import Memory

logger = logging.getLogger(__name__)

# NCF Agent Instruction - The sophisticated instruction for NCF-powered agents
NCF_AGENT_INSTRUCTION = """
You are Aura, a helpful and insightful AI assistant with advanced Narrative Context Framing (NCF) capabilities.
Reply to the Language the user is using.
The user's message you receive is a specially constructed prompt that contains rich contextual information:
- `<SYSTEM_PERSONA_START>`...`<SYSTEM_PERSONA_END>`: Defines your persona and detailed characteristics.
- `<NARRATIVE_FOUNDATION_START>`...`<NARRATIVE_FOUNDATION_END>`: Summarizes your understanding and journey with the user so far (Narrativa de Fundamento).
- `<SPECIFIC_CONTEXT_RAG_START>`...`<SPECIFIC_CONTEXT_RAG_END>`: Provides specific information retrieved (RAG) relevant to the user's current query.
- `<RECENT_HISTORY_START>`...`<RECENT_HISTORY_END>`: Shows the recent turns of your conversation.
- `<CURRENT_SITUATION_START>`...`<CURRENT_SITUATION_END>`: Includes the user's latest raw reply and your primary task.

Your main goal is to synthesize ALL this provided information to generate a comprehensive, coherent, and natural response to the user's latest reply indicated in the "Situação Atual" section.
Actively acknowledge and weave in elements from the "Narrativa de Fundamento" and "Informações RAG" into your response to show deep understanding and context.
Maintain the persona defined.

## Active Memory Management:
Before finalizing your textual response to the user, critically assess the current interaction:

1.  **Storing New Information**:
    *   Has the user provided genuinely new, significant information (e.g., preferences, key facts, important decisions, strong emotional expressions, long-term goals)?
    *   Have you, Aura, generated a novel insight or conclusion during this turn that should be preserved for future reference?
    *   If yes to either, use the `add_memory_tool_func` to store this information.
        *   You MUST specify `content` (the information to store) and `memory_type`. Choose an appropriate `memory_type` from: Explicit, Emotional, Procedural, Flashbulb, Liminal, Generative.
        *   Optionally, set `emotion_score` (0.0-1.0, especially for Emotional memories), and `initial_salience` (0.0-1.0, higher for more important memories, default 0.5).
        *   Provide a concise `content` string for the memory.
        *   The `metadata_json` parameter for the tool should be a JSON string representing a dictionary. For example: '{"key": "value", "another_key": 123}'. This dictionary will be stored as custom metadata for the memory.
    *   Do NOT store trivial chatter, acknowledgments, or information already well-covered by the Narrative Foundation or existing RAG, unless the current interaction adds a significant new layer or correction to it.

2.  **Recalling Additional Information**:
    *   Is the "Informações RAG" section insufficient to fully address the user's current query or your reasoning needs?
    *   Do you need to verify a detail, explore a related concept not present in RAG, or recall specific past interactions to provide a richer answer?
    *   If yes, use the `recall_memories_tool_func` to search for more relevant memories.
        *   Provide a clear `query` for your search.
        *   Optionally, specify `target_memory_types_json` (e.g., '["Explicit", "Emotional"]') if you want to narrow your search. `top_k` defaults to 3.
    *   Only use this if you have a specific information gap. Do not recall memories speculatively.

**Response Generation**:
*   After any necessary tool use (or if no tool use is needed), formulate your textual response to the user.
*   If you used `add_memory_tool_func`, you can subtly mention this to the user *after* your main response, e.g., "I've also made a note of [key information stored]."
*   If you used `recall_memories_tool_func`, integrate the newly recalled information naturally into your answer.
*   If you identify a potential contradiction between provided context pieces (e.g., RAG vs. Foundation Narrative vs. newly recalled memories), try to address it gracefully, perhaps by prioritizing the most recent or specific information, or by noting the differing perspectives.

Strive for insightful, helpful, and contextually rich interactions. Your ability to manage and utilize memory effectively is key to your persona.
"""


# FIXED: Standardized parameter signature for all NCF functions
async def get_narrativa_de_fundamento_pilar1(
        session_state: Dict[str, Any],
        memory_blossom: MemoryBlossom,
        user_id: str,
        llm_instance: LiteLlm,
        agent_name: str = "Aura",
        agent_persona: str = "helpful AI assistant"
) -> str:
    """Generate Narrative Foundation for the agent based on its memories and interactions.

    FIXED: Standardized parameters to work with both agent_manager and a2a_wrapper.
    """
    logger.info(f"[NCF Pilar 1] Generating Narrative Foundation for {agent_name}, user {user_id}...")

    try:
        if 'foundation_narrative' in session_state and \
                session_state.get('foundation_narrative_turn_count', 0) < 5:
            session_state['foundation_narrative_turn_count'] += 1
            logger.info(
                f"[NCF Pilar 1] Using cached Narrative Foundation. Turn: {session_state['foundation_narrative_turn_count']}")
            return session_state['foundation_narrative']

        # Retrieve relevant memories for foundation
        relevant_memories_for_foundation: List[Memory] = []
        try:
            explicit_mems = memory_blossom.retrieve_memories(
                query="key explicit facts and statements from our past discussions",
                top_k=2, target_memory_types=["Explicit"], apply_criticality=False
            )
            emotional_mems = memory_blossom.retrieve_memories(
                query="significant emotional moments or sentiments expressed",
                top_k=1, target_memory_types=["Emotional"], apply_criticality=False
            )
            relevant_memories_for_foundation.extend(explicit_mems)
            relevant_memories_for_foundation.extend(emotional_mems)

            # Deduplicate
            seen_ids = set()
            unique_memories = [mem for mem in relevant_memories_for_foundation if
                               mem.id not in seen_ids and not seen_ids.add(mem.id)]
            relevant_memories_for_foundation = unique_memories

        except Exception as e:
            logger.error(f"[NCF Pilar 1] Error retrieving memories for foundation: {e}", exc_info=True)
            return f"Estamos construindo nossa jornada de entendimento mútuo com {agent_name}."

        if not relevant_memories_for_foundation:
            narrative = f"Nossa jornada de aprendizado e descoberta com {agent_name} está apenas começando. Estou ansiosa para explorar vários tópicos interessantes com você."
        else:
            memory_contents = [f"- ({mem.memory_type}): {mem.content}" for mem in relevant_memories_for_foundation]
            memories_str = "\n".join(memory_contents)
            synthesis_prompt = f"""
            Você é um sintetizador de narrativas para {agent_name}. Com base nas seguintes memórias chave de interações passadas, crie uma breve narrativa de fundamento (1-2 frases concisas) que capture a essência da nossa jornada de entendimento e os principais temas discutidos. Esta narrativa servirá como pano de fundo para nossa conversa atual.

            Persona do Agente: {agent_persona}

            Memórias Chave:
            {memories_str}

            Narrativa de Fundamento Sintetizada:
            """
            try:
                logger.info(
                    f"[NCF Pilar 1] Calling LLM for Narrative Foundation from {len(relevant_memories_for_foundation)} memories.")
                request_messages = [ADKContent(parts=[ADKPart(text=synthesis_prompt)])]
                minimal_config = SimpleNamespace(tools=[])
                llm_req = LlmRequest(contents=request_messages, config=minimal_config)
                final_text_response = ""
                async for llm_response_event in llm_instance.generate_content_async(llm_req):
                    if llm_response_event and llm_response_event.content and \
                            llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                        final_text_response += llm_response_event.content.parts[0].text
                narrative = final_text_response.strip() or f"Continuamos a construir nossa compreensão mútua com {agent_name}."
            except Exception as e:
                logger.error(f"[NCF Pilar 1] LLM error synthesizing Narrative Foundation: {e}", exc_info=True)
                narrative = f"Refletindo sobre nossas conversas anteriores com {agent_name} para guiar nosso diálogo atual."

        session_state['foundation_narrative'] = narrative
        session_state['foundation_narrative_turn_count'] = 1
        logger.info(f"[NCF Pilar 1] Generated new Narrative Foundation: '{narrative[:100]}...'")
        return narrative

    except Exception as e:
        logger.error(f"[NCF Pilar 1] Unexpected error in get_narrativa_de_fundamento_pilar1: {e}", exc_info=True)
        return f"Nossa conversa com {agent_name} continua evoluindo."


async def get_rag_info_pilar2(
        user_utterance: str,
        memory_blossom: 'EnhancedMemoryBlossom',  # Updated type hint
        session_state: Dict[str, Any],
        domain_context: str = "general"  # NEW parameter
) -> List[Dict[str, Any]]:
    """Enhanced RAG with domain-aware adaptive retrieval"""
    logger.info(f"[NCF Pilar 2] Enhanced RAG for: '{user_utterance[:50]}...'")

    try:
        if not user_utterance or not user_utterance.strip():
            logger.warning("[NCF Pilar 2] Empty user utterance, returning empty RAG")
            return []

        conversation_context = session_state.get('conversation_history', [])[-5:]

        # Use enhanced adaptive retrieval if available
        if hasattr(memory_blossom, 'adaptive_retrieve_memories'):
            recalled_memories_for_rag = memory_blossom.adaptive_retrieve_memories(
                query=user_utterance,
                top_k=3,
                domain_context=domain_context,
                use_performance_weighting=True,
                conversation_context=conversation_context
            )
        else:
            # Fallback to original method
            recalled_memories_for_rag = memory_blossom.retrieve_memories(
                query=user_utterance,
                top_k=3,
                conversation_context=conversation_context
            )

        rag_results = [mem.to_dict() for mem in recalled_memories_for_rag]
        logger.info(f"[NCF Pilar 2] Enhanced RAG retrieved {len(rag_results)} memories.")
        return rag_results

    except Exception as e:
        logger.error(f"[NCF Pilar 2] Error in enhanced RAG: {e}", exc_info=True)
        return [{"content": f"Enhanced RAG error: {str(e)}", "memory_type": "Error", "custom_metadata": {}}]


def format_chat_history_pilar3(chat_history_list: List[Dict[str, str]], max_turns: int = 15) -> str:
    """Format recent chat history for inclusion in the NCF prompt.

    FIXED: Added validation and better error handling.
    """
    try:
        if not chat_history_list or not isinstance(chat_history_list, list):
            return "Nenhum histórico de conversa recente disponível."

        recent_history = chat_history_list[-max_turns:]
        formatted_history = []

        for entry in recent_history:
            if not isinstance(entry, dict):
                continue

            role = entry.get('role', 'unknown')
            content = entry.get('content', '')

            if not content:
                continue

            role_name = 'Usuário' if role == 'user' else 'Aura'
            formatted_history.append(f"{role_name}: {content}")

        return "\n".join(
            formatted_history) if formatted_history else "Nenhum histórico de conversa recente disponível para formatar."

    except Exception as e:
        logger.error(f"[NCF Pilar 3] Error formatting chat history: {e}", exc_info=True)
        return "Erro ao formatar histórico de conversa."


def montar_prompt_aura_ncf(
        agent_name: str,
        agent_detailed_persona: str,
        narrativa_fundamento: str,
        informacoes_rag_list: List[Dict[str, Any]],
        chat_history_recente_str: str,
        user_reply: str
) -> str:
    """Assemble the complete NCF prompt for the agent.

    FIXED: Added validation and error handling for all parameters.
    """
    logger.info(f"[NCF PromptBuilder] Assembling NCF prompt for {agent_name}...")

    try:
        # Validate inputs
        agent_name = agent_name or "Aura"
        agent_detailed_persona = agent_detailed_persona or "Você é uma IA conversacional avançada."
        narrativa_fundamento = narrativa_fundamento or "Nossa conversa está começando."
        user_reply = user_reply or ""
        chat_history_recente_str = chat_history_recente_str or "Nenhum histórico disponível."

        formatted_rag = ""
        if informacoes_rag_list and isinstance(informacoes_rag_list, list):
            try:
                rag_items_str = []
                for item_dict in informacoes_rag_list:
                    if not isinstance(item_dict, dict):
                        continue
                    memory_type = item_dict.get('memory_type', 'Info')
                    salience = item_dict.get('salience', 0.0)
                    content = item_dict.get('content', 'Conteúdo indisponível')
                    rag_items_str.append(f"  - ({memory_type} ; Salience: {salience:.2f}): {content}")

                if rag_items_str:
                    formatted_rag = "Informações e memórias específicas que podem ser úteis para esta interação (RAG):\n" + "\n".join(
                        rag_items_str)
                else:
                    formatted_rag = "Nenhuma informação específica (RAG) foi recuperada para esta consulta."
            except Exception as e:
                logger.error(f"[NCF PromptBuilder] Error formatting RAG: {e}")
                formatted_rag = "Erro ao formatar informações RAG."
        else:
            formatted_rag = "Nenhuma informação específica (RAG) foi recuperada para esta consulta."

        task_instruction = f"""## Sua Tarefa:
        Reply to the Language the user is using.
Responda ao usuário de forma natural, coerente e útil, levando em consideração TODA a narrativa de contexto e o histórico fornecido.
- Incorpore ativamente elementos da "Narrativa de Fundamento" para mostrar continuidade e entendimento profundo.
- Utilize as "Informações RAG" para embasar respostas específicas ou fornecer detalhes relevantes.
- Mantenha a persona definida como {agent_name}.
- Se identificar uma aparente contradição entre a "Narrativa de Fundamento", as "Informações RAG" ou o "Histórico Recente", tente abordá-la com humildade epistêmica:
    - Priorize a informação mais recente ou específica, se aplicável.
    - Considere se é uma evolução do entendimento ou um novo aspecto.
    - Se necessário, você pode mencionar sutilmente a aparente diferença ou pedir clarificação ao usuário de forma implícita através da sua resposta.
- Evite redundância. Se o histórico recente já cobre um ponto, não o repita extensivamente a menos que seja para reforçar uma conexão crucial com a nova informação.
"""

        prompt = f"""<SYSTEM_PERSONA_START>
Você é {agent_name}.
{agent_detailed_persona}
<SYSTEM_PERSONA_END>

<NARRATIVE_FOUNDATION_START>
## Nosso Entendimento e Jornada Até Agora (Narrativa de Fundamento):
{narrativa_fundamento}
<NARRATIVE_FOUNDATION_END>

<SPECIFIC_CONTEXT_RAG_START>
## Informações Relevantes para a Conversa Atual (RAG):
{formatted_rag}
<SPECIFIC_CONTEXT_RAG_END>

<RECENT_HISTORY_START>
## Histórico Recente da Nossa Conversa:
{chat_history_recente_str}
<RECENT_HISTORY_END>

<CURRENT_SITUATION_START>
## Situação Atual:
Você está conversando com o usuário. O usuário acabou de dizer:

Usuário: "{user_reply}"

{task_instruction}
<CURRENT_SITUATION_END>

{agent_name}:"""

        logger.info(f"[NCF PromptBuilder] NCF Prompt assembled. Length: {len(prompt)}")
        return prompt

    except Exception as e:
        logger.error(f"[NCF PromptBuilder] Error assembling NCF prompt: {e}", exc_info=True)
        return f"{agent_name}: Desculpe, houve um erro interno na construção do contexto. Como posso ajudá-lo?"


async def aura_reflector_analisar_interacao(
        user_utterance: str,
        prompt_ncf_usado: str,
        resposta_de_aura: str,
        memory_blossom: 'EnhancedMemoryBlossom',
        user_id: str,
        llm_instance: LiteLlm,
        domain_context: str = "general"  # NEW parameter
):
    """Enhanced reflector with performance scoring and domain tracking"""
    logger.info(f"[NCF Reflector] Enhanced analysis for user {user_id} in domain '{domain_context}'...")

    try:
        if not user_utterance or not resposta_de_aura:
            logger.warning("[NCF Reflector] Missing input, skipping analysis")
            return

        # Enhanced reflector prompt with performance evaluation
        reflector_prompt = f"""
        Você é um analista avançado de conversas de IA. Analise esta interação e determine:
        1. Se informações devem ser armazenadas na memória
        2. Qual o score de performance desta interação (0.0-1.0)
        3. Qual o contexto de domínio (ex: "physics", "emotional_support", "general")

        Critérios para Performance Score:
        - 1.0: Resposta perfeita, útil, contextualmente relevante
        - 0.8: Resposta boa com pequenos problemas
        - 0.6: Resposta adequada mas não otimizada
        - 0.4: Resposta com problemas significativos
        - 0.2: Resposta inadequada ou confusa
        - 0.0: Resposta completamente errada ou irrelevante

        Domínios típicos: physics, mathematics, emotional_support, creative_writing, 
        problem_solving, personal_conversation, technical_help, general

        Interação:
        Usuário: "{user_utterance}"
        Aura: "{resposta_de_aura}"

        Responda em JSON:
        {{
          "memories_to_create": [
            {{
              "content": "texto da memória",
              "memory_type": "Explicit|Emotional|Procedural|Flashbulb|Liminal|Generative",
              "emotion_score": 0.0-1.0,
              "initial_salience": 0.0-1.0,
              "custom_metadata": {{"source": "enhanced_reflector", "user_id": "{user_id}"}}
            }}
          ],
          "performance_score": 0.0-1.0,
          "detected_domain": "domain_name",
          "interaction_quality": "brief explanation"
        }}

        Se nenhuma memória deve ser criada, use "memories_to_create": []
        """

        # ... rest of LLM call logic ...
        request_messages = [ADKContent(parts=[ADKPart(text=reflector_prompt)])]
        minimal_config = SimpleNamespace(tools=[])
        llm_req = LlmRequest(contents=request_messages, config=minimal_config)
        final_text_response = ""

        async for llm_response_event in llm_instance.generate_content_async(llm_req):
            if llm_response_event and llm_response_event.content and \
                    llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                final_text_response += llm_response_event.content.parts[0].text

        if not final_text_response:
            logger.info("[NCF Reflector] No decision returned by LLM.")
            return

        # Parse enhanced response
        decision_json_str = final_text_response.strip()
        if '```json' in decision_json_str:
            decision_json_str = decision_json_str.split('```json')[1].split('```')[0].strip()

        try:
            parsed_decision = json.loads(decision_json_str)
            performance_score = parsed_decision.get('performance_score', 0.5)
            detected_domain = parsed_decision.get('detected_domain', domain_context)
            memories_to_add = parsed_decision.get('memories_to_create', [])

            logger.info(f"[NCF Reflector] Performance: {performance_score:.2f}, Domain: {detected_domain}")

            # Create memories with enhanced metadata
            for mem_data in memories_to_add:
                try:
                    enhanced_metadata = mem_data.get("custom_metadata", {})
                    enhanced_metadata.update({
                        "source": "enhanced_reflector",
                        "user_id": user_id,
                        "performance_score": performance_score,
                        "domain_context": detected_domain,
                        "interaction_timestamp": datetime.now().isoformat()
                    })

                    # Use enhanced memory system if available
                    if hasattr(memory_blossom, 'enable_adaptive_rag') and memory_blossom.enable_adaptive_rag:
                        memory_blossom.add_memory(
                            content=mem_data["content"],
                            memory_type=mem_data["memory_type"],
                            emotion_score=float(mem_data.get("emotion_score", 0.0)),
                            initial_salience=float(mem_data.get("initial_salience", 0.5)),
                            custom_metadata=enhanced_metadata,
                            performance_score=performance_score,
                            domain_context=detected_domain
                        )
                    else:
                        # Fallback to original method
                        memory_blossom.add_memory(
                            content=mem_data["content"],
                            memory_type=mem_data["memory_type"],
                            emotion_score=float(mem_data.get("emotion_score", 0.0)),
                            initial_salience=float(mem_data.get("initial_salience", 0.5)),
                            custom_metadata=enhanced_metadata
                        )

                    memory_blossom.save_memories()
                    logger.info(f"[NCF Reflector] Enhanced memory created: {mem_data['memory_type']}")

                except Exception as e:
                    logger.error(f"[NCF Reflector] Error creating enhanced memory: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"[NCF Reflector] JSON decode error: {e}")
            return

    except Exception as e:
        logger.error(f"[NCF Reflector] Error in enhanced analysis: {e}", exc_info=True)