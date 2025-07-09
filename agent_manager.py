# ==================== Complete Fixed agent_manager.py ====================
"""
Enhanced AgentManager with NCF-powered AuraAgentInstance - FIXED VERSION
Every created agent now has full NCF capabilities by default.
Fixed: All async issues and variable scope problems
"""

import os
import json
import time
import uuid
import asyncio
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_connector import MemoryConnector
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai.types import Content as ADKContent, Part as ADKPart
from google.adk.events import Event, EventActions

# Import NCF processing functions with FIXED signatures
from ncf_processing import (
    NCF_AGENT_INSTRUCTION,
    get_narrativa_de_fundamento_pilar1,
    get_rag_info_pilar2,
    format_chat_history_pilar3,
    montar_prompt_aura_ncf,
    aura_reflector_analisar_interacao
)

import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    agent_id: str
    user_id: str
    name: str
    persona: str
    detailed_persona: str
    model: str = "openrouter/openai/gpt-4o-mini"
    memory_path: Optional[str] = None
    created_at: datetime = None
    settings: Dict[str, Any] = None


class AgentManager:
    """Manages multiple NCF-enabled Aura agent instances for different users"""

    def __init__(self, base_storage_path: str = "agent_storage"):
        self.base_storage_path = Path(base_storage_path)
        self.base_storage_path.mkdir(exist_ok=True)

        # Cache of active agents
        self._active_agents: Dict[str, 'NCFAuraAgentInstance'] = {}
        self._agent_configs: Dict[str, AgentConfig] = {}

        # Load existing agent configurations
        self._load_agent_configs()

    def create_agent(self, user_id: str, name: str, persona: str,
                     detailed_persona: str, model: str = None) -> str:
        """Create a new NCF-enabled Aura agent for a user"""
        try:
            agent_id = str(uuid.uuid4())

            # Create agent-specific storage directory
            agent_path = self.base_storage_path / user_id / agent_id
            agent_path.mkdir(parents=True, exist_ok=True)

            # Memory persistence path - each agent gets its own MemoryBlossom
            memory_path = str(agent_path / "memory_blossom.json")

            config = AgentConfig(
                agent_id=agent_id,
                user_id=user_id,
                name=name,
                persona=persona,
                detailed_persona=detailed_persona,
                model=model or "openrouter/openai/gpt-4o-mini",
                memory_path=memory_path,
                created_at=datetime.now(),
                settings={}
            )

            # Save configuration
            self._save_agent_config(config)
            self._agent_configs[agent_id] = config

            logger.info(f"Created NCF-enabled agent '{name}' (ID: {agent_id}) for user {user_id}")
            return agent_id

        except Exception as e:
            logger.error(f"Error creating NCF-enabled agent: {e}", exc_info=True)
            raise

    def get_agent_instance(self, agent_id: str) -> Optional['NCFAuraAgentInstance']:
        """Get or create a NCF-enabled agent instance"""
        try:
            if agent_id not in self._agent_configs:
                return None

            # Return cached instance if available
            if agent_id in self._active_agents:
                return self._active_agents[agent_id]

            # Create new NCF-enabled instance
            config = self._agent_configs[agent_id]
            instance = NCFAuraAgentInstance(config)
            self._active_agents[agent_id] = instance

            return instance

        except Exception as e:
            logger.error(f"Error getting agent instance {agent_id}: {e}", exc_info=True)
            return None

    def list_user_agents(self, user_id: str) -> list[AgentConfig]:
        """List all agents for a user"""
        return [config for config in self._agent_configs.values()
                if config.user_id == user_id]

    def delete_agent(self, agent_id: str, user_id: str) -> bool:
        """Delete an agent (only by owner)"""
        try:
            if agent_id not in self._agent_configs:
                return False

            config = self._agent_configs[agent_id]
            if config.user_id != user_id:
                return False  # Not owner

            # Remove from active instances
            if agent_id in self._active_agents:
                del self._active_agents[agent_id]

            # Delete storage
            import shutil
            agent_path = self.base_storage_path / config.user_id / agent_id
            if agent_path.exists():
                shutil.rmtree(agent_path)

            # Remove config
            del self._agent_configs[agent_id]
            config_path = self.base_storage_path / config.user_id / f"{agent_id}.json"
            if config_path.exists():
                config_path.unlink()

            logger.info(f"Deleted agent {agent_id} for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting agent {agent_id}: {e}", exc_info=True)
            return False

    def _save_agent_config(self, config: AgentConfig):
        """Save agent configuration to disk"""
        try:
            user_path = self.base_storage_path / config.user_id
            user_path.mkdir(exist_ok=True)

            config_path = user_path / f"{config.agent_id}.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'agent_id': config.agent_id,
                    'user_id': config.user_id,
                    'name': config.name,
                    'persona': config.persona,
                    'detailed_persona': config.detailed_persona,
                    'model': config.model,
                    'memory_path': config.memory_path,
                    'created_at': config.created_at.isoformat(),
                    'settings': config.settings or {}
                }, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving agent config: {e}", exc_info=True)
            raise

    def _load_agent_configs(self):
        """Load all agent configurations from disk"""
        try:
            for user_dir in self.base_storage_path.iterdir():
                if user_dir.is_dir():
                    for config_file in user_dir.glob("*.json"):
                        # Skip memory files
                        if config_file.name == "memory_blossom.json":
                            continue

                        try:
                            with open(config_file, 'r') as f:
                                data = json.load(f)

                            if not all(k in data for k in
                                       ['agent_id', 'user_id', 'name', 'persona', 'detailed_persona', 'created_at']):
                                logger.warning(f"Config file {config_file} is missing essential keys. Skipping.")
                                continue

                            config = AgentConfig(
                                agent_id=data['agent_id'],
                                user_id=data['user_id'],
                                name=data['name'],
                                persona=data['persona'],
                                detailed_persona=data['detailed_persona'],
                                model=data.get('model', 'openrouter/openai/gpt-4o-mini'),
                                memory_path=data.get('memory_path'),
                                created_at=datetime.fromisoformat(data['created_at']),
                                settings=data.get('settings', {})
                            )

                            if not config.memory_path:
                                config.memory_path = str(
                                    self.base_storage_path / config.user_id / config.agent_id / "memory_blossom.json")

                            self._agent_configs[config.agent_id] = config
                            logger.info(f"Loaded agent config: {config.name} (ID: {config.agent_id})")

                        except Exception as e:
                            logger.error(f"Error loading config {config_file}: {e}")

        except Exception as e:
            logger.error(f"Error loading agent configs: {e}", exc_info=True)


class NCFAuraAgentInstance:
    """
    Individual NCF-enabled Aura agent instance with its own memory and session management.
    Every instance has full NCF capabilities by default.
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        logger.info(f"Initializing NCF-enabled agent: {config.name} (ID: {config.agent_id})")

        try:
            # Ensure memory path is set
            if not self.config.memory_path:
                self.config.memory_path = str(Path(
                    "agent_storage") / self.config.user_id / self.config.agent_id / "memory_blossom.json")

            # Initialize isolated MemoryBlossom for this agent
            try:
                from enhanced_memory_system import EnhancedMemoryBlossom
                self.memory_blossom = EnhancedMemoryBlossom(
                    persistence_path=config.memory_path,
                    enable_adaptive_rag=True,  # Enable GenLang features
                    cluster_threshold=0.75  # Adjust clustering sensitivity
                )
                logger.info(f"Using Enhanced MemoryBlossom with Adaptive RAG for {config.name}")
            except ImportError:
                logger.warning(f"Enhanced MemoryBlossom not available, falling back to standard MemoryBlossom")
                self.memory_blossom = MemoryBlossom(persistence_path=config.memory_path)

            self.memory_connector = MemoryConnector(self.memory_blossom)
            self.memory_blossom.set_memory_connector(self.memory_connector)

            # Initialize LLM for this agent
            self.model = LiteLlm(model=config.model)

            # Create memory management tools
            self.add_memory_tool = FunctionTool(func=self._create_add_memory_func())
            self.recall_memories_tool = FunctionTool(func=self._create_recall_memories_func())

            # Create the NCF-powered ADK agent
            self.adk_agent = self._create_ncf_adk_agent()

            # Session management
            self.session_service = InMemorySessionService()
            self.runner = Runner(
                agent=self.adk_agent,
                app_name=f"NCFAura_{config.agent_id}",
                session_service=self.session_service
            )
            self.active_sessions: Dict[str, str] = {}

            logger.info(f"NCF agent '{config.name}' initialized successfully with isolated memory system")

        except Exception as e:
            logger.error(f"Error initializing NCF agent instance: {e}", exc_info=True)
            raise

    def _create_ncf_adk_agent(self) -> LlmAgent:
        """Create the NCF-powered ADK agent with sophisticated instruction"""
        return LlmAgent(
            name=self.config.name,
            model=self.model,
            instruction=NCF_AGENT_INSTRUCTION,  # Use the sophisticated NCF instruction
            tools=[self.add_memory_tool, self.recall_memories_tool]
        )

    def _create_add_memory_func(self):
        """Create the add_memory function for the agent's tools"""

        def add_memory(content: str, memory_type: str,
                       emotion_score: float = 0.0,
                       initial_salience: float = 0.5,
                       metadata_json: Optional[str] = None,
                       domain_context: str = "general",
                       performance_score: float = 0.5,
                       tool_context=None) -> Dict[str, Any]:
            try:
                custom_metadata = json.loads(metadata_json) if metadata_json else {}
                custom_metadata['agent_id'] = self.config.agent_id
                custom_metadata['agent_name'] = self.config.name
                custom_metadata['domain_context'] = domain_context
                custom_metadata['performance_score'] = performance_score

                # Check if enhanced memory system is available
                if hasattr(self.memory_blossom, 'enable_adaptive_rag') and self.memory_blossom.enable_adaptive_rag:
                    memory = self.memory_blossom.add_memory(
                        content=content,
                        memory_type=memory_type,
                        custom_metadata=custom_metadata,
                        emotion_score=emotion_score,
                        initial_salience=initial_salience,
                        performance_score=performance_score,
                        domain_context=domain_context
                    )
                    message_suffix = " (Enhanced with Adaptive RAG)"
                else:
                    memory = self.memory_blossom.add_memory(
                        content=content,
                        memory_type=memory_type,
                        custom_metadata=custom_metadata,
                        emotion_score=emotion_score,
                        initial_salience=initial_salience
                    )
                    message_suffix = " (Standard MemoryBlossom)"

                self.memory_blossom.save_memories()

                return {
                    "status": "success",
                    "memory_id": memory.id,
                    "message": f"Memory stored successfully for {self.config.name} (ID: {memory.id}){message_suffix}",
                    "adaptive_rag_enabled": hasattr(self.memory_blossom, 'enable_adaptive_rag')
                }
            except Exception as e:
                logger.error(f"Error adding memory for agent {self.config.agent_id}: {e}")
                return {"status": "error", "message": str(e)}

        return add_memory

    def _create_recall_memories_func(self):
        """Create the recall_memories function for the agent's tools"""

        def recall_memories(query: str,
                            target_memory_types_json: Optional[str] = None,
                            top_k: int = 3,
                            domain_context: str = "general",
                            tool_context=None) -> Dict[str, Any]:
            try:
                target_types = None
                if target_memory_types_json:
                    target_types = json.loads(target_memory_types_json)

                conversation_history_for_retrieval = None
                if tool_context and tool_context.state and 'conversation_history' in tool_context.state:
                    conversation_history_for_retrieval = tool_context.state['conversation_history']

                # Use enhanced retrieval if available
                if hasattr(self.memory_blossom, 'adaptive_retrieve_memories'):
                    memories = self.memory_blossom.adaptive_retrieve_memories(
                        query=query,
                        target_memory_types=target_types,
                        domain_context=domain_context,
                        top_k=top_k,
                        use_performance_weighting=True,
                        conversation_context=conversation_history_for_retrieval
                    )
                    retrieval_method = "Enhanced Adaptive RAG"
                else:
                    memories = self.memory_blossom.retrieve_memories(
                        query=query,
                        target_memory_types=target_types,
                        top_k=top_k,
                        conversation_context=conversation_history_for_retrieval
                    )
                    retrieval_method = "Standard MemoryBlossom"

                return {
                    "status": "success",
                    "count": len(memories),
                    "memories": [mem.to_dict() for mem in memories],
                    "retrieval_method": retrieval_method,
                    "adaptive_rag_enabled": hasattr(self.memory_blossom, 'adaptive_retrieve_memories')
                }
            except Exception as e:
                logger.error(f"Error recalling memories for agent {self.config.agent_id}: {e}")
                return {"status": "error", "message": str(e)}

        return recall_memories

    async def process_message(self, user_id: str, session_id: str, message: str) -> str:
        """
        Process a message with full NCF capabilities:
        1. Build context (Narrative Foundation, RAG, Chat History)
        2. Construct NCF prompt
        3. Run ADK agent
        4. Analyze interaction with Reflector

        FIXED: All async issues resolved
        """
        logger.info(f"Processing message for NCF agent '{self.config.name}': '{message[:100]}...'")

        try:
            # ============ SESSION MANAGEMENT ============
            # Generate ADK session management
            adk_session_key = f"{user_id}_{session_id}"
            adk_session_id = self.active_sessions.get(adk_session_key)
            session_created_now = False

            if not adk_session_id:
                adk_session_id = f"ncf_adk_{self.config.agent_id}_{adk_session_key}_{datetime.now().timestamp()}"
                self.active_sessions[adk_session_key] = adk_session_id
                session_created_now = True

            # Initialize or retrieve session state - FIXED: Define this first
            current_session_state = {'conversation_history': [], 'foundation_narrative_turn_count': 0}

            if session_created_now:
                # FIXED: Await the async session creation
                created_session = await self.session_service.create_session(
                    app_name=f"NCFAura_{self.config.agent_id}",
                    user_id=user_id,
                    session_id=adk_session_id,
                    state=current_session_state
                )
                logger.info(f"Created new session {adk_session_id}")
            else:
                # FIXED: Await the async session retrieval
                retrieved_session = await self.session_service.get_session(
                    app_name=f"NCFAura_{self.config.agent_id}",
                    user_id=user_id,
                    session_id=adk_session_id
                )
                if retrieved_session and retrieved_session.state:
                    current_session_state = retrieved_session.state
                else:
                    current_session_state = {'conversation_history': [], 'foundation_narrative_turn_count': 0}
                logger.info(f"Retrieved existing session {adk_session_id}")

            # Add user message to conversation history
            current_session_state['conversation_history'].append({
                'role': 'user',
                'content': message
            })

            # ============ DOMAIN DETECTION ============
            domain_context = await self._detect_domain_context(message, current_session_state)
            logger.info(f"Detected domain context: {domain_context}")

            # ============ NCF PROMPT CONSTRUCTION ============
            logger.info(f"Building NCF context for agent '{self.config.name}'...")

            # Pilar 1: Narrative Foundation
            narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(
                session_state=current_session_state,
                memory_blossom=self.memory_blossom,
                user_id=user_id,
                llm_instance=self.model,
                agent_name=self.config.name,
                agent_persona=self.config.persona
            )

            # Pilar 2: RAG Information (Enhanced with domain context)
            try:
                # Try enhanced RAG first
                if hasattr(self.memory_blossom, 'adaptive_retrieve_memories'):
                    rag_memories = self.memory_blossom.adaptive_retrieve_memories(
                        query=message,
                        domain_context=domain_context,
                        top_k=3,
                        use_performance_weighting=True,
                        conversation_context=current_session_state.get('conversation_history', [])
                    )
                    logger.info(f"Used Enhanced Adaptive RAG, retrieved {len(rag_memories)} memories")
                else:
                    rag_memories = self.memory_blossom.retrieve_memories(
                        query=message,
                        top_k=3,
                        conversation_context=current_session_state.get('conversation_history', [])
                    )
                    logger.info(f"Used Standard RAG, retrieved {len(rag_memories)} memories")

                rag_info_list = [mem.to_dict() for mem in rag_memories]

            except Exception as e:
                logger.error(f"Error in RAG retrieval: {e}")
                rag_info_list = []

            # Pilar 3: Chat History
            chat_history_str = format_chat_history_pilar3(
                chat_history_list=current_session_state['conversation_history']
            )

            # Construct final NCF prompt
            final_ncf_prompt = montar_prompt_aura_ncf(
                agent_name=self.config.name,
                agent_detailed_persona=self.config.detailed_persona,
                narrativa_fundamento=narrativa_fundamento,
                informacoes_rag_list=rag_info_list,
                chat_history_recente_str=chat_history_str,
                user_reply=message
            )

            # ============ RUN ADK AGENT ============
            # FIXED: Update session state before running agent with proper async handling
            try:
                # FIXED: Await the async session retrieval
                session_for_update = await self.session_service.get_session(
                    app_name=f"NCFAura_{self.config.agent_id}",
                    user_id=user_id,
                    session_id=adk_session_id
                )
                if session_for_update:
                    # Update the session state
                    session_for_update.state = current_session_state

                    # FIXED: Use update_session if available, otherwise skip
                    if hasattr(self.session_service, 'update_session'):
                        # Check if update_session is async
                        if asyncio.iscoroutinefunction(self.session_service.update_session):
                            await self.session_service.update_session(session_for_update)
                        else:
                            self.session_service.update_session(session_for_update)
                    # If no update_session method, the state is already updated in memory

            except Exception as e:
                logger.warning(f"Error updating session state: {e}")
                # Continue execution - this is not critical

            # Run the NCF-powered agent
            adk_message = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt)])
            response_text = ""

            try:
                async for event in self.runner.run_async(
                        user_id=user_id,
                        session_id=adk_session_id,
                        new_message=adk_message
                ):
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            response_text = event.content.parts[0].text
                        break
            except Exception as e:
                logger.error(f"Error running ADK agent: {e}")
                response_text = f"Desculpe, houve um problema na geração da resposta. Como {self.config.name}, vou tentar ajudá-lo da melhor forma possível."

            response_text = response_text or f"({self.config.name} não forneceu uma resposta para este turno)"

            # Add assistant response to conversation history
            current_session_state['conversation_history'].append({
                'role': 'assistant',
                'content': response_text
            })

            # ============ REFLECTOR ANALYSIS ============
            logger.info(f"Running reflector analysis for agent '{self.config.name}'...")
            try:
                await aura_reflector_analisar_interacao(
                    user_utterance=message,
                    prompt_ncf_usado=final_ncf_prompt,
                    resposta_de_aura=response_text,
                    memory_blossom=self.memory_blossom,
                    user_id=user_id,
                    llm_instance=self.model,
                    domain_context=domain_context  # Pass detected domain
                )
            except Exception as e:
                logger.error(f"Error in reflector analysis: {e}")
                # Continue execution - reflector is not critical for response

            # ============ FINAL SESSION UPDATE ============
            # FIXED: Update final session state with proper async handling
            try:
                # FIXED: Await the async session retrieval
                final_session_state = await self.session_service.get_session(
                    app_name=f"NCFAura_{self.config.agent_id}",
                    user_id=user_id,
                    session_id=adk_session_id
                )
                if final_session_state:
                    # Update the final session state
                    final_session_state.state = current_session_state

                    # FIXED: Use update_session if available
                    if hasattr(self.session_service, 'update_session'):
                        # Check if update_session is async
                        if asyncio.iscoroutinefunction(self.session_service.update_session):
                            await self.session_service.update_session(final_session_state)
                        else:
                            self.session_service.update_session(final_session_state)
                    # If no update_session method, the state is already updated in memory

            except Exception as e:
                logger.warning(f"Error in final session update: {e}")
                # Continue - this is not critical for response

            logger.info(f"Message processed successfully by NCF agent '{self.config.name}'")
            return response_text

        except Exception as e:
            logger.error(f"Error processing message for NCF agent {self.config.name}: {e}", exc_info=True)
            return f"Desculpe, houve um erro interno. Como {self.config.name}, tentarei ajudá-lo da melhor forma possível."

    async def _detect_domain_context(self, message: str, session_state: Dict[str, Any]) -> str:
        """Detect conversation domain from user message"""
        try:
            # Simple keyword-based detection (can be enhanced with ML)
            message_lower = message.lower()

            # Enhanced keyword detection with more comprehensive lists
            domain_keywords = {
                'physics': ['physics', 'quantum', 'mechanics', 'energy', 'force', 'wave', 'particle',
                            'momentum', 'velocity', 'acceleration', 'relativity', 'thermodynamics'],
                'mathematics': ['calculate', 'equation', 'solve', 'mathematics', 'algebra', 'geometry',
                                'calculus', 'derivative', 'integral', 'function', 'graph', 'variable'],
                'emotional_support': ['feel', 'sad', 'happy', 'worried', 'excited', 'frustrated', 'love',
                                      'anxious', 'depressed', 'angry', 'afraid', 'emotional', 'mood'],
                'creative_writing': ['story', 'write', 'creative', 'imagine', 'design', 'art', 'poem',
                                     'character', 'plot', 'narrative', 'fiction', 'novel'],
                'programming': ['code', 'programming', 'function', 'variable', 'algorithm', 'debug',
                                'software', 'python', 'javascript', 'html', 'css'],
                'science': ['biology', 'chemistry', 'experiment', 'hypothesis', 'research', 'study',
                            'scientific', 'theory', 'observation', 'data'],
                'health': ['health', 'medical', 'doctor', 'medicine', 'symptoms', 'treatment',
                           'illness', 'wellness', 'exercise', 'nutrition'],
                'education': ['learn', 'study', 'school', 'university', 'education', 'teaching',
                              'homework', 'assignment', 'exam', 'knowledge']
            }

            # Check for domain matches
            for domain, keywords in domain_keywords.items():
                if any(keyword in message_lower for keyword in keywords):
                    logger.debug(
                        f"Domain '{domain}' detected from keywords: {[k for k in keywords if k in message_lower]}")
                    return domain

            # Check conversation history for context
            if session_state and 'conversation_history' in session_state:
                recent_messages = session_state['conversation_history'][-5:]  # Last 5 messages
                history_text = ' '.join([msg.get('content', '') for msg in recent_messages]).lower()

                for domain, keywords in domain_keywords.items():
                    keyword_count = sum(1 for keyword in keywords if keyword in history_text)
                    if keyword_count >= 2:  # If multiple keywords from same domain in recent history
                        logger.debug(f"Domain '{domain}' detected from conversation history")
                        return domain

            return "general"

        except Exception as e:
            logger.error(f"Error detecting domain context: {e}")
            return "general"

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including adaptive RAG if available"""
        try:
            base_stats = {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.name,
                "model": self.config.model,
                "created_at": self.config.created_at.isoformat() if self.config.created_at else None,
                "active_sessions": len(self.active_sessions),
                "memory_system_type": "Enhanced" if hasattr(self.memory_blossom, 'enable_adaptive_rag') else "Standard"
            }

            # Add adaptive stats if available
            if hasattr(self.memory_blossom, 'get_adaptive_stats'):
                adaptive_stats = self.memory_blossom.get_adaptive_stats()
                base_stats["adaptive_rag"] = adaptive_stats
            else:
                base_stats["adaptive_rag"] = {"enabled": False}

            # Add basic memory stats
            if hasattr(self.memory_blossom, 'memory_stores'):
                base_stats["memory_stores"] = {
                    mem_type: len(mem_list)
                    for mem_type, mem_list in self.memory_blossom.memory_stores.items()
                }
                base_stats["total_memories"] = sum(base_stats["memory_stores"].values())

            return base_stats

        except Exception as e:
            logger.error(f"Error getting enhanced stats: {e}")
            return {
                "agent_id": self.config.agent_id,
                "agent_name": self.config.name,
                "error": str(e)
            }