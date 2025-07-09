# ==================== api/routes.py
"""
Enhanced API routes with memory management and agent editing capabilities
"""

from fastapi import FastAPI, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any, Union
import jwt
from datetime import datetime, timedelta
import bcrypt
import os
from pathlib import Path
import sys
import logging
import json
import io
import zipfile

# Import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_manager import AgentManager  # Now using NCF-enabled AgentManager
from database.models import AgentRepository, User

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Initialize FastAPI app
app = FastAPI(
    title="Aura Multi-Agent API (NCF-Enabled)",
    description="API for creating and managing multiple NCF-powered Aura AI agents with advanced memory and context capabilities",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CORRECTED INITIALIZATION WITH ABSOLUTE PATHS ---
# Determine project root dynamically.
# If routes.py is in 'api/', and 'agent_storage' & 'aura_agents.db' are at the project root level:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_STORAGE_PATH = PROJECT_ROOT / "agent_storage"
DATABASE_FILE_PATH = PROJECT_ROOT / "aura_agents.db"

# Ensure the agent storage directory exists
AGENT_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
logger.info(f"Using AGENT_STORAGE_PATH: {AGENT_STORAGE_PATH.resolve()}")
logger.info(f"Using DATABASE_FILE_PATH: {DATABASE_FILE_PATH.resolve()}")

# Initialize services with NCF-enabled AgentManager using absolute paths
agent_manager = AgentManager(base_storage_path=str(AGENT_STORAGE_PATH))  # Convert Path object to string
db_repo = AgentRepository(db_url=f"sqlite:///{str(DATABASE_FILE_PATH)}")  # Convert Path object to string

# Security
security = HTTPBearer()


# --- Pydantic Models ---
class UserRegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


class CreateAgentRequest(BaseModel):
    name: str
    persona: str
    detailed_persona: str
    model: Optional[str] = None
    is_public: Optional[bool] = False


class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    persona: Optional[str] = None
    detailed_persona: Optional[str] = None
    avatar_url: Optional[str] = None
    is_public: Optional[bool] = None
    settings: Optional[dict] = None


class EnhancedUpdateAgentRequest(BaseModel):
    """Enhanced agent update request with all editable fields"""
    name: Optional[str] = None
    persona: Optional[str] = None  # Short description
    detailed_persona: Optional[str] = None  # Detailed personality
    avatar_url: Optional[str] = None
    is_public: Optional[bool] = None
    settings: Optional[dict] = None
    model: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    persona: str
    detailed_persona: str
    created_at: str
    is_public: bool
    owner_username: Optional[str] = None
    capabilities: List[str] = ["ncf", "memory", "narrative_foundation", "rag", "reflector"]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    ncf_enabled: bool = True


class MemorySearchRequest(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    limit: Optional[int] = 10


class MemoryUploadRequest(BaseModel):
    """Request model for uploading memories"""
    memories: List[Dict[str, Any]]
    overwrite_existing: Optional[bool] = False
    validate_format: Optional[bool] = True


class MemoryExportResponse(BaseModel):
    """Response model for memory export"""
    agent_id: str
    agent_name: str
    export_timestamp: str
    total_memories: int
    memory_types: List[str]
    memories: List[Dict[str, Any]]


class BulkMemoryUploadResponse(BaseModel):
    """Response model for bulk memory upload"""
    agent_id: str
    agent_name: str
    upload_timestamp: str
    total_uploaded: int
    successful_uploads: int
    failed_uploads: int
    errors: List[str]
    memory_types_added: List[str]


# --- Authentication Helpers ---
def create_access_token(user_id: str, username: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode = {
        "user_id": user_id,
        "username": username,
        "exp": expire
    }
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Verify JWT token and return user info"""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            "user_id": payload["user_id"],
            "username": payload["username"]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# --- User Authentication Routes ---
@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register_user(request: UserRegisterRequest):
    """Register a new user"""
    # Check if user exists
    with db_repo.SessionLocal() as session:
        existing_user = session.query(User).filter(
            (User.email == request.email) | (User.username == request.username)
        ).first()

        if existing_user:
            if existing_user.email == request.email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )

        # Hash password
        password_hash = bcrypt.hashpw(request.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        # Create user
        new_user = User(
            email=request.email,
            username=request.username,
            password_hash=password_hash
        )

        session.add(new_user)
        session.commit()
        session.refresh(new_user)

        # Create access token
        access_token = create_access_token(new_user.id, new_user.username)

        return {
            "user_id": new_user.id,
            "username": new_user.username,
            "email": new_user.email,
            "access_token": access_token,
            "token_type": "bearer",
            "message": "Welcome! You can now create NCF-powered Aura agents with advanced memory and contextual understanding."
        }


@app.post("/auth/login")
async def login_user(request: UserLoginRequest):
    """Login user and return access token"""
    with db_repo.SessionLocal() as session:
        user = session.query(User).filter(User.username == request.username).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Verify password
        if not bcrypt.checkpw(request.password.encode('utf-8'), user.password_hash.encode('utf-8')):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Create access token
        access_token = create_access_token(user.id, user.username)

        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "access_token": access_token,
            "token_type": "bearer"
        }


@app.get("/auth/me")
async def get_current_user(current_user: dict = Depends(verify_token)):
    """Get current user information"""
    with db_repo.SessionLocal() as session:
        user = session.query(User).filter(User.id == current_user["user_id"]).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat(),
            "agent_capabilities": ["ncf", "memory", "narrative_foundation", "rag", "reflector"]
        }


# --- Agent Management Routes ---
@app.post("/agents/create", response_model=dict)
async def create_agent(request: CreateAgentRequest, current_user: dict = Depends(verify_token)):
    """Create a new NCF-enabled Aura agent with advanced capabilities"""
    try:
        # Create NCF-enabled agent in file system
        agent_id = agent_manager.create_agent(
            user_id=current_user["user_id"],
            name=request.name,
            persona=request.persona,
            detailed_persona=request.detailed_persona,
            model=request.model
        )

        # Also save to database
        db_agent = db_repo.create_agent(
            agent_id=agent_id,
            user_id=current_user["user_id"],
            name=request.name,
            persona=request.persona,
            detailed_persona=request.detailed_persona,
            model=request.model,
            is_public=request.is_public
        )

        return {
            "agent_id": agent_id,
            "message": f"NCF-enabled agent '{request.name}' created successfully with advanced memory and contextual understanding",
            "capabilities": ["ncf", "memory", "narrative_foundation", "rag", "reflector", "isolated_memory_system"],
            "features": [
                "Narrative Context Framing for deep conversations",
                "Isolated MemoryBlossom for personalized memory",
                "RAG (Retrieval-Augmented Generation) capabilities",
                "Automatic reflection and memory formation",
                "Context-aware responses with conversation history"
            ]
        }
    except Exception as e:
        logger.error(f"Error creating NCF agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create NCF-enabled agent: {str(e)}"
        )


@app.get("/agents/list", response_model=List[AgentResponse])
async def list_agents(current_user: dict = Depends(verify_token)):
    """List all NCF-enabled agents for the authenticated user"""
    configs = agent_manager.list_user_agents(current_user["user_id"])

    return [
        AgentResponse(
            agent_id=config.agent_id,
            name=config.name,
            persona=config.persona,
            detailed_persona=config.detailed_persona,
            created_at=config.created_at.isoformat() if config.created_at else datetime.utcnow().isoformat(),
            is_public=False,
            owner_username=current_user["username"],
            capabilities=["ncf", "memory", "narrative_foundation", "rag", "reflector"]
        )
        for config in configs
    ]


@app.get("/agents/public", response_model=List[AgentResponse])
async def list_public_agents():
    """List all public NCF-enabled agents"""
    with db_repo.SessionLocal() as session:
        from sqlalchemy.orm import joinedload
        public_agents = session.query(db_repo.Agent).filter(
            db_repo.Agent.is_public == 1
        ).options(joinedload(db_repo.Agent.user)).all()

        return [
            AgentResponse(
                agent_id=agent.id,
                name=agent.name,
                persona=agent.persona,
                detailed_persona=agent.detailed_persona,
                created_at=agent.created_at.isoformat(),
                is_public=True,
                owner_username=agent.user.username if agent.user else None,
                capabilities=["ncf", "memory", "narrative_foundation", "rag", "reflector"]
            )
            for agent in public_agents
        ]


@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, current_user: dict = Depends(verify_token)):
    """Get details of a specific NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check if user owns the agent or it's public
    config = agent.config
    if config.user_id != current_user["user_id"]:
        # Check if agent is public in database
        db_agent = db_repo.get_agent(agent_id)
        if not db_agent or not db_agent.is_public:
            raise HTTPException(status_code=403, detail="Access denied")

    return AgentResponse(
        agent_id=config.agent_id,
        name=config.name,
        persona=config.persona,
        detailed_persona=config.detailed_persona,
        created_at=config.created_at.isoformat() if config.created_at else datetime.utcnow().isoformat(),
        is_public=db_agent.is_public if db_agent else False,
        owner_username=current_user["username"],
        capabilities=["ncf", "memory", "narrative_foundation", "rag", "reflector"]
    )


@app.put("/agents/{agent_id}", response_model=dict)
async def update_agent(agent_id: str, request: UpdateAgentRequest, current_user: dict = Depends(verify_token)):
    """Update an existing NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Update agent configuration
    config = agent.config
    if request.name is not None:
        config.name = request.name
    if request.persona is not None:
        config.persona = request.persona
    if request.detailed_persona is not None:
        config.detailed_persona = request.detailed_persona
    if request.settings is not None:
        config.settings.update(request.settings)

    # Save updated configuration
    agent_manager._save_agent_config(config)

    # Update in database if needed
    with db_repo.SessionLocal() as session:
        db_agent = session.query(db_repo.Agent).filter(db_repo.Agent.id == agent_id).first()
        if db_agent:
            if request.name is not None:
                db_agent.name = request.name
            if request.persona is not None:
                db_agent.persona = request.persona
            if request.detailed_persona is not None:
                db_agent.detailed_persona = request.detailed_persona
            if request.avatar_url is not None:
                db_agent.avatar_url = request.avatar_url
            if request.is_public is not None:
                db_agent.is_public = 1 if request.is_public else 0
            if request.settings is not None:
                db_agent.settings = request.settings

            session.commit()

    return {"message": "NCF-enabled agent updated successfully"}


@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, current_user: dict = Depends(verify_token)):
    """Delete an NCF-enabled agent"""
    success = agent_manager.delete_agent(agent_id, current_user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")

    # Also delete from database
    db_repo.delete_agent(agent_id, current_user["user_id"])

    return {"message": "NCF-enabled agent and its isolated memory system deleted successfully"}


# --- Enhanced Agent Profile Management ---
@app.put("/agents/{agent_id}/profile", response_model=dict)
async def update_agent_profile(
        agent_id: str,
        request: EnhancedUpdateAgentRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Enhanced agent profile update with comprehensive editing capabilities
    Allows editing: name, persona (short description), detailed_persona, model, and settings
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Track what was updated
        updated_fields = []

        # Update agent configuration
        config = agent.config

        if request.name is not None:
            old_name = config.name
            config.name = request.name
            updated_fields.append(f"name: '{old_name}' → '{request.name}'")

        if request.persona is not None:
            old_persona = config.persona
            config.persona = request.persona
            updated_fields.append(f"persona: '{old_persona[:50]}...' → '{request.persona[:50]}...'")

        if request.detailed_persona is not None:
            old_detailed = config.detailed_persona
            config.detailed_persona = request.detailed_persona
            updated_fields.append(f"detailed_persona: '{old_detailed[:50]}...' → '{request.detailed_persona[:50]}...'")

        if request.model is not None:
            old_model = config.model
            config.model = request.model
            updated_fields.append(f"model: '{old_model}' → '{request.model}'")

        if request.settings is not None:
            config.settings.update(request.settings)
            updated_fields.append(f"settings: {list(request.settings.keys())}")

        # Save updated configuration to file system
        agent_manager._save_agent_config(config)

        # Update in database if it exists
        try:
            with db_repo.SessionLocal() as session:
                db_agent = session.query(db_repo.Agent).filter(db_repo.Agent.id == agent_id).first()
                if db_agent:
                    if request.name is not None:
                        db_agent.name = request.name
                    if request.persona is not None:
                        db_agent.persona = request.persona
                    if request.detailed_persona is not None:
                        db_agent.detailed_persona = request.detailed_persona
                    if request.avatar_url is not None:
                        db_agent.avatar_url = request.avatar_url
                    if request.is_public is not None:
                        db_agent.is_public = 1 if request.is_public else 0
                    if request.settings is not None:
                        db_agent.settings = request.settings

                    session.commit()
                    updated_fields.append("database record")
        except Exception as db_error:
            logger.warning(f"Database update failed: {db_error}")
            # Continue execution - file system update is more critical

        # Add memory about the profile update
        try:
            update_memory = {
                "content": f"Agent profile updated. Changed: {', '.join(updated_fields)}",
                "memory_type": "system_update",
                "emotion_score": 0.0,
                "initial_salience": 0.3,
                "custom_metadata": {
                    "source": "profile_update",
                    "updated_by": current_user["user_id"],
                    "timestamp": datetime.now().isoformat(),
                    "updated_fields": updated_fields
                }
            }

            agent.memory_blossom.add_memory(**update_memory)
            agent.memory_blossom.save_memories()
        except Exception as memory_error:
            logger.warning(f"Failed to add update memory: {memory_error}")

        return {
            "message": "Agent profile updated successfully",
            "agent_id": agent_id,
            "agent_name": config.name,
            "updated_fields": updated_fields,
            "timestamp": datetime.now().isoformat(),
            "ncf_enabled": True,
            "capabilities": ["ncf", "memory", "narrative_foundation", "rag", "reflector"]
        }

    except Exception as e:
        logger.error(f"Error updating agent profile {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Profile update failed: {str(e)}")


@app.get("/agents/{agent_id}/profile", response_model=dict)
async def get_agent_profile(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Get comprehensive agent profile information
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check access permissions
    config = agent.config
    if config.user_id != current_user["user_id"]:
        # Check if agent is public
        try:
            db_agent = db_repo.get_agent(agent_id)
            if not db_agent or not db_agent.is_public:
                raise HTTPException(status_code=403, detail="Access denied")
        except:
            raise HTTPException(status_code=403, detail="Access denied")

    # Get memory statistics
    memory_stats = {
        "total_memories": sum(len(m) for m in agent.memory_blossom.memory_stores.values()),
        "memory_types": list(agent.memory_blossom.memory_stores.keys()),
        "memory_breakdown": {
            mem_type: len(mem_list)
            for mem_type, mem_list in agent.memory_blossom.memory_stores.items()
        }
    }

    return {
        "agent_id": config.agent_id,
        "name": config.name,
        "persona": config.persona,
        "detailed_persona": config.detailed_persona,
        "model": config.model,
        "created_at": config.created_at.isoformat() if config.created_at else None,
        "settings": config.settings or {},
        "memory_stats": memory_stats,
        "ncf_enabled": True,
        "capabilities": ["ncf", "memory", "narrative_foundation", "rag", "reflector"],
        "owner_id": config.user_id,
        "is_owner": config.user_id == current_user["user_id"]
    }


@app.get("/agents/{agent_id}/adaptive-stats")
async def get_agent_adaptive_stats(agent_id: str, current_user: dict = Depends(verify_token)):
    """Get adaptive RAG statistics for an agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        adaptive_stats = agent.memory_blossom.get_adaptive_stats()
        return {
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "adaptive_stats": adaptive_stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting adaptive stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get adaptive stats")


@app.post("/agents/{agent_id}/upgrade-adaptive-rag")
async def upgrade_agent_adaptive_rag(agent_id: str, current_user: dict = Depends(verify_token)):
    """Upgrade an existing agent to use Enhanced MemoryBlossom"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        from enhanced_memory_system import upgrade_agent_to_adaptive_rag
        success = upgrade_agent_to_adaptive_rag(agent, enable_adaptive_rag=True)

        if success:
            return {
                "message": f"Agent '{agent.config.name}' successfully upgraded to Enhanced MemoryBlossom",
                "adaptive_rag_enabled": True,
                "features_added": [
                    "Domain-aware clustering",
                    "Performance-weighted retrieval",
                    "Adaptive concept formation",
                    "Multi-layer memory system"
                ]
            }
        else:
            raise HTTPException(status_code=500, detail="Upgrade failed")

    except Exception as e:
        logger.error(f"Error upgrading agent: {e}")
        raise HTTPException(status_code=500, detail=f"Upgrade failed: {str(e)}")



# --- Chat Routes ---
# --- Chat Routes ---
@app.post("/agents/{agent_id}/chat", response_model=ChatResponse)
async def chat_with_agent(agent_id: str, request: ChatRequest, current_user: dict = Depends(verify_token)):
    """Chat with a specific NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check if user owns the agent or it's public
    if agent.config.user_id != current_user["user_id"]:
        db_agent = db_repo.get_agent(agent_id)
        if not db_agent or not db_agent.is_public:
            raise HTTPException(status_code=403, detail="Access denied")

    # Generate session ID if not provided
    session_id = request.session_id or f"session_{current_user['user_id']}_{datetime.utcnow().timestamp()}"

    try:
        # FIXED: Properly await the async process_message method
        response = await agent.process_message(
            user_id=current_user["user_id"],
            session_id=session_id,
            message=request.message
        )

        return ChatResponse(
            response=response,
            session_id=session_id,
            ncf_enabled=True
        )
    except Exception as e:
        logger.error(f"Error in NCF chat: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NCF chat error: {str(e)}"
        )

# --- Memory Management Routes ---
@app.get("/agents/{agent_id}/memories")
async def get_agent_memories(
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
        current_user: dict = Depends(verify_token)
):
    """Get memories for a specific NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Get memories from the agent's isolated memory blossom
    memories = []
    if memory_type:
        if memory_type in agent.memory_blossom.memory_stores:
            memories = agent.memory_blossom.memory_stores[memory_type][:limit]
    else:
        # Get all memories
        for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
            memories.extend(mem_list[:limit // len(agent.memory_blossom.memory_stores)])

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "total_memories": sum(len(m) for m in agent.memory_blossom.memory_stores.values()),
        "memory_types": list(agent.memory_blossom.memory_stores.keys()),
        "isolated_memory_system": True,
        "ncf_enabled": True,
        "memories": [mem.to_dict() for mem in memories[:limit]]
    }


@app.post("/agents/{agent_id}/memories/search")
async def search_agent_memories(
        agent_id: str,
        request: MemorySearchRequest,
        current_user: dict = Depends(verify_token)
):
    """Search memories for a specific NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Search memories using the agent's isolated memory system
    results = agent.memory_blossom.retrieve_memories(
        query=request.query,
        target_memory_types=request.memory_types,
        top_k=request.limit or 10
    )

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "query": request.query,
        "isolated_memory_system": True,
        "ncf_enabled": True,
        "results": [mem.to_dict() for mem in results]
    }


@app.delete("/agents/{agent_id}/memories/{memory_id}")
async def delete_memory(
        agent_id: str,
        memory_id: str,
        current_user: dict = Depends(verify_token)
):
    """Delete a specific memory from an NCF-enabled agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # Find and remove the memory from the agent's isolated memory system
    memory_found = False
    for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
        for i, mem in enumerate(mem_list):
            if mem.id == memory_id:
                mem_list.pop(i)
                memory_found = True
                agent.memory_blossom.save_memories()
                break
        if memory_found:
            break

    if not memory_found:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"message": "Memory deleted successfully from NCF-enabled agent"}


# --- Enhanced Memory Management Routes ---
@app.get("/agents/{agent_id}/memories/export", response_model=MemoryExportResponse)
async def export_agent_memories(
        agent_id: str,
        format: Optional[str] = "json",  # json, csv, or zip
        memory_types: Optional[str] = None,  # comma-separated list
        current_user: dict = Depends(verify_token)
):
    """
    Export all memories for a specific agent
    Supports JSON, CSV, and ZIP formats
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Get all memories from the agent's isolated memory system
        all_memories = []
        memory_types_filter = memory_types.split(",") if memory_types else None

        for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
            if memory_types_filter and mem_type not in memory_types_filter:
                continue

            for memory in mem_list:
                memory_dict = memory.to_dict()
                memory_dict['memory_type'] = mem_type  # Ensure type is included
                all_memories.append(memory_dict)

        export_data = {
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "export_timestamp": datetime.now().isoformat(),
            "total_memories": len(all_memories),
            "memory_types": list(agent.memory_blossom.memory_stores.keys()),
            "memories": all_memories
        }

        if format.lower() == "json":
            return MemoryExportResponse(**export_data)

        elif format.lower() == "csv":
            # Convert to CSV format
            try:
                import pandas as pd
            except ImportError:
                raise HTTPException(status_code=500, detail="pandas not installed. Please install: pip install pandas")

            if not all_memories:
                raise HTTPException(status_code=404, detail="No memories found to export")

            # Flatten memories for CSV
            flattened_memories = []
            for memory in all_memories:
                flat_memory = {
                    'id': memory.get('id'),
                    'content': memory.get('content'),
                    'memory_type': memory.get('memory_type'),
                    'emotion_score': memory.get('emotion_score', 0.0),
                    'coherence_score': memory.get('coherence_score', 0.5),
                    'novelty_score': memory.get('novelty_score', 0.5),
                    'salience': memory.get('salience', 0.5),
                    'created_at': memory.get('created_at'),
                    'accessed_count': memory.get('accessed_count', 0),
                    'last_accessed': memory.get('last_accessed'),
                    'custom_metadata': json.dumps(memory.get('custom_metadata', {}))
                }
                flattened_memories.append(flat_memory)

            df = pd.DataFrame(flattened_memories)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()

            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={agent.config.name}_memories.csv"}
            )

        elif format.lower() == "zip":
            # Create ZIP file with JSON + CSV
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add JSON file
                json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
                zip_file.writestr(f"{agent.config.name}_memories.json", json_content)

                # Add CSV file if memories exist
                if all_memories:
                    try:
                        import pandas as pd
                        flattened_memories = []
                        for memory in all_memories:
                            flat_memory = {
                                'id': memory.get('id'),
                                'content': memory.get('content'),
                                'memory_type': memory.get('memory_type'),
                                'emotion_score': memory.get('emotion_score', 0.0),
                                'coherence_score': memory.get('coherence_score', 0.5),
                                'novelty_score': memory.get('novelty_score', 0.5),
                                'salience': memory.get('salience', 0.5),
                                'created_at': memory.get('created_at'),
                                'accessed_count': memory.get('accessed_count', 0),
                                'last_accessed': memory.get('last_accessed'),
                                'custom_metadata': json.dumps(memory.get('custom_metadata', {}))
                            }
                            flattened_memories.append(flat_memory)

                        df = pd.DataFrame(flattened_memories)
                        csv_content = df.to_csv(index=False)
                        zip_file.writestr(f"{agent.config.name}_memories.csv", csv_content)
                    except ImportError:
                        logger.warning("pandas not available for CSV export in ZIP")

                # Add metadata file
                metadata = {
                    "agent_id": agent_id,
                    "agent_name": agent.config.name,
                    "persona": agent.config.persona,
                    "detailed_persona": agent.config.detailed_persona,
                    "export_timestamp": datetime.now().isoformat(),
                    "total_memories": len(all_memories),
                    "memory_types": list(agent.memory_blossom.memory_stores.keys())
                }
                zip_file.writestr("metadata.json", json.dumps(metadata, indent=2))

            zip_buffer.seek(0)

            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={agent.config.name}_memories.zip"}
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use 'json', 'csv', or 'zip'")

    except Exception as e:
        logger.error(f"Error exporting memories for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.post("/agents/{agent_id}/memories/upload", response_model=BulkMemoryUploadResponse)
async def upload_agent_memories(
        agent_id: str,
        request: MemoryUploadRequest,
        current_user: dict = Depends(verify_token)
):
    """
    Upload new memories to a specific agent
    Supports bulk upload with validation and error handling
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        successful_uploads = 0
        failed_uploads = 0
        errors = []
        memory_types_added = set()

        for i, memory_data in enumerate(request.memories):
            try:
                # Validate required fields
                if request.validate_format:
                    required_fields = ['content', 'memory_type']
                    missing_fields = [field for field in required_fields if field not in memory_data]
                    if missing_fields:
                        error_msg = f"Memory {i}: Missing required fields: {missing_fields}"
                        errors.append(error_msg)
                        failed_uploads += 1
                        continue

                # Set default values for optional fields
                memory_content = memory_data['content']
                memory_type = memory_data['memory_type']
                emotion_score = float(memory_data.get('emotion_score', 0.0))
                initial_salience = float(memory_data.get('initial_salience', 0.5))
                custom_metadata = memory_data.get('custom_metadata', {})

                # Add upload metadata
                custom_metadata.update({
                    "source": "user_upload",
                    "uploaded_by": current_user["user_id"],
                    "upload_timestamp": datetime.now().isoformat()
                })

                # Check for existing memory if not overwriting
                if not request.overwrite_existing:
                    existing_memories = agent.memory_blossom.memory_stores.get(memory_type, [])
                    content_exists = any(mem.content == memory_content for mem in existing_memories)
                    if content_exists:
                        error_msg = f"Memory {i}: Content already exists (use overwrite_existing=true to force)"
                        errors.append(error_msg)
                        failed_uploads += 1
                        continue

                # Add memory to agent's isolated memory system
                agent.memory_blossom.add_memory(
                    content=memory_content,
                    memory_type=memory_type,
                    emotion_score=emotion_score,
                    initial_salience=initial_salience,
                    custom_metadata=custom_metadata
                )

                memory_types_added.add(memory_type)
                successful_uploads += 1

            except Exception as e:
                error_msg = f"Memory {i}: Failed to upload - {str(e)}"
                errors.append(error_msg)
                failed_uploads += 1
                logger.error(f"Error uploading memory {i}: {e}")

        # Save all memories to disk
        if successful_uploads > 0:
            agent.memory_blossom.save_memories()

        return BulkMemoryUploadResponse(
            agent_id=agent_id,
            agent_name=agent.config.name,
            upload_timestamp=datetime.now().isoformat(),
            total_uploaded=len(request.memories),
            successful_uploads=successful_uploads,
            failed_uploads=failed_uploads,
            errors=errors,
            memory_types_added=list(memory_types_added)
        )

    except Exception as e:
        logger.error(f"Error uploading memories for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/agents/{agent_id}/memories/upload/file")
async def upload_memories_from_file(
        agent_id: str,
        file: UploadFile = File(...),
        overwrite_existing: bool = False,
        current_user: dict = Depends(verify_token)
):
    """
    Upload memories from a JSON or CSV file
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Read file content
        content = await file.read()

        if file.filename.endswith('.json'):
            # Parse JSON file
            try:
                data = json.loads(content.decode('utf-8'))

                # Handle different JSON structures
                if isinstance(data, dict) and 'memories' in data:
                    memories = data['memories']  # Export format
                elif isinstance(data, list):
                    memories = data  # Direct list format
                else:
                    raise HTTPException(status_code=400, detail="Invalid JSON format")

            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

        elif file.filename.endswith('.csv'):
            # Parse CSV file
            try:
                import pandas as pd
            except ImportError:
                raise HTTPException(status_code=500, detail="pandas not installed. Please install: pip install pandas")

            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
                memories = []

                for _, row in df.iterrows():
                    memory = {
                        'content': row.get('content'),
                        'memory_type': row.get('memory_type'),
                        'emotion_score': float(row.get('emotion_score', 0.0)),
                        'initial_salience': float(row.get('salience', 0.5)),
                    }

                    # Parse custom_metadata if it exists
                    if 'custom_metadata' in row and pd.notna(row['custom_metadata']):
                        try:
                            memory['custom_metadata'] = json.loads(row['custom_metadata'])
                        except:
                            memory['custom_metadata'] = {}

                    memories.append(memory)

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid CSV: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use .json or .csv")

        # Create upload request
        upload_request = MemoryUploadRequest(
            memories=memories,
            overwrite_existing=overwrite_existing,
            validate_format=True
        )

        # Use existing upload logic
        return await upload_agent_memories(agent_id, upload_request, current_user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")


# --- Memory Analytics ---
@app.get("/agents/{agent_id}/memories/analytics")
async def get_memory_analytics(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Get detailed analytics about agent's memory system
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        analytics = {
            "agent_id": agent_id,
            "agent_name": agent.config.name,
            "analysis_timestamp": datetime.now().isoformat(),
            "total_memories": 0,
            "memory_types": {},
            "emotion_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "salience_distribution": {"high": 0, "medium": 0, "low": 0},
            "recent_activity": {"last_7_days": 0, "last_30_days": 0},
            "top_memory_sources": {},
            "memory_timeline": []
        }

        all_memories = []

        # Collect all memories
        for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
            analytics["memory_types"][mem_type] = len(mem_list)
            all_memories.extend([(mem, mem_type) for mem in mem_list])

        analytics["total_memories"] = len(all_memories)

        # Analyze memories
        for memory, mem_type in all_memories:
            memory_dict = memory.to_dict()

            # Emotion analysis
            emotion_score = memory_dict.get('emotion_score', 0.0)
            if emotion_score > 0.1:
                analytics["emotion_distribution"]["positive"] += 1
            elif emotion_score < -0.1:
                analytics["emotion_distribution"]["negative"] += 1
            else:
                analytics["emotion_distribution"]["neutral"] += 1

            # Salience analysis
            salience = memory_dict.get('salience', 0.5)
            if salience > 0.7:
                analytics["salience_distribution"]["high"] += 1
            elif salience > 0.3:
                analytics["salience_distribution"]["medium"] += 1
            else:
                analytics["salience_distribution"]["low"] += 1

            # Source analysis
            metadata = memory_dict.get('custom_metadata', {})
            source = metadata.get('source', 'unknown')
            analytics["top_memory_sources"][source] = analytics["top_memory_sources"].get(source, 0) + 1

            # Timeline analysis (if created_at exists)
            created_at = memory_dict.get('created_at')
            if created_at:
                try:
                    if isinstance(created_at, str):
                        created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    else:
                        created_date = created_at

                    days_ago = (datetime.now() - created_date.replace(tzinfo=None)).days

                    if days_ago <= 7:
                        analytics["recent_activity"]["last_7_days"] += 1
                    if days_ago <= 30:
                        analytics["recent_activity"]["last_30_days"] += 1

                except:
                    pass

        return analytics

    except Exception as e:
        logger.error(f"Error generating memory analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analytics generation failed: {str(e)}")


# --- Helper Endpoints ---
@app.get("/agents/{agent_id}/memory-types")
async def get_agent_memory_types(
        agent_id: str,
        current_user: dict = Depends(verify_token)
):
    """
    Get list of memory types available for this agent
    """
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Verify ownership
    if agent.config.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    memory_types = {}
    for mem_type, mem_list in agent.memory_blossom.memory_stores.items():
        memory_types[mem_type] = {
            "count": len(mem_list),
            "description": f"Memories of type '{mem_type}'"
        }

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "memory_types": memory_types,
        "total_types": len(memory_types)
    }


# --- NCF Information Routes ---
@app.get("/agents/{agent_id}/ncf-status")
async def get_ncf_status(agent_id: str, current_user: dict = Depends(verify_token)):
    """Get NCF capabilities status for a specific agent"""
    agent = agent_manager.get_agent_instance(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check access permissions
    if agent.config.user_id != current_user["user_id"]:
        db_agent = db_repo.get_agent(agent_id)
        if not db_agent or not db_agent.is_public:
            raise HTTPException(status_code=403, detail="Access denied")

    return {
        "agent_id": agent_id,
        "agent_name": agent.config.name,
        "ncf_enabled": True,
        "capabilities": {
            "narrative_foundation": True,
            "rag_retrieval": True,
            "reflector_analysis": True,
            "isolated_memory_system": True,
            "contextual_prompting": True,
            "conversation_history_tracking": True,
            "memory_types_supported": ["Explicit", "Emotional", "Procedural", "Flashbulb", "Liminal", "Generative"]
        },
        "memory_statistics": {
            "total_memories": sum(len(m) for m in agent.memory_blossom.memory_stores.values()),
            "memory_stores": {mem_type: len(mem_list) for mem_type, mem_list in
                              agent.memory_blossom.memory_stores.items()},
            "memory_persistence_path": agent.config.memory_path
        },
        "model_info": {
            "model": agent.config.model,
            "llm_instance": "LiteLlm"
        }
    }


# --- Frontend Routes ---
# Serve the frontend HTML
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


    @app.get("/")
    async def read_index():
        index_path = frontend_path / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        else:
            return {"message": "Aura Multi-Agent API (NCF-Enabled) is running. Frontend not found."}
else:
    @app.get("/")
    async def read_root():
        return {
            "message": "Aura Multi-Agent API (NCF-Enabled) is running",
            "version": "2.0.0",
            "ncf_features": [
                "Narrative Context Framing for every agent",
                "Isolated MemoryBlossom per agent",
                "RAG (Retrieval-Augmented Generation)",
                "Automatic reflection and memory formation",
                "Advanced contextual understanding"
            ],
            "endpoints": {
                "auth": "/auth/register, /auth/login, /auth/me",
                "agents": "/agents/create, /agents/list, /agents/{agent_id}",
                "agent_profile": "/agents/{agent_id}/profile",
                "chat": "/agents/{agent_id}/chat",
                "memories": "/agents/{agent_id}/memories",
                "memory_export": "/agents/{agent_id}/memories/export",
                "memory_upload": "/agents/{agent_id}/memories/upload",
                "memory_analytics": "/agents/{agent_id}/memories/analytics",
                "ncf": "/agents/{agent_id}/ncf-status"
            }
        }


# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "agent_manager": "initialized_with_ncf",
            "ncf_capabilities": "enabled"
        },
        "version": "2.0.0",
        "ncf_enabled": True,
        "enhanced_features": [
            "memory_export_import",
            "agent_profile_editing",
            "memory_analytics",
            "bulk_memory_operations"
        ]
    }


# --- Error Handlers ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "ncf_enabled": True
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "ncf_enabled": True
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)