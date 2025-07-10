# Aura Multi-Agent System with NCF (Narrative Context Framing)

## 🌟 Overview

The Aura Multi-Agent System is an advanced AI platform that enables users to create and manage multiple AI agents, each with their own isolated memory system and sophisticated contextual understanding capabilities. Built with Narrative Context Framing (NCF) technology, these agents can maintain deep, contextual conversations with persistent memory and adaptive learning.

Online Demo: [httpaura-dahu.onrender.com/ ](https://aura-dahu.onrender.com/)

![image](https://github.com/user-attachments/assets/aac7e2b6-1df2-4325-9d0f-d6895da8dda2)



 - This framework works with OpenRouter + LiteLLM for LLM Inference (Default is openrouter/openai/gpt-4o-mini") 
 - Embedding models runs locally thru sentence-transformers 

## 🚀 Key Features

### Core Capabilities
- **NCF (Narrative Context Framing)**: Advanced conversation architecture that builds narrative understanding over time
   ![image](https://github.com/user-attachments/assets/b5c6d96d-68bd-4267-9f9c-ecfc37c26689)

- **Isolated Memory Systems**: Each agent has its own MemoryBlossom instance for personalized memory management
- **RAG (Retrieval-Augmented Generation)**: Context-aware memory retrieval for relevant responses
- **Reflective Analysis**: Automatic interaction analysis and memory formation
- **Adaptive Learning**: Optional enhanced memory system with domain-aware clustering and performance tracking

### User Features
- **Multi-Agent Management**: Create and manage multiple AI companions with unique personalities
- **Web Interface**: Beautiful, responsive UI for agent creation and chat
- **Memory Export/Import**: Backup and restore agent memories in JSON, CSV, or ZIP formats
- **Profile Editing**: Customize agent names, personalities, and behaviors
- **Memory Analytics**: Detailed insights into agent memory patterns and usage
- **Session Persistence**: Conversations are saved and can be resumed later
![image](https://github.com/user-attachments/assets/a0cf4739-202b-449e-94ea-5b583da65a02)
## 📋 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git

## 🛠️ Installation

### 1. Clone the repository


### 2. Create a virtual environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp TEMPLATE.env .env
```

### 5. Edit `.env` file
```bash
# Add your OpenRouter API key (REQUIRED)
OPENROUTER_API_KEY="sk-or-..."

# Optionally customize other settings
API_PORT=8000
JWT_SECRET_KEY=your-secure-secret-key
```

### 6. Initialize the database
```bash
python init_db.py
```

## 🚀 Quick Start

### 1. Start the API server
```bash
python main_app.py
```

### 2. Access the web interface
Open your browser and navigate to:
```
http://localhost:8000
```

### 3. Create an account
- Click "Register" and create your account
- Login with your credentials

### 4. Create your first agent
- Click the "+" button to create a new agent
- Give it a name (e.g., "Luna", "Sage")
- Add a short description (persona)
- Write a detailed personality description
- Click "Create Aura"

### 5. Start chatting!
- Click on your agent to open the chat
- Your agent will remember conversations and learn from them
- Each message is processed through the NCF system for contextual understanding

## 🏗️ Architecture

### Project Structure
```
aura-multi-agent/
├── api/
│   └── routes.py                # FastAPI routes for agent management
├── memory_system/
│   ├── memory_blossom.py        # Core memory system
│   ├── memory_models.py         # Memory data structures
│   ├── memory_connector.py      # Memory relationship management
│   └── embedding_utils.py       # Text embedding utilities
├── database/
│   └── models.py                # SQLAlchemy database models
├── a2a_wrapper/
│   ├── main.py                  # A2A protocol wrapper
│   └── models.py                # A2A data models
├── agent_manager.py             # NCF-enabled agent management
├── ncf_processing.py            # NCF core logic
├── enhanced_memory_system.py    # Optional adaptive RAG (see below)
├── main_app.py                  # Main application entry
└── frontend/
    └── index.html               # Web interface
```

### Core Components

#### 1. **Agent Manager** (`agent_manager.py`)
Manages agent lifecycle with NCF capabilities:
- Creates agents with isolated memory systems
- Handles agent configuration and persistence
- Provides NCF-powered conversation processing

#### 2. **NCF Processing** (`ncf_processing.py`)
The brain of the system:
- **Narrative Foundation**: Builds long-term understanding
- **RAG Retrieval**: Fetches relevant memories
- **Chat History**: Maintains conversation context
- **Reflector**: Analyzes interactions for memory creation

#### 3. **Memory System** (`memory_system/`)
Sophisticated memory management:
- Multiple memory types (Explicit, Emotional, Procedural, etc.)
- Embedding-based similarity search
- Memory decay and salience tracking
- Connection graph between memories

#### 4. **Enhanced Memory** (`enhanced_memory_system.py`) 
Advanced features that build upon the base memory system:
- Adaptive concept clustering
- Domain specialization
- Performance-weighted retrieval
- GenLang-inspired architecture


### 5. **Use Case Flexibility**
- **Simple Chatbots**: Base system is sufficient
- **Domain Experts**: Enhanced system provides specialized knowledge tracking
- **Long-term Companions**: Enhanced system better for evolving relationships

### 6. **Graceful Degradation**
The system automatically falls back to base functionality if enhanced features fail:
```python
try:
    from enhanced_memory_system import EnhancedMemoryBlossom
    # Use enhanced features
except ImportError:
    # Fallback to standard MemoryBlossom
    logger.warning("Enhanced features not available")
```

## 📚 API Documentation

### Authentication Endpoints

#### Register New User
```bash
POST /auth/register
Content-Type: application/json

{
  "username": "user123",
  "password": "secure_password",
  "email": "user@example.com"
}
```

#### Login
```bash
POST /auth/login
Content-Type: application/json

{
  "username": "user123",
  "password": "secure_password"
}
```

### Agent Management Endpoints

#### Create Agent
```bash
POST /agents/create
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Luna",
  "persona": "A wise philosopher",
  "detailed_persona": "Luna is thoughtful and contemplative...",
  "model": "openrouter/openai/gpt-4o-mini"  # Optional
}
```

#### List Your Agents
```bash
GET /agents/list
Authorization: Bearer <token>
```

#### Get Agent Details
```bash
GET /agents/{agent_id}
Authorization: Bearer <token>
```

#### Update Agent Profile
```bash
PUT /agents/{agent_id}/profile
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Luna Updated",
  "persona": "A wise and witty philosopher",
  "detailed_persona": "Updated personality description..."
}
```

#### Delete Agent
```bash
DELETE /agents/{agent_id}
Authorization: Bearer <token>
```

### Chat Endpoints

#### Send Message
```bash
POST /agents/{agent_id}/chat
Authorization: Bearer <token>
Content-Type: application/json

{
  "message": "Hello, let's discuss philosophy",
  "session_id": "optional_session_id"
}
```

### Memory Management Endpoints

#### Get Agent Memories
```bash
GET /agents/{agent_id}/memories?memory_type=Explicit&limit=50
Authorization: Bearer <token>
```

#### Search Memories
```bash
POST /agents/{agent_id}/memories/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "philosophy discussions",
  "memory_types": ["Explicit", "Emotional"],
  "limit": 10
}
```

#### Export Memories
```bash
# Export as JSON
GET /agents/{agent_id}/memories/export?format=json

# Export as CSV
GET /agents/{agent_id}/memories/export?format=csv

# Export as ZIP (includes both JSON and CSV)
GET /agents/{agent_id}/memories/export?format=zip
```

#### Upload Memories
```bash
POST /agents/{agent_id}/memories/upload
Authorization: Bearer <token>
Content-Type: application/json

{
  "memories": [
    {
      "content": "User loves discussing Stoic philosophy",
      "memory_type": "Explicit",
      "emotion_score": 0.8,
      "initial_salience": 0.9
    }
  ],
  "overwrite_existing": false
}
```

#### Upload Memory File
```bash
POST /agents/{agent_id}/memories/upload/file
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: <memory_export.json or memory_export.csv>
overwrite_existing: false
```

#### Get Memory Analytics
```bash
GET /agents/{agent_id}/memories/analytics
Authorization: Bearer <token>
```

### Enhanced Features (Optional)

#### Check Adaptive Stats
```bash
GET /agents/{agent_id}/adaptive-stats
Authorization: Bearer <token>
```

#### Upgrade to Enhanced Memory
```bash
POST /agents/{agent_id}/upgrade-adaptive-rag
Authorization: Bearer <token>
```

## 🔧 Customization Guide

### Creating Custom Memory Types

1. **Define the memory type in `memory_system/memory_blossom.py`:**
```python
def __init__(self):
    self.memory_stores["CustomType"] = []
```

2. **Add embedding model in `embedding_utils.py`:**
```python
EMBEDDING_MODELS_CONFIG = {
    "CustomType": "sentence-transformers/your-model",
    # ... other types
}
```

3. **Update the reflector in `ncf_processing.py`:**
```python
# Add CustomType to memory type selection logic
memory_types = ["Explicit", "Emotional", "CustomType", ...]
```

### Modifying Agent Behavior

#### Change Core Instructions
Edit `ncf_processing.py`:
```python
NCF_AGENT_INSTRUCTION = """
You are an AI assistant with the following traits:
- Your custom personality traits
- Your specialized knowledge domains
- Your communication style
...
"""
```

#### Adjust Memory Retrieval
```python
# In agent_manager.py, modify retrieval parameters
memories = self.memory_blossom.retrieve_memories(
    query=message,
    top_k=10,  # Increase for more context
    min_similarity_threshold=0.2,  # Lower for broader matches
    apply_criticality=True  # Enable/disable criticality scoring
)
```

### Enhancing the Reflector

The reflector analyzes conversations to create memories. Customize in `ncf_processing.py`:

```python
async def aura_reflector_analisar_interacao():
    # Add custom memory creation rules
    if "important" in user_utterance.lower():
        initial_salience = 0.9  # Boost importance
    
    # Domain-specific analysis
    if detected_domain == "technical":
        # Create procedural memories for technical discussions
        memory_type = "Procedural"
```

### Frontend Customization

#### Change Theme Colors
Edit `frontend/index.html`:
```css
:root {
    --primary-color: #667eea;  /* Change primary color */
    --background: #0a0a0a;     /* Change background */
    --text-color: #e0e0e0;     /* Change text color */
}
```

#### Add New UI Features
```javascript
// Add custom functionality
function addCustomFeature() {
    // Your code here
}

// Hook into existing events
document.addEventListener('DOMContentLoaded', function() {
    addCustomFeature();
});
```

## 🚀 Advanced Configuration

### Memory System Tuning

#### Adjust Decay Rates
```python
# In memory_models.py
def apply_decay(self, rate: float = 0.01):
    # Modify decay algorithm
    # Lower rate = slower forgetting
```

#### Customize Salience Calculation
```python
# In memory_models.py
def get_effective_salience(self) -> float:
    # Add custom factors
    recency_boost = self.calculate_recency_boost()
    importance_factor = self.custom_importance_score()
    return base_salience * recency_boost * importance_factor
```

### Performance Optimization

#### 1. **Use Faster Embedding Models**
```python
# In embedding_utils.py
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast
# Alternative: "all-mpnet-base-v2"  # More accurate but slower
```

#### 2. **Implement Caching**
```python
# Add to memory_blossom.py
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_retrieve_memories(self, query_hash, **kwargs):
    return self.retrieve_memories(**kwargs)
```

#### 3. **Batch Processing**
```python
# Process multiple messages efficiently
async def batch_process_messages(messages: List[str]):
    embeddings = model.encode(messages, batch_size=32)
    # Process in parallel
```

### A2A Protocol Integration

#### Enable A2A Wrapper
```bash
# Start A2A service
python -m a2a_wrapper.main

# Or for multi-agent A2A
python -m a2a_wrapper.multi_agent_a2a_wrapper
```

#### Register with AIRA Hub
```python
POST /register-to-hub/{agent_id}
{
    "hub_url": "https://hub.aira.com"
}
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **"OPENROUTER_API_KEY not set"**
```bash
# Check if key is set
echo $OPENROUTER_API_KEY

# Set in .env file
OPENROUTER_API_KEY="sk-or-v1-..."

# Or export directly
export OPENROUTER_API_KEY="sk-or-v1-..."
```

#### 2. **Database Connection Errors**
```bash
# Reset database
rm aura_agents.db
python init_db.py

# Check permissions
chmod 664 aura_agents.db
```

#### 3. **Memory Not Persisting**
```bash
# Check directory permissions
chmod -R 755 agent_storage/

# Verify save is called
# Look for "Saved memories to" in logs
```

#### 4. **Slow Response Times**
- Reduce memory retrieval count: `top_k=3`
- Disable enhanced features temporarily
- Use faster models
- Enable debug logging to identify bottlenecks

#### 5. **Import Errors**
```bash
# Ensure you're in the project root
cd /path/to/aura-multi-agent

# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Debug Mode

#### Enable Detailed Logging
```bash
# In .env
LOG_LEVEL=DEBUG

# Or via environment
export LOG_LEVEL=DEBUG
python main_app.py
```

#### Monitor Memory Usage
```python
# Add to any module
import psutil
import os

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.debug(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
```

## 🔐 Security Best Practices

### 1. **Authentication & Authorization**
```python
# Always use strong JWT secrets
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", generate_strong_secret())

# Implement rate limiting
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
```

### 2. **Input Validation**
```python
# Sanitize user inputs
from bleach import clean

def sanitize_input(text: str) -> str:
    return clean(text, tags=[], strip=True)
```

### 3. **Data Encryption**
```python
# Encrypt sensitive memories
from cryptography.fernet import Fernet

def encrypt_memory_content(content: str, key: bytes) -> str:
    f = Fernet(key)
    return f.encrypt(content.encode()).decode()
```

### 4. **API Security**
```nginx
# Use HTTPS in production (nginx config)
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 5. **Backup Strategy**
```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/aura"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup database
cp aura_agents.db "$BACKUP_DIR/aura_agents_$DATE.db"

# Backup agent storage
tar -czf "$BACKUP_DIR/agent_storage_$DATE.tar.gz" agent_storage/

# Keep only last 30 days
find "$BACKUP_DIR" -mtime +30 -delete
```

## 📊 Monitoring & Analytics

### System Metrics
```python
# Add to routes.py
@app.get("/metrics")
async def get_system_metrics():
    return {
        "total_agents": count_all_agents(),
        "active_sessions": count_active_sessions(),
        "memory_usage": get_memory_stats(),
        "api_calls_today": get_api_call_count()
    }
```

### Agent Performance Tracking
```python
# Track conversation quality
def track_conversation_metrics(agent_id: str, session_id: str):
    metrics = {
        "response_time": measure_response_time(),
        "memory_retrievals": count_memory_calls(),
        "user_satisfaction": estimate_satisfaction(),
        "conversation_depth": calculate_depth()
    }
    save_metrics(agent_id, session_id, metrics)
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) for robust API development
- [SQLAlchemy](https://www.sqlalchemy.org/) for database management
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- Google ADK for agent development framework
- Inspired by cognitive science and memory research

## 📞 Support

- **Documentation**: Check this README and code comments
- **Issues**: Open a GitHub issue for bugs or features
- **Discussions**: Use GitHub Discussions for questions
- **Email**: support@example.com (replace with actual)


---
