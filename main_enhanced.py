"""
Law GPT - Enhanced Production Backend with Multilingual RAG
Complete version with semantic search and improved performance
"""

import os
import json
import logging
import hashlib
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
from datetime import datetime
import redis

# Import enhanced components
from enhanced_multilingual_rag import EnhancedMultilingualRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIRECTORY = "data"
CORS_ORIGINS = ["*"]

# Remote dataset configuration
KANOON_REMOTE_REPO = "Nivedh12121/law-gpt-backend2.0"
KANOON_REMOTE_PATH = "Kanoon data cleande"
KANOON_BRANCH = os.getenv("KANOON_BRANCH", "main")
KANOON_ENABLE_REMOTE = os.getenv("KANOON_ENABLE_REMOTE", "1") == "1"
KANOON_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "_remote_cache")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ")

# Initialize Redis cache
try:
    redis_cache = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    redis_cache.ping()
    logger.info("Redis cache connected for API responses")
except:
    redis_cache = None
    logger.warning("Redis cache not available for API responses")

def _safe_extend_json(records: List[Dict[str, Any]], payload: Any, source_hint: str = "") -> int:
    """Append JSON payload to records, handling list/dict and tagging source."""
    added = 0
    try:
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if source_hint and "source" not in item:
                        item["source"] = source_hint
                    records.append(item)
                    added += 1
        elif isinstance(payload, dict):
            item = payload
            if source_hint and "source" not in item:
                item["source"] = source_hint
            records.append(item)
            added += 1
    except Exception as e:
        logger.warning(f"Error extending records from {source_hint}: {e}")
    return added

def load_all_json_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all JSON data from local and remote sources"""
    all_records = []
    
    # Load local data
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        added = _safe_extend_json(all_records, data, f"local:{filename}")
                        logger.info(f"Loaded {added} records from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
    
    # Load remote data if enabled
    if KANOON_ENABLE_REMOTE:
        try:
            import requests
            import tempfile
            
            # GitHub API to list files
            api_url = f"https://api.github.com/repos/{KANOON_REMOTE_REPO}/contents/{KANOON_REMOTE_PATH}"
            params = {"ref": KANOON_BRANCH}
            
            response = requests.get(api_url, params=params, timeout=30)
            if response.status_code == 200:
                files = response.json()
                
                # Ensure cache directory exists
                os.makedirs(KANOON_CACHE_DIR, exist_ok=True)
                
                for file_info in files:
                    if file_info["name"].endswith(".json"):
                        cache_path = os.path.join(KANOON_CACHE_DIR, file_info["name"])
                        
                        # Download if not cached or outdated
                        if not os.path.exists(cache_path):
                            file_url = file_info["download_url"]
                            file_response = requests.get(file_url, timeout=60)
                            
                            if file_response.status_code == 200:
                                with open(cache_path, 'w', encoding='utf-8') as f:
                                    f.write(file_response.text)
                                logger.info(f"Downloaded {file_info['name']}")
                        
                        # Load cached file
                        try:
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                added = _safe_extend_json(all_records, data, f"remote:{file_info['name']}")
                                logger.info(f"Loaded {added} records from remote {file_info['name']}")
                        except Exception as e:
                            logger.error(f"Error loading cached {file_info['name']}: {e}")
            else:
                logger.warning(f"Failed to fetch remote file list: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error loading remote data: {e}")
    
    logger.info(f"Total records loaded: {len(all_records)}")
    return all_records

# Initialize FastAPI app
app = FastAPI(
    title="Law GPT Enhanced API",
    description="Advanced Legal AI Assistant with Multilingual RAG",
    version="13.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load knowledge base and initialize Enhanced RAG pipeline
KNOWLEDGE_BASE = load_all_json_data(DATA_DIRECTORY)
rag_pipeline = EnhancedMultilingualRAGPipeline(KNOWLEDGE_BASE, GEMINI_API_KEY)
logger.info(f"Enhanced knowledge base ready with {len(KNOWLEDGE_BASE)} records")

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    language: str | None = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    topic: str
    session_id: str
    processing_time: float
    sources: list = []
    detected_language: str | None = None
    retrieval_method: str | None = None

def get_response_cache_key(query: str, session_id: str) -> str:
    """Generate cache key for API responses"""
    return f"api_response:{hashlib.md5(f'{query}:{session_id}'.encode()).hexdigest()}"

@app.get("/")
async def root():
    """Enhanced root status with multilingual RAG information"""
    return {
        "message": "⚡ Law GPT Enhanced Multilingual API is running!",
        "status": "healthy",
        "version": "13.0.0",
        "features": {
            "multilingual_rag": True,
            "semantic_search": True,
            "caching": redis_cache is not None,
            "ai_enabled": rag_pipeline.ai_enabled
        },
        "ai_status": "enabled" if rag_pipeline.ai_enabled else "template_mode",
        "knowledge_base_size": len(KNOWLEDGE_BASE),
        "supported_languages": ["en", "hi", "ta", "te", "bn"],
        "remote_sources": {
            "kanoon_repo": KANOON_REMOTE_REPO,
            "kanoon_path": KANOON_REMOTE_PATH,
            "kanoon_branch": KANOON_BRANCH,
            "remote_enabled": KANOON_ENABLE_REMOTE
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    return {
        "status": "healthy",
        "ai_model_status": "enabled" if rag_pipeline.ai_enabled else "template_mode",
        "knowledge_base_size": len(KNOWLEDGE_BASE),
        "cache_status": "enabled" if redis_cache else "disabled",
        "multilingual_rag": True,
        "semantic_search": True,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/features")
async def get_features():
    """Get available features and capabilities"""
    return {
        "multilingual_support": {
            "enabled": True,
            "languages": ["English", "Hindi", "Tamil", "Telugu", "Bengali"],
            "semantic_search": True
        },
        "ai_capabilities": {
            "legal_reasoning": rag_pipeline.ai_enabled,
            "structured_responses": True,
            "citation_tracking": True,
            "confidence_scoring": True
        },
        "performance": {
            "caching": redis_cache is not None,
            "semantic_indexing": True,
            "parallel_processing": True
        },
        "knowledge_base": {
            "size": len(KNOWLEDGE_BASE),
            "topics": ["criminal_law", "contract_law", "constitutional_law", "property_law", "company_law"],
            "sources": ["IPC", "Constitution", "Contract Act", "Companies Act", "CrPC"]
        }
    }

@app.options("/chat")
async def chat_options():
    """Handle CORS preflight requests for chat endpoint"""
    return {"message": "OK"}

@app.post("/chat", response_model=ChatResponse)
async def enhanced_chat_endpoint(request: ChatRequest):
    """Enhanced chat endpoint with multilingual RAG and caching"""
    start_time = datetime.now()
    
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        session_id = request.session_id or str(uuid.uuid4())
        
        # Check cache first
        cache_key = get_response_cache_key(request.query, session_id)
        if redis_cache:
            try:
                cached_response = redis_cache.get(cache_key)
                if cached_response:
                    logger.info("Retrieved response from cache")
                    cached_data = json.loads(cached_response)
                    return ChatResponse(**cached_data)
            except Exception as e:
                logger.warning(f"Cache retrieval error: {e}")
        
        # Process through Enhanced Multilingual RAG pipeline
        result = await rag_pipeline.process_query(
            request.query, 
            session_id, 
            request.language
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response_data = {
            "response": result.get("response", ""),
            "confidence": float(result.get("confidence", 0.0)),
            "topic": result.get("topic", "general_law"),
            "session_id": session_id,
            "processing_time": processing_time,
            "sources": result.get("sources", []),
            "detected_language": result.get("language", "en"),
            "retrieval_method": result.get("retrieval_method", "enhanced_multilingual")
        }
        
        # Cache the response
        if redis_cache:
            try:
                redis_cache.setex(cache_key, 3600, json.dumps(response_data))  # Cache for 1 hour
            except Exception as e:
                logger.warning(f"Cache storage error: {e}")
        
        return ChatResponse(**response_data)

    except Exception as e:
        logger.error(f"Error processing enhanced chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/stats")
async def get_stats():
    """Get system statistics and performance metrics"""
    cache_stats = {}
    if redis_cache:
        try:
            info = redis_cache.info()
            cache_stats = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
        except:
            cache_stats = {"status": "error"}
    
    return {
        "knowledge_base": {
            "total_documents": len(KNOWLEDGE_BASE),
            "indexed_documents": len(KNOWLEDGE_BASE)
        },
        "cache": cache_stats,
        "ai_model": {
            "status": "enabled" if rag_pipeline.ai_enabled else "template_mode",
            "model_type": "gemini-pro" if rag_pipeline.ai_enabled else "template"
        },
        "multilingual": {
            "semantic_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "supported_languages": ["en", "hi", "ta", "te", "bn"]
        }
    }

@app.get("/clear-cache")
async def clear_cache():
    """Clear Redis cache (admin endpoint)"""
    if redis_cache:
        try:
            redis_cache.flushdb()
            return {"message": "Cache cleared successfully"}
        except Exception as e:
            return {"error": f"Failed to clear cache: {e}"}
    else:
        return {"message": "Cache not available"}

if __name__ == "__main__":
    uvicorn.run(
        "main_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )