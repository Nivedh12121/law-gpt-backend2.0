"""
Law GPT - Enhanced Production Backend
Complete version with company law support
"""

import os
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
from datetime import datetime
from googletrans import Translator, LANGUAGES

# Integrate Advanced RAG
from advanced_rag import AdvancedRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIRECTORY = "data"
CORS_ORIGINS = ["*"]

# Remote dataset configuration (GitHub folder with Kanoon cleaned JSON)
KANOON_REMOTE_REPO = "Nivedh12121/law-gpt-backend2.0"
KANOON_REMOTE_PATH = "Kanoon data cleande"
KANOON_BRANCH = os.getenv("KANOON_BRANCH", "main")
KANOON_ENABLE_REMOTE = os.getenv("KANOON_ENABLE_REMOTE", "1") == "1"
KANOON_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "_remote_cache")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ")

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
        logger.error(f"Failed to extend records from {source_hint}: {e}")
    return added

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all_json_data(data_dir: str) -> List[Dict[str, Any]]:
    all_data: List[Dict[str, Any]] = []
    base_dir = os.path.join(os.path.dirname(__file__), data_dir)
    if not os.path.exists(base_dir):
        logger.warning(f"Data directory {base_dir} does not exist")
    else:
        # Load local JSON (including subfolders)
        for root, _, files in os.walk(base_dir):
            for filename in files:
                if not filename.lower().endswith(".json"):
                    continue
                filepath = os.path.join(root, filename)
                try:
                    payload = _load_json_file(filepath)
                    added = _safe_extend_json(all_data, payload, source_hint=os.path.relpath(filepath, base_dir))
                    logger.debug(f"Loaded {added} items from {filepath}")
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")

    # Optionally load remote GitHub JSON files from Kanoon path
    if KANOON_ENABLE_REMOTE:
        try:
            import requests
            # Use GitHub API to list contents of the folder (supports spaces via URL encoding)
            api_url = f"https://api.github.com/repos/{KANOON_REMOTE_REPO}/contents/{requests.utils.requote_uri(KANOON_REMOTE_PATH)}?ref={KANOON_BRANCH}"
            logger.info(f"Fetching remote dataset index: {api_url}")
            r = requests.get(api_url, timeout=30)
            r.raise_for_status()
            listing = r.json()
            if isinstance(listing, list):
                _ensure_dir(KANOON_CACHE_DIR)
                for entry in listing:
                    if entry.get("type") != "file":
                        continue
                    name = entry.get("name", "")
                    if not name.lower().endswith(".json"):
                        continue
                    download_url = entry.get("download_url")
                    if not download_url:
                        # Fallback to raw URL
                        download_url = f"https://raw.githubusercontent.com/{KANOON_REMOTE_REPO}/{KANOON_BRANCH}/{KANOON_REMOTE_PATH}/{name}"
                    cache_path = os.path.join(KANOON_CACHE_DIR, name)
                    try:
                        logger.info(f"Downloading remote dataset file: {name}")
                        rr = requests.get(download_url, timeout=60)
                        rr.raise_for_status()
                        with open(cache_path, "wb") as out:
                            out.write(rr.content)
                        payload = _load_json_file(cache_path)
                        added = _safe_extend_json(all_data, payload, source_hint=f"remote:{name}")
                        logger.info(f"Loaded {added} items from remote {name}")
                    except Exception as e:
                        logger.error(f"Failed remote fetch {name}: {e}")
            else:
                logger.warning("Remote listing did not return a file array; skipping remote load.")
        except Exception as e:
            logger.error(f"Remote dataset load failed: {e}")

    logger.info(f"Total knowledge records loaded: {len(all_data)}")
    return all_data

# Initialize FastAPI app
app = FastAPI(
    title="Law GPT API - Advanced Legal AI",
    description="Next-generation AI-powered Indian legal assistant",
    version="12.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Load knowledge base and initialize Advanced RAG pipeline
KNOWLEDGE_BASE = load_all_json_data(DATA_DIRECTORY)
rag_pipeline = AdvancedRAGPipeline(KNOWLEDGE_BASE, GEMINI_API_KEY)
logger.info(f"Knowledge base ready with {len(KNOWLEDGE_BASE)} records (remote enabled={KANOON_ENABLE_REMOTE})")
translator = Translator()

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    topic: str
    session_id: str
    processing_time: float
    sources: list = []
    detected_language: str | None = None

@app.get("/")
async def root():
    """Public root status with explicit versioning and AI status."""
    return {
        "message": "⚡ Law GPT Enhanced API is running!",
        "status": "healthy",
        "version": "12.0.0",
        "ai_status": "enabled" if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ" else "template_mode",
        "knowledge_base_size": len(KNOWLEDGE_BASE),
        "remote_sources": {
            "kanoon_repo": KANOON_REMOTE_REPO,
            "kanoon_path": KANOON_REMOTE_PATH,
            "kanoon_branch": KANOON_BRANCH,
            "remote_enabled": KANOON_ENABLE_REMOTE
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ai_model_status": "enabled" if GEMINI_API_KEY and GEMINI_API_KEY != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ" else "template_mode",
        "knowledge_base_size": len(KNOWLEDGE_BASE),
        "timestamp": datetime.now().isoformat()
    }

@app.options("/chat")
async def chat_options():
    """Handle CORS preflight requests for chat endpoint"""
    return {"message": "OK"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now()
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        session_id = request.session_id or str(uuid.uuid4())

        # Language detection and translation
        detected = translator.detect(request.query)
        detected_lang = detected.lang
        
        if detected_lang != 'en':
            translated_query = translator.translate(request.query, dest='en').text
            logger.info(f"Translated query ({detected_lang} -> en): {translated_query}")
        else:
            translated_query = request.query

        # Route through Advanced RAG pipeline
        result = await rag_pipeline.process_query(translated_query, session_id)
        
        english_response = result.get("response", "")
        logger.info(f"RAG response (English): {english_response}")
        
        if detected_lang != 'en':
            final_response = translator.translate(english_response, dest=detected_lang).text
        else:
            final_response = english_response

        processing_time = (datetime.now() - start_time).total_seconds()

        return ChatResponse(
            response=final_response,
            confidence=float(result.get("confidence", 0.0)),
            topic=result.get("topic", "general_law"),
            session_id=session_id,
            processing_time=processing_time,
            sources=result.get("sources", []),
            detected_language=LANGUAGES.get(detected_lang, detected_lang)
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/languages")
async def get_language_support():
    return {
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "Hindi"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"},
            {"code": "bn", "name": "Bengali"}
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )