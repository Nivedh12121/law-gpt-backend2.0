import os
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DATA_DIRECTORY = "C:\\Users\\nived\\Downloads\\Kanoon data cleande-20250723T145657Z-1-001\\Kanoon data cleande"
CORS_ORIGINS = [
    "https://law-gpt-frontend-2-0.vercel.app",
    "https://law-gpt-frontend-2-0-y3zc-1sfz8xujo-nivedhs-projects-ce31ae36.vercel.app",
    "http://localhost:3000",
]

# --- Data Loading ---
def load_all_json_data(directory: str) -> List[Dict[str, Any]]:
    """Loads and combines data from all JSON files in a directory."""
    all_data = []
    logger.info(f"Loading data from: {directory}")
    if not os.path.isdir(directory):
        logger.error(f"Data directory not found: {directory}")
        return []
        
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        logger.warning(f"Skipping non-list JSON file: {filename}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading or parsing {filename}: {e}")
    logger.info(f"Successfully loaded {len(all_data)} records from {len(os.listdir(directory))} files.")
    return all_data

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Law GPT API - Enhanced",
    description="AI Legal Assistant powered by a large knowledge base.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory Knowledge Base ---
KNOWLEDGE_BASE = load_all_json_data(DATA_DIRECTORY)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

# --- Search Logic ---
def find_best_answer(query: str) -> Dict[str, Any]:
    """
    Finds the best answer from the knowledge base using simple keyword matching.
    """
    query_lower = query.lower().strip()
    
    # Simple search: find the first question that contains the query
    for item in KNOWLEDGE_BASE:
        question = item.get("question", "").lower()
        if query_lower in question:
            logger.info(f"Found a match for '{query}' in question: '{item.get('question')}'")
            return {
                "response": item.get("answer", "No answer found for this question."),
                "sources": [item.get("context", "Kanoon Database")]
            }
            
    # If no direct match, return a default response
    return {
        "response": "I could not find a specific answer to your question in my database. Please try rephrasing your query.",
        "sources": ["Knowledge Base Search"]
    }

# --- API Endpoints ---
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "message": "⚡ Law GPT API v2 is running!",
        "status": "healthy",
        "knowledge_base_size": len(KNOWLEDGE_BASE)
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for legal queries."""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not KNOWLEDGE_BASE:
            raise HTTPException(status_code=503, detail="Knowledge base is not loaded. Please check server logs.")

        response_data = find_best_answer(request.query)
        
        return ChatResponse(
            response=response_data["response"],
            sources=response_data.get("sources", [])
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True # Added for easier development
    )