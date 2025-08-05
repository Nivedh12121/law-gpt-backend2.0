import os
import json
import logging
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
# The data directory is now relative to the script, inside the project
DATA_DIRECTORY = "Kanoon data cleande"
CORS_ORIGINS = [
    "https://law-gpt-frontend-2-0.vercel.app",
    "https://law-gpt-frontend-2-0-y3zc-1sfz8xujo-nivedhs-projects-ce31ae36.vercel.app",
    "https://law-gpt-frontend-2-0-y3zc-fcol8q14r-nivedhs-projects-ce31ae36.vercel.app",
    "http://localhost:3000",
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Loading ---
def load_all_json_data(directory: str) -> List[Dict[str, Any]]:
    """Loads and combines data from all JSON files in a directory."""
    all_data = []
    # Get the absolute path to the data directory
    # This is important for when the script is run by a server like Railway
    script_dir = os.path.dirname(__file__)
    data_dir_path = os.path.join(script_dir, directory)
    
    logger.info(f"Attempting to load data from: {data_dir_path}")
    
    if not os.path.isdir(data_dir_path):
        logger.error(f"Data directory not found: {data_dir_path}")
        return []
        
    for filename in os.listdir(data_dir_path):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        logger.warning(f"Skipping non-list JSON file: {filename}")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading or parsing {filename}: {e}")
    
    logger.info(f"Successfully loaded {len(all_data)} records from {len(os.listdir(data_dir_path))} files.")
    return all_data

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Law GPT API - Hybrid Model",
    description="AI Legal Assistant powered by a large knowledge base and ready for generative AI integration.",
    version="3.0.0"
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
    This is the function to be enhanced with a call to a generative model like Gemini.
    """
    query_lower = query.lower().strip()
    
    # Priority 1: Search the structured knowledge base first
    for item in KNOWLEDGE_BASE:
        question = item.get("question", "").lower()
        if query_lower in question:
            logger.info(f"Found a direct match for '{query}' in the knowledge base.")
            return {
                "response": item.get("answer", "An answer was found but could not be formatted."),
                "sources": [item.get("context", "Kanoon Database")]
            }
            
    # Priority 2: If no direct match, this is where you would call a generative model
    # For now, we will return a helpful default message.
    # Example placeholder for a future Gemini call:
    # if USE_GEMINI_API:
    #     return call_gemini_api(query, context=KNOWLEDGE_BASE)
    
    logger.info(f"No direct match found for '{query}'. Returning default response.")
    return {
        "response": f"""I could not find a specific answer for "{query}" in my structured database. 
        
This is an opportunity to use a generative AI model like Google's Gemini to provide a more detailed, conversational answer. 

For now, please try rephrasing your question or asking about a different legal topic.""",
        "sources": ["Default Response"]
    }

# --- API Endpoints ---
@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "message": "⚡ Law GPT API v3 (Hybrid) is running!",
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
        reload=True
    )