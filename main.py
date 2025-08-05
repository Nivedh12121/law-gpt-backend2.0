import os
import json
import logging
import re
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# --- Configuration ---
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
    all_data = []
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
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error reading or parsing {filename}: {e}")
    
    logger.info(f"Successfully loaded {len(all_data)} records.")
    return all_data

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Law GPT API - Final Version",
    description="AI Legal Assistant powered by a large knowledge base.",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

KNOWLEDGE_BASE = load_all_json_data(DATA_DIRECTORY)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

# --- Intelligent Search Logic ---
STOP_WORDS = set(["a", "about", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "who", "will", "with", "the", "www"])

def tokenize(text: str) -> set:
    text = re.sub(r'[^\w\s]', '', text)
    words = text.lower().split()
    return set(words) - STOP_WORDS

def find_best_answer(query: str) -> Dict[str, Any]:
    query_tokens = tokenize(query)
    best_match = None
    highest_score = 0

    if not query_tokens:
        return {"response": "Please provide a more specific query.", "sources": ["Query Processor"]}

    for item in KNOWLEDGE_BASE:
        question = item.get("question", "")
        if not question:
            continue
        question_tokens = tokenize(question)
        common_tokens = query_tokens.intersection(question_tokens)
        score = len(common_tokens)
        
        if score > highest_score:
            highest_score = score
            best_match = item

    if best_match and highest_score > 1:
        return {
            "response": best_match.get("answer", "Answer not found."),
            "sources": [best_match.get("context", "Kanoon Database")]
        }
            
    return {
        "response": "I could not find a specific answer in my database. Please try rephrasing your question.",
        "sources": ["Default Response"]
    }

# --- API Endpoints ---
@app.get("/")
async def health_check():
    return {"message": "Law GPT API is running!", "knowledge_base_size": len(KNOWLEDGE_BASE)}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not KNOWLEDGE_BASE:
        raise HTTPException(status_code=503, detail="Knowledge base is not loaded.")
    response_data = find_best_answer(request.query)
    return ChatResponse(**response_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)