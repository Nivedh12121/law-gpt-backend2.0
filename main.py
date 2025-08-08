"""
Optimized Law GPT Backend - Fast Deployment Version
Ready for immediate Railway deployment with LegalBERT integration concepts
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
import asyncio
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import optimized components
try:
    from optimized_legal_rag import OptimizedLegalRAGPipeline
except ImportError:
    # Fallback import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("optimized_legal_rag", 
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "optimized_legal_rag.py"))
    optimized_legal_rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimized_legal_rag)
    OptimizedLegalRAGPipeline = optimized_legal_rag.OptimizedLegalRAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIRECTORY = "data"
CORS_ORIGINS = ["*"]

# Environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCgxJNRNc96O1SrCMiEPpFnvzPU--888Z8")
KANOON_ENABLE_REMOTE = os.getenv("KANOON_ENABLE_REMOTE", "1") == "1"
KANOON_BRANCH = os.getenv("KANOON_BRANCH", "main")

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    session_id: str = None
    language: str = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    topic: str
    language: str
    sources: List[str]
    processing_time: float
    model_type: str
    session_id: str

# Global variables
app = FastAPI(title="Law GPT - Optimized Backend", version="2.0-optimized")
rag_pipeline = None

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_sample_data() -> List[Dict[str, Any]]:
    """Load essential legal data for fast startup"""
    sample_data = [
        # Criminal Law - IPC Sections
        {
            "question": "What is Section 302 IPC?",
            "answer": "Section 302 of the Indian Penal Code deals with punishment for murder. Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
            "act": "Indian Penal Code, 1860",
            "sections": ["302"],
            "topic": "criminal_law"
        },
        {
            "question": "धारा 302 आईपीसी क्या है?",
            "answer": "भारतीय दंड संहिता की धारा 302 हत्या के लिए सजा से संबंधित है। जो कोई हत्या करता है, उसे मृत्यु या आजीवन कारावास की सजा दी जाएगी।",
            "act": "भारतीय दंड संहिता, 1860",
            "sections": ["302"],
            "topic": "criminal_law"
        },
        {
            "question": "What is Section 420 IPC?",
            "answer": "Section 420 of IPC deals with cheating and dishonestly inducing delivery of property. Punishment is imprisonment up to 7 years and fine.",
            "act": "Indian Penal Code, 1860",
            "sections": ["420"],
            "topic": "criminal_law"
        },
        {
            "question": "How to file an FIR?",
            "answer": "An FIR (First Information Report) can be filed at the nearest police station. It should contain details of the incident, time, place, and persons involved. FIR can be filed orally or in writing.",
            "act": "Code of Criminal Procedure, 1973",
            "sections": ["154"],
            "topic": "criminal_law"
        },
        
        # Contract Law
        {
            "question": "What are essential elements of a valid contract?",
            "answer": "Essential elements of a valid contract are: 1) Offer and Acceptance, 2) Consideration, 3) Capacity to contract, 4) Free consent, 5) Lawful object, 6) Not expressly declared void.",
            "act": "Indian Contract Act, 1872",
            "sections": ["10"],
            "topic": "contract_law"
        },
        
        # Constitutional Law
        {
            "question": "What is Article 21?",
            "answer": "Article 21 of the Indian Constitution guarantees the right to life and personal liberty. No person shall be deprived of his life or personal liberty except according to procedure established by law.",
            "act": "Constitution of India",
            "sections": ["21"],
            "topic": "constitutional_law"
        },
        
        # Family Law
        {
            "question": "What are grounds for divorce under Hindu Marriage Act?",
            "answer": "Grounds for divorce under Hindu Marriage Act include: adultery, cruelty, desertion for 2 years, conversion to another religion, mental disorder, communicable disease, and renunciation of world.",
            "act": "Hindu Marriage Act, 1955",
            "sections": ["13"],
            "topic": "family_law"
        },
        
        # Property Law
        {
            "question": "What is required for property registration?",
            "answer": "Property registration requires: sale deed, identity proof, address proof, PAN card, property documents, stamp duty payment, and registration fee payment.",
            "act": "Registration Act, 1908",
            "sections": ["17"],
            "topic": "property_law"
        }
    ]
    
    # Load additional data if available
    try:
        data_files = [
            "indian_penal_code.json",
            "contract_law_essentials.json",
            "criminal_procedure_code.json"
        ]
        
        for filename in data_files:
            filepath = os.path.join(DATA_DIRECTORY, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    additional_data = json.load(f)
                    if isinstance(additional_data, list):
                        sample_data.extend(additional_data[:50])  # Limit to 50 per file for fast startup
                        logger.info(f"Loaded {len(additional_data[:50])} records from {filename}")
    except Exception as e:
        logger.warning(f"Could not load additional data files: {e}")
    
    logger.info(f"Total sample data loaded: {len(sample_data)} records")
    return sample_data

def load_remote_data_sample() -> List[Dict[str, Any]]:
    """Load a sample of remote data for faster startup"""
    remote_data = []
    
    if not KANOON_ENABLE_REMOTE:
        return remote_data
    
    try:
        # Load only a few key files for fast startup
        key_files = [
            "1kaanoon_pages1-50_qas_cleaned.json",
            "IndicLegalQA Dataset_10K_Revised.json"
        ]
        
        cache_dir = os.path.join(DATA_DIRECTORY, "_remote_cache")
        
        for filename in key_files:
            filepath = os.path.join(cache_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Take only first 1000 records for fast startup
                        sample = data[:1000]
                        remote_data.extend(sample)
                        logger.info(f"Loaded {len(sample)} sample records from {filename}")
                    break  # Load only one file for fastest startup
    except Exception as e:
        logger.warning(f"Could not load remote data sample: {e}")
    
    return remote_data

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline
    
    logger.info("🚀 Starting Optimized Law GPT Backend...")
    
    # Load data
    sample_data = load_sample_data()
    remote_sample = load_remote_data_sample()
    
    all_data = sample_data + remote_sample
    logger.info(f"Total knowledge base: {len(all_data)} documents")
    
    # Initialize optimized RAG pipeline
    rag_pipeline = OptimizedLegalRAGPipeline(
        knowledge_base=all_data,
        api_key=GEMINI_API_KEY
    )
    
    logger.info("✅ Optimized Law GPT Backend ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Law GPT - Optimized Backend API",
        "version": "2.0-optimized",
        "status": "operational",
        "features": ["dual_model_retrieval", "fast_startup", "legal_bert_concepts"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ai_model_status": "optimized_dual_model",
        "knowledge_base_size": len(rag_pipeline.knowledge_base) if rag_pipeline else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint with optimized processing"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Process query
        result = await rag_pipeline.process_query(
            query=request.query,
            session_id=request.session_id,
            language=request.language
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_pipeline:
        return {"error": "Pipeline not initialized"}
    
    return {
        "knowledge_base_size": len(rag_pipeline.knowledge_base),
        "model_type": "optimized_dual_model",
        "ai_enabled": rag_pipeline.ai_enabled,
        "startup_time": "< 30 seconds",
        "features": [
            "keyword_based_retrieval",
            "legal_term_weighting", 
            "dual_model_scoring",
            "fast_topic_classification",
            "multilingual_support"
        ]
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)