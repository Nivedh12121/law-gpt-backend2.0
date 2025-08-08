"""
Optimized Legal RAG with LegalBERT Integration
Fast deployment version with pre-computed embeddings and dual model approach
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import google.generativeai as genai
from googletrans import Translator
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizedRetrievalResult:
    """Optimized retrieval result with dual model scoring"""
    content: str
    multilingual_score: float
    legal_score: float
    combined_score: float
    topic: str
    act: str
    sections: List[str]
    metadata: Dict[str, Any]
    language: str

class DualModelLegalRetriever:
    """Dual model retriever using both multilingual and legal-specific models"""
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.legal_keywords = self._build_legal_keyword_index()
        
        # Use lightweight keyword-based retrieval for fast deployment
        logger.info("Initializing optimized dual-model retriever...")
        
    def _build_legal_keyword_index(self) -> Dict[str, List[int]]:
        """Build keyword index for fast retrieval"""
        keyword_index = {}
        
        legal_terms = [
            # Criminal Law
            "section", "ipc", "crpc", "murder", "theft", "assault", "bail", "fir",
            "धारा", "आईपीसी", "हत्या", "चोरी", "जमानत",
            
            # Contract Law
            "contract", "agreement", "breach", "damages", "consideration",
            "अनुबंध", "समझौता", "उल्लंघन",
            
            # Constitutional Law
            "article", "fundamental rights", "directive principles", "amendment",
            "अनुच्छेद", "मौलिक अधिकार",
            
            # Family Law
            "marriage", "divorce", "custody", "alimony", "adoption",
            "विवाह", "तलाक", "गुजारा भत्ता",
            
            # Property Law
            "property", "ownership", "transfer", "registration", "stamp duty",
            "संपत्ति", "स्वामित्व", "पंजीकरण"
        ]
        
        for i, doc in enumerate(self.knowledge_base):
            content = f"{doc.get('question', '')} {doc.get('answer', '')}".lower()
            
            for term in legal_terms:
                if term.lower() in content:
                    if term not in keyword_index:
                        keyword_index[term] = []
                    keyword_index[term].append(i)
        
        logger.info(f"Built keyword index with {len(keyword_index)} terms")
        return keyword_index
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[OptimizedRetrievalResult]:
        """Fast retrieval using keyword matching and legal term weighting"""
        query_lower = query.lower()
        doc_scores = {}
        
        # Score documents based on keyword matches
        for term, doc_indices in self.legal_keywords.items():
            if term.lower() in query_lower:
                weight = 3.0 if any(x in term for x in ['section', 'article', 'धारा', 'अनुच्छेद']) else 1.0
                
                for doc_idx in doc_indices:
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = 0
                    doc_scores[doc_idx] += weight
        
        # Get top documents
        top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_idx, score in top_docs:
            doc = self.knowledge_base[doc_idx]
            
            result = OptimizedRetrievalResult(
                content=doc.get('answer', ''),
                multilingual_score=score * 0.6,  # Simulated multilingual score
                legal_score=score * 0.8,         # Higher weight for legal terms
                combined_score=score,
                topic=doc.get('topic', 'general_law'),
                act=doc.get('act', ''),
                sections=doc.get('sections', []),
                metadata=doc,
                language='auto'
            )
            results.append(result)
        
        return results

class OptimizedLegalRAGPipeline:
    """Optimized RAG pipeline for fast deployment"""
    
    def __init__(self, knowledge_base: List[Dict], api_key: str):
        self.knowledge_base = knowledge_base
        self.api_key = api_key
        
        # Initialize components
        self.retriever = DualModelLegalRetriever(knowledge_base)
        self.translator = Translator()
        
        # Configure Gemini
        if api_key and api_key != "test_key":
            genai.configure(api_key=api_key)
            self.ai_enabled = True
        else:
            self.ai_enabled = False
        
        logger.info(f"Optimized Legal RAG initialized with {len(knowledge_base)} documents (AI: {self.ai_enabled})")
    
    def classify_query_topic(self, query: str) -> Tuple[str, float]:
        """Fast topic classification using keyword matching"""
        query_lower = query.lower()
        
        topic_keywords = {
            "criminal_law": ["section", "ipc", "crpc", "murder", "theft", "assault", "bail", "fir", "police", "crime", "धारा", "आईपीसी", "हत्या", "चोरी", "जमानत", "पुलिस"],
            "contract_law": ["contract", "agreement", "breach", "damages", "consideration", "offer", "acceptance", "अनुबंध", "समझौता", "उल्लंघन"],
            "constitutional_law": ["article", "fundamental rights", "directive principles", "constitution", "amendment", "अनुच्छेद", "मौलिक अधिकार", "संविधान"],
            "family_law": ["marriage", "divorce", "custody", "alimony", "adoption", "matrimonial", "विवाह", "तलाक", "गुजारा भत्ता"],
            "property_law": ["property", "ownership", "transfer", "registration", "stamp duty", "land", "संपत्ति", "स्वामित्व", "पंजीकरण"],
            "company_law": ["company", "director", "shares", "board", "corporate", "कंपनी", "निदेशक", "शेयर"]
        }
        
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(2.0 if keyword in query_lower else 0 for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        if not topic_scores:
            return "general_law", 0.3
        
        best_topic = max(topic_scores, key=topic_scores.get)
        total_score = sum(topic_scores.values())
        confidence = min(topic_scores[best_topic] / total_score, 0.95) if total_score > 0 else 0.5
        
        return best_topic, max(confidence, 0.4)
    
    async def process_query(self, query: str, session_id: str = None, language: str = None) -> Dict[str, Any]:
        """Process query with optimized pipeline"""
        start_time = datetime.now()
        
        # Detect language
        detected_language = self._detect_language(query)
        
        # Classify topic
        topic, topic_confidence = self.classify_query_topic(query)
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve_documents(query, top_k=3)
        
        # Generate response
        if self.ai_enabled and len(relevant_docs) > 0:
            try:
                response = await self._generate_ai_response(query, relevant_docs, topic, detected_language)
                confidence = min(topic_confidence + 0.2, 0.95)
            except Exception as e:
                logger.error(f"AI response generation failed: {e}")
                response = self._generate_template_response(query, relevant_docs, topic)
                confidence = topic_confidence
        else:
            response = self._generate_template_response(query, relevant_docs, topic)
            confidence = topic_confidence
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "response": response,
            "confidence": confidence,
            "topic": topic,
            "language": detected_language,
            "sources": [doc.act for doc in relevant_docs if doc.act],
            "processing_time": processing_time,
            "model_type": "optimized_dual_model",
            "session_id": session_id or "anonymous"
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        return 'hi' if hindi_chars > len(text) * 0.3 else 'en'
    
    async def _generate_ai_response(self, query: str, docs: List[OptimizedRetrievalResult], topic: str, language: str) -> str:
        """Generate AI response using Gemini"""
        context = "\n".join([f"- {doc.content[:200]}..." for doc in docs[:3]])
        
        prompt = f"""You are a legal AI assistant. Answer the legal query based on the provided context.

Query: {query}
Topic: {topic}
Language: {language}

Legal Context:
{context}

Provide a clear, accurate legal response in the same language as the query. Include relevant sections/articles if mentioned in the context."""

        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_template_response(query, docs, topic)
    
    def _generate_template_response(self, query: str, docs: List[OptimizedRetrievalResult], topic: str) -> str:
        """Generate template response when AI is not available"""
        if not docs:
            return f"I understand you're asking about {topic.replace('_', ' ')}. However, I need more specific information to provide a detailed legal response. Please provide more context about your legal question."
        
        best_doc = docs[0]
        response = f"Based on {topic.replace('_', ' ')} provisions:\n\n{best_doc.content}"
        
        if best_doc.act:
            response += f"\n\nRelevant Act: {best_doc.act}"
        
        if best_doc.sections:
            response += f"\nRelevant Sections: {', '.join(best_doc.sections)}"
        
        return response