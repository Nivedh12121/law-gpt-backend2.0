"""
Enhanced Multilingual RAG Pipeline for Law GPT
Implements semantic search with sentence transformers for better multilingual support
"""

import os
import json
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import redis
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from googletrans import Translator, LANGUAGES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedRetrievalResult:
    """Enhanced retrieval result with semantic similarity"""
    content: str
    semantic_score: float
    keyword_score: float
    combined_score: float
    topic: str
    act: str
    sections: List[str]
    metadata: Dict[str, Any]
    language: str

class MultilingualSemanticRetriever:
    """Semantic retrieval using multilingual sentence transformers"""
    
    def __init__(self, knowledge_base: List[Dict], cache_dir: str = "model_cache"):
        self.knowledge_base = knowledge_base
        self.cache_dir = cache_dir
        
        # Initialize multilingual model
        logger.info("Loading multilingual sentence transformer...")
        self.model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2',
            cache_folder=cache_dir
        )
        
        # Initialize FAISS index
        self.index = None
        self.document_embeddings = None
        self.documents = []
        
        # Initialize Redis cache (optional)
        try:
            self.cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.cache.ping()
            logger.info("Redis cache connected")
        except:
            self.cache = None
            logger.warning("Redis cache not available, using in-memory cache")
            self.memory_cache = {}
        
        self._build_semantic_index()
    
    def _build_semantic_index(self):
        """Build FAISS index for semantic search"""
        logger.info(f"Building semantic index for {len(self.knowledge_base)} documents...")
        
        # Extract document texts
        self.documents = []
        for doc in self.knowledge_base:
            # Combine question and answer for better retrieval
            text = ""
            if 'question' in doc:
                text += doc['question'] + " "
            if 'answer' in doc:
                text += doc['answer']
            elif 'content' in doc:
                text += doc['content']
            
            self.documents.append({
                'text': text,
                'metadata': doc
            })
        
        # Generate embeddings
        texts = [doc['text'] for doc in self.documents]
        logger.info("Generating embeddings...")
        self.document_embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.document_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.document_embeddings)
        self.index.add(self.document_embeddings.astype('float32'))
        
        logger.info(f"Semantic index built with {self.index.ntotal} documents")
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """Generate cache key for query"""
        return f"semantic_search:{hashlib.md5(f'{query}:{top_k}'.encode()).hexdigest()}"
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[EnhancedRetrievalResult]:
        """Perform semantic search using sentence transformers"""
        
        # Check cache first
        cache_key = self._get_cache_key(query, top_k)
        if self.cache:
            try:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.info("Retrieved from Redis cache")
                    return json.loads(cached_result)
            except:
                pass
        elif hasattr(self, 'memory_cache') and cache_key in self.memory_cache:
            logger.info("Retrieved from memory cache")
            return self.memory_cache[cache_key]
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                metadata = doc['metadata']
                
                result = EnhancedRetrievalResult(
                    content=doc['text'],
                    semantic_score=float(score),
                    keyword_score=0.0,  # Will be calculated separately
                    combined_score=float(score),
                    topic=metadata.get('topic', 'general_law'),
                    act=metadata.get('act', ''),
                    sections=metadata.get('sections', []),
                    metadata=metadata,
                    language=metadata.get('language', 'en')
                )
                results.append(result)
        
        # Cache results
        if self.cache:
            try:
                self.cache.setex(cache_key, 3600, json.dumps([r.__dict__ for r in results]))
            except:
                pass
        elif hasattr(self, 'memory_cache'):
            self.memory_cache[cache_key] = results
        
        return results

class EnhancedTopicClassifier:
    """Enhanced topic classifier with multilingual support"""
    
    def __init__(self):
        self.legal_topics = {
            "contract_law": {
                "keywords": {
                    "en": ["contract", "agreement", "offer", "acceptance", "consideration", "breach", "damages"],
                    "hi": ["अनुबंध", "समझौता", "प्रस्ताव", "स्वीकृति", "विचार", "उल्लंघन", "हर्जाना"]
                },
                "acts": ["indian contract act 1872"],
                "sections": ["10", "11", "13", "14", "15", "16", "17", "18", "23"]
            },
            "criminal_law": {
                "keywords": {
                    "en": ["ipc", "indian penal code", "crpc", "murder", "theft", "bail", "fir", "section 302"],
                    "hi": ["आईपीसी", "भारतीय दंड संहिता", "हत्या", "चोरी", "जमानत", "एफआईआर", "धारा 302"]
                },
                "acts": ["indian penal code 1860", "code of criminal procedure 1973"],
                "sections": ["302", "420", "436", "437", "438", "154", "161", "173"]
            },
            "constitutional_law": {
                "keywords": {
                    "en": ["constitution", "article", "fundamental rights", "article 14", "article 19", "article 21"],
                    "hi": ["संविधान", "अनुच्छेद", "मौलिक अधिकार", "अनुच्छेद 14", "अनुच्छेद 19", "अनुच्छेद 21"]
                },
                "acts": ["constitution of india 1950"],
                "sections": ["14", "19", "21", "32", "226", "356", "370"]
            }
        }
    
    def classify_multilingual_query(self, query: str, language: str = 'en') -> Tuple[str, float]:
        """Classify query with multilingual keyword support"""
        query_lower = query.lower()
        topic_scores = {}
        
        for topic, data in self.legal_topics.items():
            score = 0.0
            
            # Check keywords for detected language
            keywords = data["keywords"].get(language, data["keywords"].get("en", []))
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 2.0
            
            # Check English keywords as fallback
            if language != 'en':
                for keyword in data["keywords"].get("en", []):
                    if keyword.lower() in query_lower:
                        score += 1.0
            
            # Act and section matching
            for act in data["acts"]:
                if any(word in query_lower for word in act.split()):
                    score += 3.0
            
            for section in data["sections"]:
                if f"section {section}" in query_lower or f"धारा {section}" in query_lower:
                    score += 5.0
            
            topic_scores[topic] = score
        
        if not topic_scores or max(topic_scores.values()) == 0:
            return "general_law", 0.3
        
        best_topic = max(topic_scores, key=topic_scores.get)
        total_score = sum(topic_scores.values())
        confidence = topic_scores[best_topic] / total_score if total_score > 0 else 0.5
        
        return best_topic, max(confidence, 0.3)

class EnhancedMultilingualRAGPipeline:
    """Enhanced RAG pipeline with multilingual semantic search"""
    
    def __init__(self, knowledge_base: List[Dict], api_key: str):
        self.knowledge_base = knowledge_base
        self.api_key = api_key
        
        # Initialize components
        self.retriever = MultilingualSemanticRetriever(knowledge_base)
        self.topic_classifier = EnhancedTopicClassifier()
        self.translator = Translator()
        
        # Configure Gemini
        if api_key and api_key != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.ai_enabled = True
        else:
            self.model = None
            self.ai_enabled = False
        
        logger.info(f"Enhanced Multilingual RAG Pipeline initialized (AI: {self.ai_enabled})")
    
    async def process_query(self, query: str, session_id: str = None, language: str = None) -> Dict[str, Any]:
        """Process query with enhanced multilingual support"""
        start_time = datetime.now()
        
        try:
            # Detect language if not provided
            if not language:
                detected = self.translator.detect(query)
                language = detected.lang
            
            # Classify topic with multilingual support
            topic, topic_confidence = self.topic_classifier.classify_multilingual_query(query, language)
            
            # Perform semantic search (works directly with multilingual queries)
            retrieval_results = self.retriever.semantic_search(query, top_k=10)
            
            # Prepare context for AI
            context_docs = []
            sources = []
            
            for result in retrieval_results[:5]:  # Top 5 results
                context_docs.append(result.content)
                sources.append({
                    'content': result.content[:200] + "...",
                    'score': result.semantic_score,
                    'topic': result.topic,
                    'act': result.act
                })
            
            # Generate AI response
            if self.ai_enabled and context_docs:
                response = await self._generate_ai_response(query, context_docs, topic, language)
                confidence = min(0.9, topic_confidence + 0.3)  # Boost confidence for semantic search
            else:
                response = self._generate_template_response(query, context_docs, topic)
                confidence = topic_confidence
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "response": response,
                "confidence": confidence,
                "topic": topic,
                "sources": sources,
                "processing_time": processing_time,
                "language": language,
                "retrieval_method": "semantic_multilingual"
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced RAG pipeline: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try again.",
                "confidence": 0.1,
                "topic": "general_law",
                "sources": [],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "language": language or "en",
                "error": str(e)
            }
    
    async def _generate_ai_response(self, query: str, context_docs: List[str], topic: str, language: str) -> str:
        """Generate AI response with enhanced prompting"""
        
        context = "\n\n".join(context_docs[:3])  # Use top 3 documents
        
        # Enhanced prompt with language awareness
        prompt = f"""You are an expert Indian legal AI assistant. Answer the legal query using the provided context.

Query: {query}
Topic: {topic}
Language: {language}

Legal Context:
{context}

Instructions:
1. Provide a comprehensive legal answer
2. Use bullet points for clarity
3. Include relevant legal sections and acts
4. Add a confidence indicator
5. Include a legal disclaimer
6. Respond in the same language as the query
7. Structure your response professionally

Response Format:
⚖️ **Legal Answer**
• [Key points in bullet format]

📚 **Legal Sources**
• [Relevant acts and sections]

🎯 **Confidence**: [High/Medium/Low]

⚠️ **Legal Disclaimer**
This information is for educational purposes only. Consult a qualified lawyer for specific legal advice.

Answer:"""

        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return self._generate_template_response(query, context_docs, topic)
    
    def _generate_template_response(self, query: str, context_docs: List[str], topic: str) -> str:
        """Generate template response when AI is not available"""
        
        if context_docs:
            relevant_content = context_docs[0][:500] + "..."
            return f"""⚖️ **Legal Information**

Based on your query about {topic}, here's relevant information:

{relevant_content}

📚 **Topic**: {topic.replace('_', ' ').title()}

🎯 **Confidence**: Medium

⚠️ **Legal Disclaimer**
This information is for educational purposes only. Please consult a qualified lawyer for specific legal advice.

Note: This response was generated using template mode. For more detailed analysis, please ensure the AI service is properly configured."""
        else:
            return f"""⚖️ **Legal Query Received**

I understand you're asking about {topic.replace('_', ' ')}.

Unfortunately, I couldn't find specific information in my knowledge base for your query. This might be because:
• The query is very specific or unique
• The topic requires specialized legal expertise
• The information might not be in my current database

🎯 **Recommendation**: Please consult with a qualified lawyer who specializes in {topic.replace('_', ' ')} for accurate legal advice.

⚠️ **Legal Disclaimer**
This response is for informational purposes only and does not constitute legal advice."""

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the enhanced pipeline
    pass