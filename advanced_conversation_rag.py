"""
Advanced Conversation-Aware RAG Pipeline with Cross-Encoder Reranking
Implements conversation memory, topic switching detection, and advanced reranking
"""

import os
import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Single conversation turn"""
    query: str
    response: str
    topic: str
    confidence: float
    timestamp: datetime
    sources: List[str]

@dataclass
class ConversationSession:
    """Complete conversation session with memory"""
    session_id: str
    turns: List[ConversationTurn]
    current_topic: str
    topic_history: List[str]
    created_at: datetime
    last_activity: datetime
    
    def add_turn(self, turn: ConversationTurn):
        """Add a new conversation turn"""
        self.turns.append(turn)
        self.last_activity = datetime.now()
        
        # Update topic if changed
        if turn.topic != self.current_topic:
            if self.current_topic != "general_law":
                self.topic_history.append(self.current_topic)
            self.current_topic = turn.topic
    
    def get_recent_context(self, max_turns: int = 2) -> str:
        """Get recent conversation context"""
        if not self.turns:
            return ""
        
        recent_turns = self.turns[-max_turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"Q: {turn.query}")
            context_parts.append(f"A: {turn.response[:200]}...")
        
        return "\n".join(context_parts)
    
    def is_follow_up_query(self, new_query: str, new_topic: str) -> bool:
        """Determine if this is a follow-up query"""
        if not self.turns:
            return False
        
        last_turn = self.turns[-1]
        time_diff = (datetime.now() - last_turn.timestamp).total_seconds()
        
        # Consider follow-up if:
        # 1. Same topic and recent (within 5 minutes)
        # 2. Or contains reference words
        is_same_topic = new_topic == last_turn.topic
        is_recent = time_diff < 300  # 5 minutes
        
        reference_words = ["this", "that", "above", "mentioned", "same", "also", "further", "more"]
        has_reference = any(word in new_query.lower() for word in reference_words)
        
        return (is_same_topic and is_recent) or has_reference

class AdvancedTopicClassifier:
    """Enhanced topic classification with confidence scoring"""
    
    def __init__(self):
        self.legal_topics = {
            "contract_law": {
                "keywords": [
                    "contract", "agreement", "offer", "acceptance", "consideration", 
                    "essential elements", "valid contract", "indian contract act", 
                    "section 10", "breach", "damages", "void", "voidable", "capacity",
                    "free consent", "lawful object", "specific performance"
                ],
                "sections": ["10", "11", "13", "14", "15", "16", "17", "18", "23"],
                "acts": ["indian contract act 1872"]
            },
            "criminal_law": {
                "keywords": [
                    "ipc", "indian penal code", "crpc", "code of criminal procedure",
                    "bail", "bailable", "non-bailable", "anticipatory bail", "arrest",
                    "cognizable", "non-cognizable", "fir", "charge sheet", "investigation",
                    "murder", "theft", "rape", "kidnapping", "cheating", "fraud",
                    "section 302", "section 420", "section 436", "section 437", "section 438"
                ],
                "sections": ["302", "420", "436", "437", "438", "154", "161", "173", "299", "300"],
                "acts": ["indian penal code 1860", "code of criminal procedure 1973"]
            },
            "company_law": {
                "keywords": [
                    "company", "companies act", "annual return", "director", "roc",
                    "registrar", "compliance", "mgr", "aoc", "section 92", "section 137",
                    "board meeting", "agm", "egm", "shares", "shareholders", "dividend",
                    "incorporation", "winding up", "liquidation", "merger", "acquisition",
                    "section 164", "section 248", "disqualification"
                ],
                "sections": ["92", "137", "164", "248", "173", "96", "100", "149"],
                "acts": ["companies act 2013"]
            },
            "constitutional_law": {
                "keywords": [
                    "constitution", "article", "fundamental rights", "directive principles",
                    "article 14", "article 19", "article 21", "supreme court", "high court",
                    "constitutional", "amendment", "judicial review", "separation of powers",
                    "federalism", "emergency", "president", "parliament", "legislature"
                ],
                "sections": ["14", "19", "21", "32", "226", "356", "370"],
                "acts": ["constitution of india 1950"]
            },
            "property_law": {
                "keywords": [
                    "property", "land", "registration", "sale deed", "mortgage", "ownership",
                    "title", "easement", "lease", "rent", "tenant", "landlord", "possession",
                    "transfer of property", "stamp duty", "registration act", "immovable"
                ],
                "sections": ["54", "58", "105", "107", "17", "18"],
                "acts": ["transfer of property act 1882", "registration act 1908"]
            }
        }
    
    def classify_query(self, query: str, context: str = "") -> Tuple[str, float]:
        """Classify query with context awareness"""
        combined_text = f"{context} {query}".lower()
        topic_scores = {}
        
        for topic, data in self.legal_topics.items():
            score = 0.0
            
            # Keyword matching with weights
            for keyword in data["keywords"]:
                if keyword in combined_text:
                    # Higher weight for exact phrase matches
                    if f" {keyword} " in f" {combined_text} ":
                        score += 2.0
                    else:
                        score += 1.0
            
            # Act name matching (high weight)
            for act in data["acts"]:
                act_words = act.split()
                if all(word in combined_text for word in act_words):
                    score += 3.0
            
            # Section number matching (very high weight)
            for section in data["sections"]:
                if f"section {section}" in combined_text or f"sec {section}" in combined_text:
                    score += 5.0
            
            topic_scores[topic] = score
        
        # Find best topic
        if not topic_scores or max(topic_scores.values()) == 0:
            return "general_law", 0.0
        
        best_topic = max(topic_scores, key=topic_scores.get)
        total_score = sum(topic_scores.values())
        confidence = topic_scores[best_topic] / total_score if total_score > 0 else 0.0
        
        return best_topic, confidence
    
    def detect_topic_change(self, current_query: str, last_query: str, last_topic: str) -> Tuple[bool, str, str]:
        """Detect if topic has changed between queries"""
        if not last_query:
            current_topic, _ = self.classify_query(current_query)
            return True, current_topic, "general_law"
        
        current_topic, current_conf = self.classify_query(current_query)
        
        # Topic changed if different topics with reasonable confidence
        topic_changed = (current_topic != last_topic and current_conf > 0.3)
        
        return topic_changed, current_topic, last_topic

class CrossEncoderReranker:
    """Cross-encoder reranking for better relevance"""
    
    def __init__(self):
        # Simplified cross-encoder using TF-IDF similarity
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            lowercase=True
        )
    
    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank documents using cross-encoder approach"""
        if not documents:
            return []
        
        try:
            # Prepare document texts
            doc_texts = []
            for doc in documents:
                text = f"{doc.get('question', '')} {doc.get('answer', '')} {doc.get('context', '')}"
                doc_texts.append(text)
            
            # Add query to corpus for vectorization
            all_texts = [query] + doc_texts
            
            # Vectorize
            vectors = self.vectorizer.fit_transform(all_texts)
            query_vector = vectors[0:1]
            doc_vectors = vectors[1:]
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # Create scored documents
            scored_docs = []
            for i, doc in enumerate(documents):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(similarities[i])
                scored_docs.append(doc_copy)
            
            # Sort by rerank score
            scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return scored_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return documents[:top_k]

class ConversationMemoryManager:
    """Manages conversation sessions and memory"""
    
    def __init__(self, max_sessions: int = 1000, session_timeout_hours: int = 24):
        self.sessions: Dict[str, ConversationSession] = {}
        self.max_sessions = max_sessions
        self.session_timeout = timedelta(hours=session_timeout_hours)
        self.topic_classifier = AdvancedTopicClassifier()
    
    def get_or_create_session(self, session_id: str = None) -> ConversationSession:
        """Get existing session or create new one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Clean up old sessions
        self._cleanup_old_sessions()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                turns=[],
                current_topic="general_law",
                topic_history=[],
                created_at=datetime.now(),
                last_activity=datetime.now()
            )
        
        return self.sessions[session_id]
    
    def add_conversation_turn(self, session_id: str, query: str, response: str, 
                            topic: str, confidence: float, sources: List[str]):
        """Add a conversation turn to session"""
        session = self.get_or_create_session(session_id)
        
        turn = ConversationTurn(
            query=query,
            response=response,
            topic=topic,
            confidence=confidence,
            timestamp=datetime.now(),
            sources=sources
        )
        
        session.add_turn(turn)
    
    def should_use_context(self, session_id: str, current_query: str, current_topic: str) -> Tuple[bool, str]:
        """Determine if conversation context should be used"""
        if session_id not in self.sessions:
            return False, ""
        
        session = self.sessions[session_id]
        
        if session.is_follow_up_query(current_query, current_topic):
            context = session.get_recent_context(max_turns=2)
            return True, context
        
        return False, ""
    
    def _cleanup_old_sessions(self):
        """Remove old sessions to prevent memory bloat"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        # Also limit total sessions
        if len(self.sessions) > self.max_sessions:
            # Remove oldest sessions
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].last_activity
            )
            
            sessions_to_remove = len(self.sessions) - self.max_sessions
            for i in range(sessions_to_remove):
                session_id = sorted_sessions[i][0]
                del self.sessions[session_id]

class AdvancedConversationRAG:
    """Complete conversation-aware RAG pipeline"""
    
    def __init__(self, knowledge_base: List[Dict], gemini_api_key: str = None):
        self.knowledge_base = knowledge_base
        self.topic_classifier = AdvancedTopicClassifier()
        self.reranker = CrossEncoderReranker()
        self.memory_manager = ConversationMemoryManager()
        
        # Configure Gemini if API key provided
        self.gemini_model = None
        if gemini_api_key and gemini_api_key != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ":
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        logger.info(f"Initialized Advanced Conversation RAG with {len(knowledge_base)} documents")
    
    async def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Main query processing with conversation awareness"""
        try:
            # Step 1: Get or create session
            session = self.memory_manager.get_or_create_session(session_id)
            
            # Step 2: Check for conversation context
            use_context, context = self.memory_manager.should_use_context(
                session.session_id, query, session.current_topic
            )
            
            # Step 3: Topic classification with context
            classification_text = f"{context} {query}" if use_context else query
            query_topic, topic_confidence = self.topic_classifier.classify_query(classification_text)
            
            # Step 4: Detect topic change
            last_query = session.turns[-1].query if session.turns else ""
            topic_changed, _, _ = self.topic_classifier.detect_topic_change(
                query, last_query, session.current_topic
            )
            
            logger.info(f"Query: '{query}' | Topic: {query_topic} | Confidence: {topic_confidence:.2f} | Context: {use_context} | Topic Changed: {topic_changed}")
            
            # Step 5: Retrieve documents with topic filtering
            relevant_docs = self._retrieve_documents(query, query_topic, context if use_context else "")
            
            # Step 6: Cross-encoder reranking
            if relevant_docs:
                relevant_docs = self.reranker.rerank(query, relevant_docs, top_k=5)
            
            # Step 7: Generate response
            if relevant_docs and relevant_docs[0].get('rerank_score', 0) > 0.1:
                response = await self._generate_response(query, relevant_docs, query_topic, context if use_context else "")
                confidence = min(relevant_docs[0].get('rerank_score', 0.5) + topic_confidence * 0.3, 0.9)
            else:
                response = self._fallback_response(query, query_topic)
                confidence = 0.1
            
            # Step 8: Update conversation memory
            sources = [doc.get("question", "") for doc in relevant_docs[:3]]
            self.memory_manager.add_conversation_turn(
                session.session_id, query, response, query_topic, confidence, sources
            )
            
            return {
                "response": response,
                "confidence": confidence,
                "topic": query_topic,
                "topic_confidence": topic_confidence,
                "sources": sources,
                "retrieved_count": len(relevant_docs),
                "used_context": use_context,
                "topic_changed": topic_changed,
                "session_id": session.session_id,
                "conversation_turns": len(session.turns)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your query. Please try again or consult a qualified legal professional.",
                "confidence": 0.0,
                "topic": "error",
                "sources": [],
                "error": str(e)
            }
    
    def _retrieve_documents(self, query: str, query_topic: str, context: str = "") -> List[Dict]:
        """Retrieve documents with advanced filtering"""
        query_lower = query.lower()
        context_lower = context.lower()
        combined_query = f"{context} {query}".lower()
        
        scored_items = []
        
        for item in self.knowledge_base:
            question = item.get("question", "")
            answer = item.get("answer", "")
            item_context = item.get("context", "")
            category = item.get("category", "")
            
            # Topic filtering
            item_text = f"{question} {answer} {item_context}".lower()
            item_topic, _ = self.topic_classifier.classify_query(item_text)
            
            # Skip if topics don't match (unless general query)
            if (query_topic != "general_law" and 
                item_topic != "general_law" and 
                query_topic != item_topic and 
                category != query_topic):
                continue
            
            # Calculate relevance score
            score = 0.0
            
            # Question similarity
            question_words = set(question.lower().split())
            query_words = set(combined_query.split())
            question_overlap = len(question_words.intersection(query_words))
            score += question_overlap * 0.3
            
            # Answer keyword matching
            for word in query_words:
                if len(word) > 3 and word in answer.lower():
                    score += 0.2
            
            # Topic boost
            if query_topic == item_topic or category == query_topic:
                score += 1.0
            
            # Section number matching
            query_sections = re.findall(r'section\s*(\d+)', combined_query)
            item_sections = item.get("sections", [])
            if query_sections and item_sections:
                section_matches = len(set(query_sections).intersection(set(str(s) for s in item_sections)))
                score += section_matches * 0.5
            
            # Context relevance (if using conversation context)
            if context and any(word in item_text for word in context_lower.split() if len(word) > 3):
                score += 0.3
            
            if score > 0:
                scored_items.append({
                    **item,
                    "retrieval_score": score
                })
        
        # Sort by score and return top matches
        scored_items.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return scored_items[:10]  # Return more for reranking
    
    async def _generate_response(self, query: str, docs: List[Dict], topic: str, context: str = "") -> str:
        """Generate response using AI or templates"""
        if self.gemini_model:
            return await self._generate_ai_response(query, docs, topic, context)
        else:
            return self._generate_template_response(query, docs, topic)
    
    async def _generate_ai_response(self, query: str, docs: List[Dict], topic: str, context: str = "") -> str:
        """Generate AI response using Gemini with conversation awareness"""
        try:
            # Prepare context from retrieved documents
            context_docs = "\n\n".join([
                f"Document {i+1}:\nQ: {doc.get('question', '')}\nA: {doc.get('answer', '')[:400]}..."
                for i, doc in enumerate(docs[:3])
            ])
            
            conversation_context = f"\n\nConversation Context:\n{context}" if context else ""
            
            prompt = f"""You are an expert Indian legal assistant with conversation awareness. Answer the query using ONLY the provided legal documents.

CRITICAL INSTRUCTIONS:
1. Answer ONLY about {topic.replace('_', ' ').title()}
2. Use ONLY information from the provided documents
3. Consider the conversation context if provided
4. Cite specific sections and acts mentioned in documents
5. If this is a follow-up question, build upon previous context appropriately
6. Do not mix information from different legal domains

CURRENT QUERY: {query}
{conversation_context}

RELEVANT LEGAL DOCUMENTS:
{context_docs}

Provide a comprehensive answer following this format:
⚖️ **[Topic] - [Specific Subject]**

📘 **Overview**: 
[Brief explanation based on documents]

📜 **Legal Provisions**:
• Act/Law: [From documents]
• Section(s): [From documents]

💼 **Key Points**:
[Main legal points from documents]

🛠️ **Practical Application**:
[How this applies in practice]

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified advocate for specific legal matters."""

            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            if response and hasattr(response, 'text') and response.text:
                return response.text
            else:
                return self._generate_template_response(query, docs, topic)
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_template_response(query, docs, topic)
    
    def _generate_template_response(self, query: str, docs: List[Dict], topic: str) -> str:
        """Generate template-based response"""
        query_lower = query.lower()
        
        # Use existing template logic from main.py
        if topic == "contract_law" and "essential elements" in query_lower:
            return """⚖️ **Essential Elements of Valid Contract - Indian Contract Act, 1872**

📘 **Overview**: 
Under Section 10 of the Indian Contract Act, 1872, a valid contract must contain all essential elements for legal enforceability.

📜 **Legal Provisions**:
• Act/Law: Indian Contract Act, 1872
• Section(s): 10, 11, 13-22, 23
• Key provision: "All agreements are contracts if made by free consent of competent parties for lawful consideration and lawful object"

💼 **Essential Elements (Section 10)**:
• **Offer & Acceptance**: Clear proposal and unconditional acceptance
• **Lawful Consideration**: Something valuable given in exchange
• **Capacity of Parties**: Parties must be of sound mind, major (18+), not disqualified by law
• **Free Consent**: No coercion, undue influence, fraud, misrepresentation, or mistake
• **Lawful Object**: Purpose must be legal and not against public policy
• **Not Declared Void**: Must not fall under void agreements (Sections 24-30)

🛠️ **Practical Application**:
• ALL elements must be present for validity
• Missing any element makes contract invalid
• Courts examine each element separately

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified advocate for specific legal matters."""
        
        # Add other template responses...
        if docs:
            primary_doc = docs[0]
            return f"""⚖️ **Legal Response**

📘 **Overview**: 
{primary_doc.get('answer', '')[:400]}...

📜 **Source**: 
{primary_doc.get('act', 'Legal Database')}

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified advocate for specific legal matters."""
        
        return self._fallback_response(query, topic)
    
    def _fallback_response(self, query: str, topic: str) -> str:
        """Fallback response when no relevant documents found"""
        return f"""⚖️ **Legal Query Response**

📘 **Overview**: 
I don't have sufficient information in my legal database to provide an accurate answer for your {topic.replace('_', ' ')} query.

🛑 **Recommendation**: 
Please consult with a qualified legal professional who specializes in {topic.replace('_', ' ')} for detailed guidance on this matter.

📌 **Alternative**: 
You may also try rephrasing your question with more specific legal terms or section numbers."""

# Export the main class
__all__ = ['AdvancedConversationRAG']