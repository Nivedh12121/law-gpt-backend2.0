"""
Advanced Topic-Switch-Safe RAG Pipeline for Law GPT
Implements multi-stage retrieval with conversation memory and topic classification
"""

import os
import json
import re
import math
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Stores conversation context and topic history"""
    session_id: str
    current_topic: str
    last_query: str
    last_response: str
    topic_history: List[str]
    timestamp: datetime

@dataclass
class RetrievalResult:
    """Structured retrieval result with metadata"""
    content: str
    score: float
    topic: str
    act: str
    sections: List[str]
    metadata: Dict[str, Any]

class TopicClassifier:
    """Advanced topic classification for legal queries"""
    
    def __init__(self):
        self.legal_topics = {
            "contract_law": {
                "keywords": [
                    "contract", "agreement", "offer", "acceptance", "consideration", 
                    "essential elements", "valid contract", "indian contract act", 
                    "section 10", "breach", "damages", "specific performance",
                    "void", "voidable", "unenforceable", "capacity", "consent",
                    "contract review", "review contract", "contract analysis",
                    "contract terms", "contract clauses", "contract drafting",
                    "employment contract", "service contract", "sale contract"
                ],
                "acts": ["indian contract act 1872"],
                "sections": ["10", "11", "13", "14", "15", "16", "17", "18", "23"]
            },
            "criminal_law": {
                "keywords": [
                    "ipc", "indian penal code", "crpc", "code of criminal procedure",
                    "bail", "bailable", "non-bailable", "anticipatory bail", "arrest",
                    "cognizable", "non-cognizable", "fir", "charge sheet", "investigation",
                    "murder", "theft", "rape", "kidnapping", "section 302", "section 420",
                    "section 436", "section 437", "section 438", "criminal", "punishment"
                ],
                "acts": ["indian penal code 1860", "code of criminal procedure 1973"],
                "sections": ["302", "420", "436", "437", "438", "154", "161", "173"]
            },
            "company_law": {
                "keywords": [
                    "company", "companies act", "annual return", "director", "roc",
                    "registrar", "compliance", "mgr", "aoc", "section 92", "section 137",
                    "board meeting", "agm", "egm", "shares", "shareholders", "dividend",
                    "incorporation", "winding up", "liquidation", "merger", "acquisition"
                ],
                "acts": ["companies act 2013"],
                "sections": ["92", "137", "164", "248", "173", "96", "100", "149"]
            },
            "constitutional_law": {
                "keywords": [
                    "constitution", "article", "fundamental rights", "directive principles",
                    "article 14", "article 19", "article 21", "supreme court", "high court",
                    "constitutional", "amendment", "judicial review", "separation of powers",
                    "federalism", "emergency", "president", "parliament", "legislature"
                ],
                "acts": ["constitution of india 1950"],
                "sections": ["14", "19", "21", "32", "226", "356", "370"]
            },
            "property_law": {
                "keywords": [
                    "property", "land", "registration", "sale deed", "mortgage", "ownership",
                    "title", "easement", "lease", "rent", "tenant", "landlord", "possession",
                    "transfer of property", "stamp duty", "registration act", "immovable"
                ],
                "acts": ["transfer of property act 1882", "registration act 1908"],
                "sections": ["54", "58", "105", "107", "17", "18"]
            },
            "general_law": {
                "keywords": [
                    "legal rights", "rights", "legal help", "legal advice", "legal assistance",
                    "help me with", "what are my rights", "legal guidance", "legal support",
                    "legal consultation", "legal query", "legal question", "law help",
                    "legal information", "legal matter", "legal issue", "legal problem"
                ],
                "acts": ["constitution of india 1950", "indian contract act 1872"],
                "sections": ["14", "19", "21", "10"]
            }
        }
    
    def classify_query(self, query: str) -> Tuple[str, float]:
        """Classify query into legal topic with confidence score"""
        query_lower = query.lower()
        topic_scores = {}
        
        for topic, data in self.legal_topics.items():
            score = 0.0
            
            # Keyword matching with weights
            for keyword in data["keywords"]:
                if keyword in query_lower:
                    # Higher weight for exact matches
                    if f" {keyword} " in f" {query_lower} ":
                        score += 2.0
                    else:
                        score += 1.0
            
            # Act name matching (high weight)
            for act in data["acts"]:
                if any(word in query_lower for word in act.split()):
                    score += 3.0
            
            # Section number matching (very high weight)
            for section in data["sections"]:
                if f"section {section}" in query_lower or f"sec {section}" in query_lower:
                    score += 5.0
            
            topic_scores[topic] = score
        
        # Find best topic
        if not topic_scores or max(topic_scores.values()) == 0:
            # If no keywords match, try to infer from query structure
            if any(word in query_lower for word in ["contract", "agreement", "review"]):
                return "contract_law", 0.5
            elif any(word in query_lower for word in ["rights", "help", "legal"]):
                return "general_law", 0.5
            elif any(word in query_lower for word in ["company", "director", "annual"]):
                return "company_law", 0.5
            else:
                return "general_law", 0.3
        
        best_topic = max(topic_scores, key=topic_scores.get)
        total_score = sum(topic_scores.values())
        confidence = topic_scores[best_topic] / total_score if total_score > 0 else 0.5
        
        # Ensure minimum confidence for matched topics
        if confidence < 0.3:
            confidence = 0.5
        
        return best_topic, confidence
    
    def detect_topic_change(self, current_query: str, last_query: str) -> Tuple[bool, str, str]:
        """Detect if topic has changed between queries"""
        if not last_query:
            current_topic, _ = self.classify_query(current_query)
            return True, current_topic, "general_law"
        
        current_topic, current_conf = self.classify_query(current_query)
        last_topic, last_conf = self.classify_query(last_query)
        
        # Topic changed if different topics with reasonable confidence
        topic_changed = (current_topic != last_topic and 
                        current_conf > 0.3 and 
                        last_conf > 0.3)
        
        return topic_changed, current_topic, last_topic

class AdvancedRetriever:
    """Multi-stage retrieval with topic filtering and reranking"""
    
    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.topic_classifier = TopicClassifier()
        self.vectorizer = None
        self.doc_vectors = None
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index for fast retrieval"""
        documents = []
        for item in self.knowledge_base:
            doc_text = f"{item.get('question', '')} {item.get('answer', '')} {item.get('context', '')}"
            documents.append(doc_text)
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        if documents:
            self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        logger.info(f"Built retrieval index with {len(documents)} documents")
    
    def _topic_filter(self, query_topic: str, top_k: int = 50) -> List[int]:
        """Filter documents by topic before similarity search"""
        if query_topic == "general_law":
            # For general law, include more diverse documents
            return list(range(min(top_k * 2, len(self.knowledge_base))))
        
        filtered_indices = []
        fallback_indices = []
        
        for i, item in enumerate(self.knowledge_base):
            item_category = item.get("category", "")
            item_text = f"{item.get('question', '')} {item.get('answer', '')} {item.get('context', '')}"
            item_topic, confidence = self.topic_classifier.classify_query(item_text)
            
            # Primary matches - exact topic match
            if (item_topic == query_topic or 
                item_category == query_topic):
                filtered_indices.append(i)
            # Secondary matches - partial relevance
            elif (query_topic in item_text.lower() or 
                  confidence > 0.2):
                fallback_indices.append(i)
        
        # Combine primary and fallback results
        combined_indices = filtered_indices + fallback_indices
        
        # If we don't have enough results, include more general documents
        if len(combined_indices) < top_k:
            remaining = top_k - len(combined_indices)
            general_indices = [i for i in range(len(self.knowledge_base)) 
                             if i not in combined_indices][:remaining]
            combined_indices.extend(general_indices)
        
        return combined_indices[:top_k]
    
    def _vector_search(self, query: str, filtered_indices: List[int], top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform vector similarity search on filtered documents"""
        if not self.vectorizer or self.doc_vectors is None:
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities only for filtered documents
            similarities = []
            for idx in filtered_indices:
                if idx < self.doc_vectors.shape[0]:
                    doc_vector = self.doc_vectors[idx:idx+1]
                    sim_matrix = cosine_similarity(query_vector, doc_vector)
                    sim_score = float(sim_matrix[0, 0])
                    similarities.append((idx, sim_score))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            # Fallback to simple scoring
            return [(idx, 0.5) for idx in filtered_indices[:top_k]]
    
    def _rerank_results(self, query: str, candidates: List[Tuple[int, float]]) -> List[RetrievalResult]:
        """Rerank candidates using advanced scoring"""
        results = []
        query_lower = query.lower()
        
        for idx, base_score in candidates:
            if idx >= len(self.knowledge_base):
                continue
                
            item = self.knowledge_base[idx]
            question = item.get("question", "")
            answer = item.get("answer", "")
            context = item.get("context", "")
            
            # Calculate enhanced score
            enhanced_score = base_score
            
            # Question similarity boost
            if question:
                question_words = set(question.lower().split())
                query_words = set(query_lower.split())
                question_overlap = len(question_words.intersection(query_words))
                enhanced_score += question_overlap * 0.1
            
            # Exact phrase matching
            for phrase in re.findall(r'"([^"]*)"', query):
                if phrase.lower() in answer.lower():
                    enhanced_score += 0.3
            
            # Section number matching
            query_sections = re.findall(r'section\s*(\d+)', query_lower)
            item_sections = item.get("sections", [])
            section_matches = len(set(query_sections).intersection(set(item_sections)))
            enhanced_score += section_matches * 0.2
            
            # Create result object
            result = RetrievalResult(
                content=f"Q: {question}\nA: {answer}",
                score=enhanced_score,
                topic=item.get("category", "general_law"),
                act=item.get("act", ""),
                sections=item.get("sections", []),
                metadata=item
            )
            
            results.append(result)
        
        # Sort by enhanced score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def retrieve(self, query: str, query_topic: str, top_k: int = 5) -> List[RetrievalResult]:
        """Main retrieval method with multi-stage pipeline"""
        logger.info(f"Retrieving for query: '{query}' in topic: '{query_topic}'")
        
        # Stage 1: Topic-based filtering
        filtered_indices = self._topic_filter(query_topic, top_k=30)
        logger.info(f"Stage 1: Filtered to {len(filtered_indices)} documents")
        
        if not filtered_indices:
            return []
        
        # Stage 2: Vector similarity search
        vector_results = self._vector_search(query, filtered_indices, top_k=10)
        logger.info(f"Stage 2: Vector search returned {len(vector_results)} candidates")
        
        if not vector_results:
            return []
        
        # Stage 3: Reranking with enhanced scoring
        final_results = self._rerank_results(query, vector_results)
        logger.info(f"Stage 3: Reranked to {len(final_results)} final results")
        
        return final_results[:top_k]

class ConversationManager:
    """Manages conversation context and memory"""
    
    def __init__(self):
        self.contexts: Dict[str, ConversationContext] = {}
        self.topic_classifier = TopicClassifier()
    
    def get_or_create_context(self, session_id: str) -> ConversationContext:
        """Get existing context or create new one"""
        if session_id not in self.contexts:
            self.contexts[session_id] = ConversationContext(
                session_id=session_id,
                current_topic="general_law",
                last_query="",
                last_response="",
                topic_history=[],
                timestamp=datetime.now()
            )
        return self.contexts[session_id]
    
    def update_context(self, session_id: str, query: str, response: str, topic: str):
        """Update conversation context"""
        context = self.get_or_create_context(session_id)
        
        # Detect topic change
        topic_changed, new_topic, old_topic = self.topic_classifier.detect_topic_change(
            query, context.last_query
        )
        
        if topic_changed:
            logger.info(f"Topic changed from {old_topic} to {new_topic}")
            context.topic_history.append(old_topic)
        
        context.current_topic = new_topic
        context.last_query = query
        context.last_response = response
        context.timestamp = datetime.now()
    
    def should_use_context(self, session_id: str, current_topic: str) -> bool:
        """Determine if previous context should be used"""
        if session_id not in self.contexts:
            return False
        
        context = self.contexts[session_id]
        
        # Use context if same topic and recent conversation
        time_diff = (datetime.now() - context.timestamp).total_seconds()
        return (context.current_topic == current_topic and 
                time_diff < 300 and  # 5 minutes
                context.last_response)
    
    def get_context_for_retrieval(self, session_id: str) -> str:
        """Get context string for retrieval augmentation"""
        if session_id not in self.contexts:
            return ""
        
        context = self.contexts[session_id]
        if context.last_query and context.last_response:
            return f"Previous Q: {context.last_query}\nPrevious A: {context.last_response[:200]}..."
        return ""

class AdvancedRAGPipeline:
    """Complete RAG pipeline with topic-switch safety"""
    
    def __init__(self, knowledge_base: List[Dict], gemini_api_key: str = None):
        self.retriever = AdvancedRetriever(knowledge_base)
        self.conversation_manager = ConversationManager()
        self.topic_classifier = TopicClassifier()
        
        # Configure Gemini if API key provided
        self.gemini_model = None
        if gemini_api_key and gemini_api_key != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ":
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
    
    async def process_query(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Main query processing method"""
        try:
            # Step 1: Topic classification
            query_topic, topic_confidence = self.topic_classifier.classify_query(query)
            logger.info(f"Classified query as {query_topic} with confidence {topic_confidence:.2f}")
            
            # Step 2: Check conversation context
            context = self.conversation_manager.get_or_create_context(session_id)
            use_context = self.conversation_manager.should_use_context(session_id, query_topic)
            
            # Step 3: Retrieve relevant documents
            retrieval_query = query
            if use_context:
                context_str = self.conversation_manager.get_context_for_retrieval(session_id)
                retrieval_query = f"{context_str}\n\nCurrent Question: {query}"
            
            retrieved_docs = self.retriever.retrieve(retrieval_query, query_topic, top_k=5)
            
            # Step 4: Generate response
            if retrieved_docs and retrieved_docs[0].score > 0.1:
                response = await self._generate_response(query, retrieved_docs, query_topic)
                confidence = min(retrieved_docs[0].score, 0.9)
            else:
                response = self._fallback_response(query, query_topic)
                confidence = 0.1
            
            # Step 5: Update conversation context
            self.conversation_manager.update_context(session_id, query, response, query_topic)
            
            return {
                "response": response,
                "confidence": confidence,
                "topic": query_topic,
                "topic_confidence": topic_confidence,
                "sources": [doc.metadata.get("question", "") for doc in retrieved_docs[:3]],
                "retrieved_count": len(retrieved_docs),
                "used_context": use_context
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
    
    async def _generate_response(self, query: str, docs: List[RetrievalResult], topic: str) -> str:
        """Generate response using retrieved documents"""
        if self.gemini_model:
            return await self._generate_ai_response(query, docs, topic)
        else:
            return self._generate_template_response(query, docs, topic)
    
    async def _generate_ai_response(self, query: str, docs: List[RetrievalResult], topic: str) -> str:
        """Generate AI response using Gemini"""
        try:
            # Prepare context from retrieved documents
            context_docs = "\n\n".join([
                f"Document {i+1}:\n{doc.content[:500]}..."
                for i, doc in enumerate(docs[:3])
            ])
            
            prompt = f"""You are an expert Indian legal assistant. Answer the query using ONLY the provided legal documents. 

CRITICAL INSTRUCTIONS:
1. Answer ONLY about {topic.replace('_', ' ').title()}
2. Use ONLY information from the provided documents
3. Cite specific sections and acts mentioned in documents
4. If documents don't contain relevant information, say so clearly
5. Do not mix information from different legal domains

QUERY: {query}

RELEVANT LEGAL DOCUMENTS:
{context_docs}

Provide a comprehensive answer following this format:
⚖️ **[Topic] - [Specific Subject]**

📘 **Overview**: 
[Brief explanation based on documents]

📜 **Legal Provisions**:
• Act/Law: [From documents]
• Section(s): [From documents]
• Key provisions: [From documents]

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
    
    def _generate_template_response(self, query: str, docs: List[RetrievalResult], topic: str) -> str:
        """Generate template-based response"""
        if not docs:
            return self._fallback_response(query, topic)
        
        primary_doc = docs[0]
        
        # Topic-specific templates
        if topic == "contract_law":
            return self._contract_law_template(query, primary_doc)
        elif topic == "criminal_law":
            return self._criminal_law_template(query, primary_doc)
        elif topic == "company_law":
            return self._company_law_template(query, primary_doc)
        else:
            return self._general_template(query, primary_doc)
    
    def _contract_law_template(self, query: str, doc: RetrievalResult) -> str:
        """Contract law specific template"""
        if "essential elements" in query.lower():
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
        
        return f"""⚖️ **Contract Law Response**

📘 **Overview**: 
{doc.content[:300]}...

📜 **Source**: 
{doc.act} - {', '.join(doc.sections) if doc.sections else 'General Provisions'}

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified advocate for specific legal matters."""
    
    def _criminal_law_template(self, query: str, doc: RetrievalResult) -> str:
        """Criminal law specific template"""
        if any(term in query.lower() for term in ["bailable", "bail", "non-bailable"]):
            return """⚖️ **Bailable vs Non-Bailable Offences - Code of Criminal Procedure**

📘 **Overview**: 
Under the Code of Criminal Procedure (CrPC), 1973, offences are classified as bailable and non-bailable based on severity and nature of crime.

📜 **Legal Provisions**:
• Act/Law: Code of Criminal Procedure, 1973
• Section(s): 436, 437, 437A, 438
• Key provisions: Bail classification, police powers, court discretion

💼 **BAILABLE OFFENCES (Section 436)**:
• **Right to Bail**: Accused has legal right to bail
• **Police Powers**: Police can grant bail at station level
• **Examples**: Simple hurt, theft under ₹5000, defamation

💼 **NON-BAILABLE OFFENCES (Section 437)**:
• **Discretionary Bail**: Court has discretion to grant or refuse
• **No Police Bail**: Police cannot grant bail
• **Examples**: Murder (IPC 302), rape (IPC 376), kidnapping

🛠️ **Practical Application**:
1. Check offence classification in First Schedule CrPC
2. For bailable: Apply to police or magistrate
3. For non-bailable: Apply to magistrate/sessions court

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified criminal lawyer for specific legal matters."""
        
        return f"""⚖️ **Criminal Law Response**

📘 **Overview**: 
{doc.content[:300]}...

📜 **Source**: 
{doc.act} - {', '.join(doc.sections) if doc.sections else 'General Provisions'}

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified criminal lawyer for specific legal matters."""
    
    def _company_law_template(self, query: str, doc: RetrievalResult) -> str:
        """Company law specific template"""
        if "annual return" in query.lower():
            return """⚖️ **Non-Filing of Annual Returns - Legal Consequences**

📘 **Overview**: 
Non-filing of annual returns results in severe legal consequences under the Companies Act, 2013.

📜 **Legal Provisions**:
• Act/Law: Companies Act, 2013
• Section(s): 92, 137, 164(2), 248
• Key provisions: Mandatory annual return filing, penalties, director disqualification

💼 **Legal Consequences/Penalties**:
• Section 92(5): ₹5 lakh penalty for company + ₹1 lakh per officer in default
• Section 137: ₹500 per day continuing penalty
• Section 164(2): Automatic director disqualification after 3 years
• Section 248: ROC may initiate striking off proceedings

🛠️ **Available Remedies**:
• File all pending annual returns immediately
• Pay prescribed penalties and additional fees
• Apply for removal of director disqualification

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified company law advocate for specific legal matters."""
        
        return f"""⚖️ **Company Law Response**

📘 **Overview**: 
{doc.content[:300]}...

📜 **Source**: 
{doc.act} - {', '.join(doc.sections) if doc.sections else 'General Provisions'}

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified company law advocate for specific legal matters."""
    
    def _general_template(self, query: str, doc: RetrievalResult) -> str:
        """General template for other topics"""
        return f"""⚖️ **Legal Response**

📘 **Overview**: 
{doc.content[:400]}...

📜 **Source**: 
{doc.act if doc.act else 'Legal Database'}

🛑 **Legal Disclaimer**: 
This information is for educational purposes only. Consult a qualified advocate for specific legal matters."""
    
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
__all__ = ['AdvancedRAGPipeline']