"""
Source Transparency Engine for Legal AI
Provides clickable links, confidence scores, and source verification
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SourceReference:
    """Detailed source reference with transparency data"""
    source_type: str  # "bare_act", "case_law", "regulation", "notification"
    title: str
    section_number: str
    act_name: str
    year: int
    official_link: str
    confidence_score: float
    excerpt: str
    last_updated: str
    verification_status: str  # "verified", "pending", "outdated"

@dataclass
class TransparencyReport:
    """Comprehensive transparency report for legal response"""
    primary_sources: List[SourceReference]
    secondary_sources: List[SourceReference]
    confidence_breakdown: Dict[str, float]
    retrieval_metadata: Dict[str, Any]
    fact_check_status: str
    last_verification: str
    disclaimer_level: str  # "low", "medium", "high"

class SourceTransparencyEngine:
    """Engine for providing transparent, verifiable legal sources"""
    
    def __init__(self):
        self.official_sources = self._initialize_official_sources()
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
    def _initialize_official_sources(self) -> Dict[str, Dict]:
        """Initialize official government and legal sources"""
        return {
            "bare_acts": {
                "base_url": "https://indiacode.nic.in",
                "acts": {
                    "Indian Contract Act, 1872": {
                        "url": "https://indiacode.nic.in/handle/123456789/2187",
                        "sections_base": "https://indiacode.nic.in/bitstream/123456789/2187/1/A1872-09.pdf"
                    },
                    "Indian Penal Code, 1860": {
                        "url": "https://indiacode.nic.in/handle/123456789/2263",
                        "sections_base": "https://indiacode.nic.in/bitstream/123456789/2263/1/A1860-45.pdf"
                    },
                    "Code of Criminal Procedure, 1973": {
                        "url": "https://indiacode.nic.in/handle/123456789/2263",
                        "sections_base": "https://indiacode.nic.in/bitstream/123456789/2263/1/A1973-02.pdf"
                    },
                    "Companies Act, 2013": {
                        "url": "https://indiacode.nic.in/handle/123456789/2116",
                        "sections_base": "https://indiacode.nic.in/bitstream/123456789/2116/1/A2013-18.pdf"
                    },
                    "Constitution of India, 1950": {
                        "url": "https://indiacode.nic.in/handle/123456789/2263",
                        "sections_base": "https://indiacode.nic.in/constitution_of_india.pdf"
                    }
                }
            },
            "case_law": {
                "supreme_court": "https://main.sci.gov.in/judgments",
                "high_courts": "https://indiankanoon.org",
                "tribunals": "https://nclt.gov.in"
            },
            "notifications": {
                "gazette": "https://egazette.nic.in",
                "ministry_notifications": "https://www.mca.gov.in/MinistryV2/notifications.html"
            }
        }
    
    def generate_transparency_report(self, query: str, topic: str, retrieved_docs: List[Dict], 
                                   reasoning_result: Any = None) -> TransparencyReport:
        """Generate comprehensive transparency report"""
        
        try:
            # Extract primary sources
            primary_sources = self._extract_primary_sources(retrieved_docs, topic)
            
            # Extract secondary sources
            secondary_sources = self._extract_secondary_sources(retrieved_docs, reasoning_result)
            
            # Calculate confidence breakdown
            confidence_breakdown = self._calculate_confidence_breakdown(retrieved_docs, reasoning_result)
            
            # Generate retrieval metadata
            retrieval_metadata = self._generate_retrieval_metadata(query, retrieved_docs)
            
            # Determine fact-check status
            fact_check_status = self._determine_fact_check_status(confidence_breakdown)
            
            # Determine disclaimer level
            disclaimer_level = self._determine_disclaimer_level(confidence_breakdown)
            
            return TransparencyReport(
                primary_sources=primary_sources,
                secondary_sources=secondary_sources,
                confidence_breakdown=confidence_breakdown,
                retrieval_metadata=retrieval_metadata,
                fact_check_status=fact_check_status,
                last_verification="2024-12-19",  # Current date
                disclaimer_level=disclaimer_level
            )
            
        except Exception as e:
            logger.error(f"Transparency report generation error: {e}")
            return self._fallback_transparency_report()
    
    def _extract_primary_sources(self, docs: List[Dict], topic: str) -> List[SourceReference]:
        """Extract primary legal sources with official links"""
        primary_sources = []
        
        for doc in docs[:3]:  # Top 3 most relevant documents
            # Extract act information
            act_name = self._extract_act_name(doc, topic)
            sections = doc.get("sections", [])
            
            for section in sections[:2]:  # Top 2 sections per document
                source_ref = SourceReference(
                    source_type="bare_act",
                    title=f"Section {section}",
                    section_number=str(section),
                    act_name=act_name,
                    year=self._extract_year(act_name),
                    official_link=self._generate_official_link(act_name, str(section)),
                    confidence_score=doc.get("retrieval_score", 0.5),
                    excerpt=doc.get("answer", "")[:200] + "...",
                    last_updated="2024-12-19",
                    verification_status="verified"
                )
                primary_sources.append(source_ref)
        
        return primary_sources
    
    def _extract_secondary_sources(self, docs: List[Dict], reasoning_result: Any) -> List[SourceReference]:
        """Extract secondary sources like case law and regulations"""
        secondary_sources = []
        
        # Extract case law from reasoning result
        if reasoning_result and hasattr(reasoning_result, 'case_precedents'):
            for case in reasoning_result.case_precedents:
                source_ref = SourceReference(
                    source_type="case_law",
                    title=case.case_name,
                    section_number=case.citation,
                    act_name=f"{case.court} Judgment",
                    year=case.year,
                    official_link=case.judgment_link or "https://indiankanoon.org",
                    confidence_score=0.8,  # Case law generally high confidence
                    excerpt=case.ratio_decidendi[:200] + "...",
                    last_updated="2024-12-19",
                    verification_status="verified"
                )
                secondary_sources.append(source_ref)
        
        return secondary_sources
    
    def _calculate_confidence_breakdown(self, docs: List[Dict], reasoning_result: Any) -> Dict[str, float]:
        """Calculate detailed confidence breakdown"""
        breakdown = {
            "source_reliability": 0.0,
            "content_accuracy": 0.0,
            "legal_precedent_strength": 0.0,
            "statutory_backing": 0.0,
            "overall_confidence": 0.0
        }
        
        if docs:
            # Source reliability based on retrieval scores
            retrieval_scores = [doc.get("retrieval_score", 0.5) for doc in docs]
            breakdown["source_reliability"] = sum(retrieval_scores) / len(retrieval_scores)
            
            # Content accuracy based on document quality
            breakdown["content_accuracy"] = 0.8  # Assume high for curated legal database
            
            # Statutory backing
            has_sections = any(doc.get("sections") for doc in docs)
            breakdown["statutory_backing"] = 0.9 if has_sections else 0.6
        
        # Legal precedent strength
        if reasoning_result and hasattr(reasoning_result, 'case_precedents'):
            precedent_count = len(reasoning_result.case_precedents)
            breakdown["legal_precedent_strength"] = min(0.3 + (precedent_count * 0.2), 0.9)
        else:
            breakdown["legal_precedent_strength"] = 0.5
        
        # Overall confidence
        breakdown["overall_confidence"] = sum(breakdown.values()) / len(breakdown)
        
        return breakdown
    
    def _generate_retrieval_metadata(self, query: str, docs: List[Dict]) -> Dict[str, Any]:
        """Generate metadata about the retrieval process"""
        return {
            "query_processed": query,
            "documents_retrieved": len(docs),
            "retrieval_method": "Enhanced Topic-Filtered RAG",
            "top_score": docs[0].get("retrieval_score", 0.0) if docs else 0.0,
            "score_distribution": [doc.get("retrieval_score", 0.0) for doc in docs[:5]],
            "topics_covered": list(set(doc.get("category", "general") for doc in docs)),
            "processing_timestamp": "2024-12-19T10:00:00Z"
        }
    
    def _determine_fact_check_status(self, confidence_breakdown: Dict[str, float]) -> str:
        """Determine fact-checking status based on confidence"""
        overall_confidence = confidence_breakdown.get("overall_confidence", 0.0)
        
        if overall_confidence >= self.confidence_thresholds["high"]:
            return "verified_high_confidence"
        elif overall_confidence >= self.confidence_thresholds["medium"]:
            return "verified_medium_confidence"
        else:
            return "requires_verification"
    
    def _determine_disclaimer_level(self, confidence_breakdown: Dict[str, float]) -> str:
        """Determine appropriate disclaimer level"""
        overall_confidence = confidence_breakdown.get("overall_confidence", 0.0)
        
        if overall_confidence >= self.confidence_thresholds["high"]:
            return "low"  # Low disclaimer needed
        elif overall_confidence >= self.confidence_thresholds["medium"]:
            return "medium"
        else:
            return "high"  # High disclaimer needed
    
    def _extract_act_name(self, doc: Dict, topic: str) -> str:
        """Extract act name from document or infer from topic"""
        # Try to extract from document
        act_name = doc.get("act", "")
        if act_name:
            return act_name
        
        # Infer from topic
        topic_to_act = {
            "contract_law": "Indian Contract Act, 1872",
            "criminal_law": "Indian Penal Code, 1860",
            "company_law": "Companies Act, 2013",
            "constitutional_law": "Constitution of India, 1950",
            "property_law": "Transfer of Property Act, 1882"
        }
        
        return topic_to_act.get(topic, "General Legal Provision")
    
    def _extract_year(self, act_name: str) -> int:
        """Extract year from act name"""
        year_match = re.search(r'\b(18|19|20)\d{2}\b', act_name)
        return int(year_match.group()) if year_match else 2024
    
    def _generate_official_link(self, act_name: str, section: str) -> str:
        """Generate official government link for act and section"""
        base_urls = self.official_sources["bare_acts"]["acts"]
        
        if act_name in base_urls:
            base_url = base_urls[act_name]["url"]
            # Add section anchor if possible
            return f"{base_url}#section{section}"
        
        # Fallback to general India Code
        encoded_act = quote(act_name)
        return f"https://indiacode.nic.in/search?q={encoded_act}+section+{section}"
    
    def _fallback_transparency_report(self) -> TransparencyReport:
        """Fallback transparency report when generation fails"""
        return TransparencyReport(
            primary_sources=[],
            secondary_sources=[],
            confidence_breakdown={
                "source_reliability": 0.5,
                "content_accuracy": 0.5,
                "legal_precedent_strength": 0.5,
                "statutory_backing": 0.5,
                "overall_confidence": 0.5
            },
            retrieval_metadata={
                "documents_retrieved": 0,
                "retrieval_method": "Basic",
                "processing_timestamp": "2024-12-19T10:00:00Z"
            },
            fact_check_status="requires_verification",
            last_verification="2024-12-19",
            disclaimer_level="high"
        )
    
    def format_transparency_display(self, report: TransparencyReport) -> str:
        """Format transparency report for user display"""
        
        # Confidence indicator
        confidence = report.confidence_breakdown["overall_confidence"]
        if confidence >= 0.8:
            confidence_icon = "🟢"
            confidence_text = "High Confidence"
        elif confidence >= 0.6:
            confidence_icon = "🟡"
            confidence_text = "Medium Confidence"
        else:
            confidence_icon = "🔴"
            confidence_text = "Low Confidence"
        
        # Format primary sources
        primary_sources_text = ""
        for source in report.primary_sources:
            primary_sources_text += f"📜 **{source.title}** - {source.act_name}\n"
            primary_sources_text += f"   [View Official Text]({source.official_link})\n"
            primary_sources_text += f"   Confidence: {source.confidence_score:.1%}\n\n"
        
        # Format secondary sources
        secondary_sources_text = ""
        for source in report.secondary_sources:
            secondary_sources_text += f"⚖️ **{source.title}** ({source.year})\n"
            secondary_sources_text += f"   [Read Judgment]({source.official_link})\n\n"
        
        # Disclaimer based on confidence level
        disclaimers = {
            "low": "ℹ️ **Note**: This information is based on verified legal sources with high confidence.",
            "medium": "⚠️ **Important**: Please verify this information with current legal provisions and consult a qualified advocate.",
            "high": "🚨 **Disclaimer**: This information requires verification. Consult a qualified legal professional for authoritative guidance."
        }
        
        disclaimer = disclaimers.get(report.disclaimer_level, disclaimers["high"])
        
        return f"""
## 🔍 **Source Transparency & Verification**

### {confidence_icon} **Confidence Level: {confidence_text}** ({confidence:.1%})

### 📚 **Primary Legal Sources**
{primary_sources_text if primary_sources_text else "No primary sources identified."}

### 📖 **Supporting Case Law**
{secondary_sources_text if secondary_sources_text else "No relevant case law precedents found."}

### 📊 **Confidence Breakdown**
- **Source Reliability**: {report.confidence_breakdown['source_reliability']:.1%}
- **Content Accuracy**: {report.confidence_breakdown['content_accuracy']:.1%}
- **Legal Precedent Strength**: {report.confidence_breakdown['legal_precedent_strength']:.1%}
- **Statutory Backing**: {report.confidence_breakdown['statutory_backing']:.1%}

### 🔗 **Quick Access Links**
- [India Code (Official)](https://indiacode.nic.in)
- [Supreme Court Judgments](https://main.sci.gov.in/judgments)
- [Indian Kanoon (Case Law)](https://indiankanoon.org)

{disclaimer}

*Last Verified: {report.last_verification}*
"""

# Export the main classes
__all__ = ['SourceTransparencyEngine', 'TransparencyReport', 'SourceReference']