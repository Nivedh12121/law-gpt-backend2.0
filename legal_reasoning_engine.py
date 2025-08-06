"""
Advanced Legal Reasoning Engine with Chain-of-Thought Prompting
Implements lawyer-like reasoning patterns for complex legal analysis
"""

import os
import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalPrecedent:
    """Case law precedent with metadata"""
    case_name: str
    citation: str
    court: str
    year: int
    bench_strength: int
    ratio_decidendi: str
    relevant_sections: List[str]
    facts_summary: str
    judgment_link: str = ""
    
@dataclass
class LegalReasoning:
    """Structured legal reasoning output"""
    applicable_laws: List[str]
    relevant_sections: List[str]
    case_precedents: List[LegalPrecedent]
    legal_analysis: str
    reasoning_chain: List[str]
    conclusion: str
    confidence_score: float
    source_links: List[str]

class ChainOfThoughtLegalReasoner:
    """Advanced legal reasoning with step-by-step analysis"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        
        if gemini_api_key and gemini_api_key != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ":
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Load case law database
        self.case_law_db = self._load_case_law_database()
        
        # Legal reasoning templates
        self.reasoning_templates = {
            "contract_law": {
                "steps": [
                    "1. Identify the nature of the agreement",
                    "2. Check essential elements under Section 10, Indian Contract Act",
                    "3. Examine validity conditions (capacity, consent, consideration, object)",
                    "4. Apply relevant case law precedents",
                    "5. Determine legal consequences and remedies"
                ],
                "key_sections": ["10", "11", "13", "14", "15", "16", "17", "18", "23"],
                "landmark_cases": ["Mohori Bibee vs Dharmodas Ghose", "Balfour vs Balfour"]
            },
            "criminal_law": {
                "steps": [
                    "1. Classify the offence (cognizable/non-cognizable, bailable/non-bailable)",
                    "2. Identify applicable IPC sections and punishment",
                    "3. Examine procedural requirements under CrPC",
                    "4. Consider constitutional safeguards and precedents",
                    "5. Determine investigation and trial procedures"
                ],
                "key_sections": ["154", "156", "161", "173", "302", "420", "436", "437", "438"],
                "landmark_cases": ["Maneka Gandhi vs Union of India", "D.K. Basu vs State of West Bengal"]
            },
            "company_law": {
                "steps": [
                    "1. Identify company type and applicable provisions",
                    "2. Check compliance requirements under Companies Act 2013",
                    "3. Examine director duties and liabilities",
                    "4. Apply relevant NCLT/NCLAT precedents",
                    "5. Determine penalties and remedial measures"
                ],
                "key_sections": ["92", "137", "164", "166", "173", "248"],
                "landmark_cases": ["Vodafone International vs Union of India", "Tata Consultancy Services vs State of Andhra Pradesh"]
            },
            "constitutional_law": {
                "steps": [
                    "1. Identify fundamental rights or constitutional provisions involved",
                    "2. Apply constitutional interpretation principles",
                    "3. Examine Supreme Court precedents and constitutional bench decisions",
                    "4. Consider doctrine of basic structure",
                    "5. Determine constitutional validity and remedies"
                ],
                "key_sections": ["14", "19", "21", "32", "226"],
                "landmark_cases": ["Kesavananda Bharati vs State of Kerala", "Maneka Gandhi vs Union of India"]
            }
        }
    
    def _load_case_law_database(self) -> List[LegalPrecedent]:
        """Load case law database with precedents"""
        # Sample case law database - in production, this would be loaded from a comprehensive database
        return [
            LegalPrecedent(
                case_name="Mohori Bibee vs Dharmodas Ghose",
                citation="(1903) ILR 30 Cal 539",
                court="Privy Council",
                year=1903,
                bench_strength=5,
                ratio_decidendi="A minor's agreement is void ab initio and cannot be ratified upon attaining majority",
                relevant_sections=["11"],
                facts_summary="Minor mortgaged property, later sought to avoid the contract",
                judgment_link="https://indiankanoon.org/doc/1396751/"
            ),
            LegalPrecedent(
                case_name="Maneka Gandhi vs Union of India",
                citation="(1978) 1 SCC 248",
                court="Supreme Court",
                year=1978,
                bench_strength=7,
                ratio_decidendi="Article 21 includes right to travel abroad; procedure must be fair, just and reasonable",
                relevant_sections=["21"],
                facts_summary="Passport impounded without hearing, challenged as violation of Article 21",
                judgment_link="https://indiankanoon.org/doc/1766147/"
            ),
            LegalPrecedent(
                case_name="D.K. Basu vs State of West Bengal",
                citation="(1997) 1 SCC 416",
                court="Supreme Court",
                year=1997,
                bench_strength=2,
                ratio_decidendi="Guidelines for arrest and detention to prevent custodial violence",
                relevant_sections=["21", "22"],
                facts_summary="PIL regarding custodial deaths and torture in police custody",
                judgment_link="https://indiankanoon.org/doc/1235094/"
            ),
            LegalPrecedent(
                case_name="Kesavananda Bharati vs State of Kerala",
                citation="(1973) 4 SCC 225",
                court="Supreme Court",
                year=1973,
                bench_strength=13,
                ratio_decidendi="Parliament cannot alter the basic structure of the Constitution",
                relevant_sections=["368"],
                facts_summary="Challenge to constitutional amendments affecting fundamental rights",
                judgment_link="https://indiankanoon.org/doc/257876/"
            ),
            LegalPrecedent(
                case_name="State of Punjab vs Ajaib Singh",
                citation="(1953) SCR 254",
                court="Supreme Court",
                year=1953,
                bench_strength=3,
                ratio_decidendi="Anticipatory bail can be granted to prevent arrest in non-bailable offences",
                relevant_sections=["438"],
                facts_summary="Application for anticipatory bail in criminal case",
                judgment_link="https://indiankanoon.org/doc/1727139/"
            )
        ]
    
    async def analyze_legal_query(self, query: str, topic: str, relevant_docs: List[Dict], 
                                context: str = "") -> LegalReasoning:
        """Perform comprehensive legal analysis with chain-of-thought reasoning"""
        
        try:
            # Step 1: Extract legal facts and issues
            legal_facts = self._extract_legal_facts(query, context)
            
            # Step 2: Identify applicable laws and sections
            applicable_laws, relevant_sections = self._identify_applicable_laws(query, topic, relevant_docs)
            
            # Step 3: Find relevant case precedents
            case_precedents = self._find_relevant_precedents(query, topic, relevant_sections)
            
            # Step 4: Generate chain-of-thought reasoning
            reasoning_chain = await self._generate_reasoning_chain(query, topic, legal_facts, 
                                                                 applicable_laws, case_precedents, relevant_docs)
            
            # Step 5: Synthesize legal analysis
            legal_analysis = await self._synthesize_legal_analysis(query, topic, reasoning_chain, 
                                                                  case_precedents, relevant_docs)
            
            # Step 6: Generate conclusion with confidence
            conclusion, confidence = await self._generate_conclusion(query, legal_analysis, case_precedents)
            
            # Step 7: Compile source links
            source_links = self._compile_source_links(applicable_laws, case_precedents, relevant_sections)
            
            return LegalReasoning(
                applicable_laws=applicable_laws,
                relevant_sections=relevant_sections,
                case_precedents=case_precedents,
                legal_analysis=legal_analysis,
                reasoning_chain=reasoning_chain,
                conclusion=conclusion,
                confidence_score=confidence,
                source_links=source_links
            )
            
        except Exception as e:
            logger.error(f"Legal reasoning error: {e}")
            return self._fallback_reasoning(query, topic, relevant_docs)
    
    def _extract_legal_facts(self, query: str, context: str) -> Dict[str, Any]:
        """Extract key legal facts from query"""
        combined_text = f"{context} {query}".lower()
        
        facts = {
            "parties": [],
            "legal_issues": [],
            "time_elements": [],
            "monetary_elements": [],
            "procedural_elements": []
        }
        
        # Extract parties (basic pattern matching)
        party_patterns = [r"company", r"director", r"accused", r"plaintiff", r"defendant", 
                         r"petitioner", r"respondent", r"complainant"]
        for pattern in party_patterns:
            if re.search(pattern, combined_text):
                facts["parties"].append(pattern)
        
        # Extract time elements
        time_patterns = [r"\d+\s*years?", r"\d+\s*months?", r"\d+\s*days?"]
        for pattern in time_patterns:
            matches = re.findall(pattern, combined_text)
            facts["time_elements"].extend(matches)
        
        # Extract monetary elements
        money_patterns = [r"₹\s*\d+", r"rupees?\s*\d+", r"\d+\s*lakh", r"\d+\s*crore"]
        for pattern in money_patterns:
            matches = re.findall(pattern, combined_text)
            facts["monetary_elements"].extend(matches)
        
        return facts
    
    def _identify_applicable_laws(self, query: str, topic: str, relevant_docs: List[Dict]) -> Tuple[List[str], List[str]]:
        """Identify applicable laws and sections"""
        applicable_laws = []
        relevant_sections = []
        
        # Topic-based law identification
        law_mapping = {
            "contract_law": ["Indian Contract Act, 1872", "Specific Relief Act, 1963"],
            "criminal_law": ["Indian Penal Code, 1860", "Code of Criminal Procedure, 1973"],
            "company_law": ["Companies Act, 2013", "SEBI Act, 1992"],
            "constitutional_law": ["Constitution of India, 1950"],
            "property_law": ["Transfer of Property Act, 1882", "Registration Act, 1908"]
        }
        
        applicable_laws = law_mapping.get(topic, ["General Legal Provisions"])
        
        # Extract sections from documents
        for doc in relevant_docs:
            doc_sections = doc.get("sections", [])
            if doc_sections:
                relevant_sections.extend([str(s) for s in doc_sections])
        
        # Extract sections from query
        section_matches = re.findall(r'section\s*(\d+)', query.lower())
        relevant_sections.extend(section_matches)
        
        return applicable_laws, list(set(relevant_sections))
    
    def _find_relevant_precedents(self, query: str, topic: str, sections: List[str]) -> List[LegalPrecedent]:
        """Find relevant case law precedents"""
        relevant_cases = []
        query_lower = query.lower()
        
        for case in self.case_law_db:
            relevance_score = 0
            
            # Check section overlap
            case_sections = [str(s) for s in case.relevant_sections]
            section_overlap = len(set(sections).intersection(set(case_sections)))
            relevance_score += section_overlap * 2
            
            # Check keyword relevance
            case_text = f"{case.case_name} {case.ratio_decidendi} {case.facts_summary}".lower()
            query_words = query_lower.split()
            keyword_matches = sum(1 for word in query_words if len(word) > 3 and word in case_text)
            relevance_score += keyword_matches
            
            # Check topic relevance
            if topic == "criminal_law" and any(term in case_text for term in ["criminal", "bail", "arrest", "ipc", "crpc"]):
                relevance_score += 3
            elif topic == "contract_law" and any(term in case_text for term in ["contract", "agreement", "consideration"]):
                relevance_score += 3
            elif topic == "constitutional_law" and any(term in case_text for term in ["constitutional", "fundamental", "article"]):
                relevance_score += 3
            
            if relevance_score > 2:
                relevant_cases.append(case)
        
        # Sort by relevance and return top 3
        return sorted(relevant_cases, key=lambda x: len(set(sections).intersection(set(x.relevant_sections))), reverse=True)[:3]
    
    async def _generate_reasoning_chain(self, query: str, topic: str, facts: Dict, 
                                      laws: List[str], precedents: List[LegalPrecedent], 
                                      docs: List[Dict]) -> List[str]:
        """Generate step-by-step legal reasoning chain"""
        
        template = self.reasoning_templates.get(topic, {})
        base_steps = template.get("steps", [
            "1. Identify the legal issue",
            "2. Determine applicable laws",
            "3. Apply legal principles",
            "4. Consider precedents",
            "5. Reach conclusion"
        ])
        
        if not self.gemini_model:
            return base_steps
        
        try:
            prompt = f"""As a legal expert, provide step-by-step reasoning for this legal query:

QUERY: {query}

APPLICABLE LAWS: {', '.join(laws)}
RELEVANT PRECEDENTS: {[case.case_name for case in precedents]}

Provide 5-7 detailed reasoning steps that a lawyer would follow:
1. [First step of legal analysis]
2. [Second step of legal analysis]
...

Focus on logical legal reasoning, not just facts."""

            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            if response and hasattr(response, 'text') and response.text:
                # Extract numbered steps
                steps = []
                lines = response.text.split('\n')
                for line in lines:
                    if re.match(r'^\d+\.', line.strip()):
                        steps.append(line.strip())
                
                return steps if steps else base_steps
            
        except Exception as e:
            logger.error(f"Reasoning chain generation error: {e}")
        
        return base_steps
    
    async def _synthesize_legal_analysis(self, query: str, topic: str, reasoning_chain: List[str], 
                                       precedents: List[LegalPrecedent], docs: List[Dict]) -> str:
        """Synthesize comprehensive legal analysis"""
        
        if not self.gemini_model:
            return self._template_legal_analysis(query, topic, precedents, docs)
        
        try:
            precedent_text = "\n".join([
                f"• {case.case_name} ({case.year}): {case.ratio_decidendi}"
                for case in precedents
            ])
            
            doc_text = "\n".join([
                f"• {doc.get('question', '')}: {doc.get('answer', '')[:200]}..."
                for doc in docs[:2]
            ])
            
            prompt = f"""Provide a comprehensive legal analysis for this query using the reasoning chain and precedents:

QUERY: {query}
TOPIC: {topic.replace('_', ' ').title()}

REASONING CHAIN:
{chr(10).join(reasoning_chain)}

RELEVANT PRECEDENTS:
{precedent_text}

LEGAL PROVISIONS:
{doc_text}

Provide a detailed legal analysis that:
1. Applies the reasoning chain systematically
2. Cites relevant precedents with their ratios
3. Explains legal principles clearly
4. Shows how precedents apply to the current query
5. Maintains professional legal language

Format as a structured analysis with clear headings."""

            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            if response and hasattr(response, 'text') and response.text:
                return response.text
            
        except Exception as e:
            logger.error(f"Legal analysis synthesis error: {e}")
        
        return self._template_legal_analysis(query, topic, precedents, docs)
    
    def _template_legal_analysis(self, query: str, topic: str, precedents: List[LegalPrecedent], 
                                docs: List[Dict]) -> str:
        """Template-based legal analysis fallback"""
        
        analysis_parts = [
            f"**Legal Analysis - {topic.replace('_', ' ').title()}**\n",
            "**Issue Identification:**",
            f"The query pertains to {topic.replace('_', ' ')} and requires analysis of applicable legal provisions.\n",
            "**Applicable Legal Framework:**"
        ]
        
        if docs:
            primary_doc = docs[0]
            analysis_parts.append(f"Primary legal provision: {primary_doc.get('answer', '')[:300]}...\n")
        
        if precedents:
            analysis_parts.append("**Relevant Precedents:**")
            for case in precedents:
                analysis_parts.append(f"• **{case.case_name}** ({case.year}): {case.ratio_decidendi}")
            analysis_parts.append("")
        
        analysis_parts.extend([
            "**Legal Application:**",
            "Based on the applicable legal framework and precedential guidance, the analysis proceeds through systematic application of legal principles to the facts presented.",
            "",
            "**Conclusion:**",
            "The legal position is determined by the interplay of statutory provisions and judicial precedents as outlined above."
        ])
        
        return "\n".join(analysis_parts)
    
    async def _generate_conclusion(self, query: str, analysis: str, precedents: List[LegalPrecedent]) -> Tuple[str, float]:
        """Generate legal conclusion with confidence score"""
        
        if not self.gemini_model:
            return self._template_conclusion(query, precedents)
        
        try:
            prompt = f"""Based on this legal analysis, provide a clear, actionable conclusion:

QUERY: {query}

LEGAL ANALYSIS:
{analysis[:1000]}...

Provide:
1. A clear, direct answer to the legal query
2. Practical next steps or recommendations
3. Any important caveats or limitations

Keep the conclusion concise but comprehensive."""

            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            if response and hasattr(response, 'text') and response.text:
                # Calculate confidence based on precedent strength and analysis depth
                confidence = 0.7
                if precedents:
                    confidence += 0.1 * len(precedents)
                if len(analysis) > 500:
                    confidence += 0.1
                
                return response.text, min(confidence, 0.95)
            
        except Exception as e:
            logger.error(f"Conclusion generation error: {e}")
        
        return self._template_conclusion(query, precedents)
    
    def _template_conclusion(self, query: str, precedents: List[LegalPrecedent]) -> Tuple[str, float]:
        """Template-based conclusion fallback"""
        conclusion = f"""**Legal Conclusion:**

Based on the applicable legal provisions and precedential guidance, the query requires careful consideration of the specific facts and circumstances.

**Recommendations:**
1. Consult the relevant statutory provisions
2. Consider the precedential guidance from established case law
3. Seek professional legal advice for specific application

**Important Note:**
This analysis is based on general legal principles. Specific legal advice should be sought for particular circumstances."""

        confidence = 0.6 + (0.1 * len(precedents))
        return conclusion, min(confidence, 0.8)
    
    def _compile_source_links(self, laws: List[str], precedents: List[LegalPrecedent], 
                            sections: List[str]) -> List[str]:
        """Compile clickable source links"""
        links = []
        
        # Add bare act links
        act_links = {
            "Indian Contract Act, 1872": "https://indiacode.nic.in/handle/123456789/2187",
            "Indian Penal Code, 1860": "https://indiacode.nic.in/handle/123456789/2263",
            "Code of Criminal Procedure, 1973": "https://indiacode.nic.in/handle/123456789/2263",
            "Companies Act, 2013": "https://indiacode.nic.in/handle/123456789/2116",
            "Constitution of India, 1950": "https://indiacode.nic.in/handle/123456789/2263"
        }
        
        for law in laws:
            if law in act_links:
                links.append(f"📜 {law} - [View Bare Act]({act_links[law]})")
        
        # Add case law links
        for case in precedents:
            if case.judgment_link:
                links.append(f"⚖️ {case.case_name} - [Read Judgment]({case.judgment_link})")
        
        # Add section-specific links
        for section in sections[:3]:  # Limit to top 3 sections
            links.append(f"📋 Section {section} - [View Details](https://indiacode.nic.in)")
        
        return links
    
    def _fallback_reasoning(self, query: str, topic: str, docs: List[Dict]) -> LegalReasoning:
        """Fallback reasoning when AI analysis fails"""
        return LegalReasoning(
            applicable_laws=[f"{topic.replace('_', ' ').title()} Provisions"],
            relevant_sections=[],
            case_precedents=[],
            legal_analysis=f"Basic legal analysis for {topic.replace('_', ' ')} query.",
            reasoning_chain=[
                "1. Identify the legal issue",
                "2. Apply relevant legal provisions",
                "3. Consider practical implications",
                "4. Provide guidance based on established law"
            ],
            conclusion="Legal guidance provided based on available information. Consult a qualified advocate for specific advice.",
            confidence_score=0.5,
            source_links=[]
        )

# Export the main class
__all__ = ['ChainOfThoughtLegalReasoner', 'LegalReasoning', 'LegalPrecedent']