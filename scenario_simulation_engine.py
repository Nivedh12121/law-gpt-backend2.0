"""
Legal Scenario Simulation Engine
Helps users test hypothetical cases and understand legal processes step-by-step
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimulationStep:
    """Single step in legal process simulation"""
    step_number: int
    title: str
    description: str
    timeline: str
    required_documents: List[str]
    estimated_cost: str
    legal_provisions: List[str]
    potential_outcomes: List[str]
    next_steps: List[str]

@dataclass
class LegalScenario:
    """Complete legal scenario with simulation"""
    scenario_id: str
    title: str
    description: str
    legal_area: str
    complexity_level: str  # "simple", "moderate", "complex"
    estimated_duration: str
    total_estimated_cost: str
    steps: List[SimulationStep]
    key_considerations: List[str]
    alternative_paths: List[str]
    success_probability: float

class ScenarioSimulationEngine:
    """Engine for simulating legal scenarios and processes"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        
        if gemini_api_key and gemini_api_key != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ":
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Pre-defined scenario templates
        self.scenario_templates = self._load_scenario_templates()
        
        # Document templates
        self.document_templates = self._load_document_templates()
    
    def _load_scenario_templates(self) -> Dict[str, Dict]:
        """Load pre-defined legal scenario templates"""
        return {
            "cheque_bounce": {
                "title": "Cheque Bounce Case (Section 138 NI Act)",
                "legal_area": "criminal_law",
                "complexity": "moderate",
                "duration": "6-12 months",
                "cost_range": "₹15,000 - ₹50,000",
                "steps": [
                    {
                        "title": "Cheque Presentation and Dishonor",
                        "timeline": "Day 1",
                        "description": "Present cheque to bank, receive dishonor memo",
                        "documents": ["Original cheque", "Bank dishonor memo"],
                        "provisions": ["Section 138 NI Act"]
                    },
                    {
                        "title": "Legal Notice",
                        "timeline": "Within 30 days of dishonor",
                        "description": "Send legal notice to drawer demanding payment",
                        "documents": ["Legal notice", "Postal receipt"],
                        "provisions": ["Section 138 NI Act"]
                    },
                    {
                        "title": "Filing Criminal Complaint",
                        "timeline": "After 15 days of notice, within 30 days",
                        "description": "File complaint in Magistrate court",
                        "documents": ["Complaint application", "Supporting documents"],
                        "provisions": ["Section 138, 142 NI Act"]
                    }
                ]
            },
            "company_incorporation": {
                "title": "Private Limited Company Incorporation",
                "legal_area": "company_law",
                "complexity": "moderate",
                "duration": "15-30 days",
                "cost_range": "₹10,000 - ₹25,000",
                "steps": [
                    {
                        "title": "Name Reservation",
                        "timeline": "1-2 days",
                        "description": "Apply for company name approval with ROC",
                        "documents": ["Form INC-1", "Fee payment"],
                        "provisions": ["Section 4 Companies Act 2013"]
                    },
                    {
                        "title": "Document Preparation",
                        "timeline": "3-5 days",
                        "description": "Prepare MOA, AOA, and other incorporation documents",
                        "documents": ["MOA", "AOA", "Form INC-2"],
                        "provisions": ["Section 7 Companies Act 2013"]
                    },
                    {
                        "title": "Filing with ROC",
                        "timeline": "7-15 days",
                        "description": "Submit incorporation application to ROC",
                        "documents": ["Complete incorporation kit"],
                        "provisions": ["Section 7, 8 Companies Act 2013"]
                    }
                ]
            },
            "consumer_complaint": {
                "title": "Consumer Complaint Filing",
                "legal_area": "consumer_law",
                "complexity": "simple",
                "duration": "3-6 months",
                "cost_range": "₹500 - ₹5,000",
                "steps": [
                    {
                        "title": "Gather Evidence",
                        "timeline": "1-7 days",
                        "description": "Collect bills, receipts, correspondence",
                        "documents": ["Purchase receipts", "Warranty cards", "Correspondence"],
                        "provisions": ["Consumer Protection Act 2019"]
                    },
                    {
                        "title": "File Complaint",
                        "timeline": "1 day",
                        "description": "Submit complaint to appropriate consumer forum",
                        "documents": ["Complaint form", "Supporting documents"],
                        "provisions": ["Section 35 Consumer Protection Act"]
                    }
                ]
            },
            "fir_filing": {
                "title": "FIR Filing Process",
                "legal_area": "criminal_law",
                "complexity": "simple",
                "duration": "1-2 days",
                "cost_range": "Free",
                "steps": [
                    {
                        "title": "Visit Police Station",
                        "timeline": "Immediately",
                        "description": "Go to nearest police station with jurisdiction",
                        "documents": ["Identity proof", "Evidence if any"],
                        "provisions": ["Section 154 CrPC"]
                    },
                    {
                        "title": "Lodge Complaint",
                        "timeline": "Same day",
                        "description": "Provide detailed information to police officer",
                        "documents": ["Written complaint", "Supporting evidence"],
                        "provisions": ["Section 154 CrPC"]
                    }
                ]
            },
            "bail_application": {
                "title": "Bail Application Process",
                "legal_area": "criminal_law",
                "complexity": "moderate",
                "duration": "1-7 days",
                "cost_range": "₹5,000 - ₹25,000",
                "steps": [
                    {
                        "title": "Assess Bail Eligibility",
                        "timeline": "Day 1",
                        "description": "Determine if offence is bailable or non-bailable",
                        "documents": ["Case details", "Charge sheet"],
                        "provisions": ["Section 436, 437 CrPC"]
                    },
                    {
                        "title": "Prepare Bail Application",
                        "timeline": "1-2 days",
                        "description": "Draft bail application with grounds",
                        "documents": ["Bail application", "Surety documents"],
                        "provisions": ["Section 437, 439 CrPC"]
                    }
                ]
            }
        }
    
    def _load_document_templates(self) -> Dict[str, str]:
        """Load document templates for common legal processes"""
        return {
            "legal_notice_cheque_bounce": """
LEGAL NOTICE UNDER SECTION 138 OF NEGOTIABLE INSTRUMENTS ACT, 1881

To,
[Drawer Name]
[Address]

Subject: Legal Notice for dishonor of Cheque No. [Cheque Number] dated [Date]

Dear Sir/Madam,

My client [Payee Name] had received from you a cheque bearing No. [Cheque Number] dated [Date] for Rs. [Amount] drawn on [Bank Name] towards [Purpose].

The said cheque was presented for collection on [Presentation Date] but was dishonored by your bank with the remarks "[Dishonor Reason]".

You are hereby called upon to pay the said amount of Rs. [Amount] within 15 days from the receipt of this notice, failing which my client will be constrained to initiate criminal proceedings against you under Section 138 of the Negotiable Instruments Act, 1881.

Yours faithfully,
[Advocate Name]
[Date]
""",
            "consumer_complaint": """
CONSUMER COMPLAINT UNDER CONSUMER PROTECTION ACT, 2019

Before the [District/State/National] Consumer Disputes Redressal Commission

Complainant: [Name]
Address: [Address]

Vs.

Opposite Party: [Company/Service Provider Name]
Address: [Address]

COMPLAINT UNDER SECTION 35 OF CONSUMER PROTECTION ACT, 2019

Respectfully submitted:

1. That the complainant is a consumer who purchased [Product/Service] from the opposite party on [Date] for Rs. [Amount].

2. That the opposite party has committed deficiency in service/defect in goods by [Details of deficiency].

3. That despite several requests, the opposite party has failed to redress the grievance.

PRAYER:
The complainant prays for:
a) Compensation of Rs. [Amount]
b) Replacement/Refund
c) Costs of litigation

[Complainant Signature]
[Date]
""",
            "fir_complaint": """
FIRST INFORMATION REPORT (FIR)

To,
The Station House Officer
[Police Station Name]

Subject: Complaint regarding [Nature of Offence]

Sir,

I, [Complainant Name], son/daughter of [Father's Name], aged [Age] years, residing at [Address], hereby lodge this complaint regarding the following incident:

Date of Incident: [Date]
Time of Incident: [Time]
Place of Incident: [Location]

Details of the Incident:
[Detailed description of the incident]

Accused Person(s):
[Name and details if known]

I request you to kindly register an FIR and take necessary action as per law.

Yours faithfully,
[Complainant Name]
[Date]
"""
        }
    
    async def simulate_scenario(self, scenario_type: str, user_facts: Dict[str, Any]) -> LegalScenario:
        """Simulate a legal scenario based on user facts"""
        
        try:
            # Get base template
            template = self.scenario_templates.get(scenario_type)
            if not template:
                return await self._generate_custom_scenario(scenario_type, user_facts)
            
            # Generate detailed simulation
            scenario = await self._build_detailed_scenario(template, user_facts, scenario_type)
            
            return scenario
            
        except Exception as e:
            logger.error(f"Scenario simulation error: {e}")
            return self._fallback_scenario(scenario_type)
    
    async def _build_detailed_scenario(self, template: Dict, user_facts: Dict, scenario_type: str) -> LegalScenario:
        """Build detailed scenario from template and user facts"""
        
        steps = []
        for i, step_template in enumerate(template["steps"], 1):
            # Enhance step with AI if available
            if self.gemini_model:
                enhanced_step = await self._enhance_step_with_ai(step_template, user_facts, scenario_type)
            else:
                enhanced_step = self._enhance_step_template(step_template, user_facts)
            
            step = SimulationStep(
                step_number=i,
                title=enhanced_step["title"],
                description=enhanced_step["description"],
                timeline=enhanced_step["timeline"],
                required_documents=enhanced_step.get("documents", []),
                estimated_cost=enhanced_step.get("cost", "Varies"),
                legal_provisions=enhanced_step.get("provisions", []),
                potential_outcomes=enhanced_step.get("outcomes", ["Success", "Requires additional action"]),
                next_steps=enhanced_step.get("next_steps", ["Proceed to next step"])
            )
            steps.append(step)
        
        # Calculate success probability
        success_probability = self._calculate_success_probability(user_facts, scenario_type)
        
        # Generate key considerations
        key_considerations = await self._generate_key_considerations(user_facts, scenario_type)
        
        return LegalScenario(
            scenario_id=f"{scenario_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=template["title"],
            description=f"Simulation for {template['title']} based on provided facts",
            legal_area=template["legal_area"],
            complexity_level=template["complexity"],
            estimated_duration=template["duration"],
            total_estimated_cost=template["cost_range"],
            steps=steps,
            key_considerations=key_considerations,
            alternative_paths=await self._generate_alternative_paths(scenario_type, user_facts),
            success_probability=success_probability
        )
    
    async def _enhance_step_with_ai(self, step_template: Dict, user_facts: Dict, scenario_type: str) -> Dict:
        """Enhance step template with AI-generated details"""
        
        try:
            prompt = f"""Enhance this legal process step with specific details based on the user's situation:

SCENARIO TYPE: {scenario_type}
USER FACTS: {json.dumps(user_facts, indent=2)}

STEP TEMPLATE:
Title: {step_template['title']}
Description: {step_template['description']}
Timeline: {step_template['timeline']}

Provide enhanced details including:
1. Specific actions to take
2. Potential challenges
3. Estimated costs
4. Required documents
5. Possible outcomes
6. Next steps

Format as JSON with keys: title, description, timeline, documents, cost, provisions, outcomes, next_steps"""

            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            if response and hasattr(response, 'text') and response.text:
                try:
                    # Try to parse JSON response
                    enhanced_data = json.loads(response.text)
                    return enhanced_data
                except json.JSONDecodeError:
                    # Fallback to template enhancement
                    return self._enhance_step_template(step_template, user_facts)
            
        except Exception as e:
            logger.error(f"AI step enhancement error: {e}")
        
        return self._enhance_step_template(step_template, user_facts)
    
    def _enhance_step_template(self, step_template: Dict, user_facts: Dict) -> Dict:
        """Enhance step template with basic customization"""
        enhanced = step_template.copy()
        
        # Add estimated costs based on scenario type
        cost_estimates = {
            "legal_notice": "₹2,000 - ₹5,000",
            "court_filing": "₹1,000 - ₹10,000",
            "documentation": "₹500 - ₹2,000",
            "registration": "₹1,000 - ₹5,000"
        }
        
        step_title_lower = step_template["title"].lower()
        for key, cost in cost_estimates.items():
            if key in step_title_lower:
                enhanced["cost"] = cost
                break
        else:
            enhanced["cost"] = "Varies"
        
        # Add potential outcomes
        enhanced["outcomes"] = [
            "Successful completion",
            "Requires additional documentation",
            "May face delays",
            "Possible legal challenges"
        ]
        
        # Add next steps
        enhanced["next_steps"] = [
            "Monitor progress",
            "Prepare for next phase",
            "Consult legal advisor if needed"
        ]
        
        return enhanced
    
    def _calculate_success_probability(self, user_facts: Dict, scenario_type: str) -> float:
        """Calculate success probability based on facts and scenario type"""
        base_probabilities = {
            "cheque_bounce": 0.75,
            "company_incorporation": 0.95,
            "consumer_complaint": 0.70,
            "fir_filing": 0.90,
            "bail_application": 0.65
        }
        
        base_prob = base_probabilities.get(scenario_type, 0.70)
        
        # Adjust based on user facts
        adjustments = 0.0
        
        # Check for strong evidence
        if user_facts.get("has_documentation", False):
            adjustments += 0.10
        
        # Check for legal representation
        if user_facts.get("has_lawyer", False):
            adjustments += 0.15
        
        # Check for complexity
        if user_facts.get("complexity", "moderate") == "simple":
            adjustments += 0.05
        elif user_facts.get("complexity", "moderate") == "complex":
            adjustments -= 0.10
        
        return min(max(base_prob + adjustments, 0.1), 0.95)
    
    async def _generate_key_considerations(self, user_facts: Dict, scenario_type: str) -> List[str]:
        """Generate key considerations for the scenario"""
        
        if self.gemini_model:
            try:
                prompt = f"""Generate 5-7 key considerations for this legal scenario:

SCENARIO: {scenario_type}
USER FACTS: {json.dumps(user_facts, indent=2)}

Provide practical considerations including:
- Legal requirements
- Potential risks
- Cost factors
- Timeline considerations
- Success factors

Format as a simple list."""

                response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
                
                if response and hasattr(response, 'text') and response.text:
                    # Extract list items
                    considerations = []
                    lines = response.text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                            # Clean up the line
                            clean_line = re.sub(r'^[-•\d\.\)]\s*', '', line)
                            if clean_line:
                                considerations.append(clean_line)
                    
                    if considerations:
                        return considerations
            except Exception as e:
                logger.error(f"Key considerations generation error: {e}")
        
        # Fallback considerations
        return [
            "Ensure all required documents are properly prepared",
            "Consider engaging a qualified legal professional",
            "Be aware of statutory timelines and deadlines",
            "Maintain proper records throughout the process",
            "Understand potential costs and fee structures",
            "Consider alternative dispute resolution methods",
            "Stay updated on relevant legal developments"
        ]
    
    async def _generate_alternative_paths(self, scenario_type: str, user_facts: Dict) -> List[str]:
        """Generate alternative legal paths"""
        
        alternatives = {
            "cheque_bounce": [
                "Negotiate settlement before filing case",
                "Send additional legal notices",
                "Consider civil recovery suit",
                "Explore mediation options"
            ],
            "consumer_complaint": [
                "Direct negotiation with company",
                "Online consumer portal complaint",
                "Approach consumer helpline",
                "Social media escalation"
            ],
            "company_incorporation": [
                "Consider LLP formation instead",
                "Explore partnership options",
                "One Person Company (OPC) option",
                "Proprietorship as interim solution"
            ]
        }
        
        return alternatives.get(scenario_type, [
            "Explore out-of-court settlement",
            "Consider alternative legal remedies",
            "Seek mediation or arbitration",
            "Consult multiple legal opinions"
        ])
    
    async def _generate_custom_scenario(self, scenario_type: str, user_facts: Dict) -> LegalScenario:
        """Generate custom scenario for unknown types"""
        
        if self.gemini_model:
            try:
                prompt = f"""Create a legal process simulation for this scenario:

SCENARIO TYPE: {scenario_type}
USER FACTS: {json.dumps(user_facts, indent=2)}

Provide a step-by-step legal process including:
1. 5-7 main steps
2. Timeline for each step
3. Required documents
4. Estimated costs
5. Legal provisions involved
6. Potential outcomes

Format as a structured response."""

                response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
                
                if response and hasattr(response, 'text') and response.text:
                    # Parse the response and create scenario
                    return self._parse_ai_scenario_response(response.text, scenario_type, user_facts)
            
            except Exception as e:
                logger.error(f"Custom scenario generation error: {e}")
        
        return self._fallback_scenario(scenario_type)
    
    def _parse_ai_scenario_response(self, ai_response: str, scenario_type: str, user_facts: Dict) -> LegalScenario:
        """Parse AI response into structured scenario"""
        
        # Basic parsing - in production, this would be more sophisticated
        steps = []
        lines = ai_response.split('\n')
        
        step_count = 1
        for line in lines:
            if 'step' in line.lower() and ':' in line:
                step = SimulationStep(
                    step_number=step_count,
                    title=line.split(':')[1].strip() if ':' in line else f"Step {step_count}",
                    description=f"Process step for {scenario_type}",
                    timeline="Varies",
                    required_documents=["As applicable"],
                    estimated_cost="Consult legal advisor",
                    legal_provisions=["Applicable legal provisions"],
                    potential_outcomes=["Success", "Requires additional action"],
                    next_steps=["Proceed as advised"]
                )
                steps.append(step)
                step_count += 1
        
        if not steps:
            # Create default steps
            steps = [
                SimulationStep(
                    step_number=1,
                    title="Initial Assessment",
                    description="Assess the legal situation and requirements",
                    timeline="1-2 days",
                    required_documents=["Relevant documents"],
                    estimated_cost="₹1,000 - ₹5,000",
                    legal_provisions=["Applicable laws"],
                    potential_outcomes=["Clear understanding of legal position"],
                    next_steps=["Proceed with legal action"]
                )
            ]
        
        return LegalScenario(
            scenario_id=f"custom_{scenario_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Legal Process: {scenario_type.replace('_', ' ').title()}",
            description=f"Custom simulation for {scenario_type}",
            legal_area="general_law",
            complexity_level="moderate",
            estimated_duration="Varies",
            total_estimated_cost="Consult legal advisor",
            steps=steps,
            key_considerations=["Consult qualified legal professional"],
            alternative_paths=["Explore alternative legal remedies"],
            success_probability=0.70
        )
    
    def _fallback_scenario(self, scenario_type: str) -> LegalScenario:
        """Fallback scenario when generation fails"""
        
        return LegalScenario(
            scenario_id=f"fallback_{scenario_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"Legal Process Guidance: {scenario_type.replace('_', ' ').title()}",
            description="Basic legal process guidance",
            legal_area="general_law",
            complexity_level="moderate",
            estimated_duration="Varies",
            total_estimated_cost="Consult legal advisor",
            steps=[
                SimulationStep(
                    step_number=1,
                    title="Consult Legal Professional",
                    description="Seek advice from qualified legal professional",
                    timeline="1-2 days",
                    required_documents=["All relevant documents"],
                    estimated_cost="₹2,000 - ₹10,000",
                    legal_provisions=["Applicable legal provisions"],
                    potential_outcomes=["Professional legal guidance"],
                    next_steps=["Follow professional advice"]
                )
            ],
            key_considerations=["Professional legal advice recommended"],
            alternative_paths=["Explore multiple legal opinions"],
            success_probability=0.70
        )
    
    def generate_document_draft(self, document_type: str, user_data: Dict[str, Any]) -> str:
        """Generate document draft based on template and user data"""
        
        template = self.document_templates.get(document_type, "")
        if not template:
            return f"Document template for {document_type} not available. Please consult a legal professional."
        
        # Replace placeholders with user data
        draft = template
        for key, value in user_data.items():
            placeholder = f"[{key.upper().replace('_', ' ')}]"
            draft = draft.replace(placeholder, str(value))
        
        # Add disclaimer
        disclaimer = """
DISCLAIMER: This is a template document for reference only. 
Please consult a qualified legal professional before using this document.
Customize as per your specific requirements and local legal practices.
"""
        
        return draft + disclaimer
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of available scenario simulations"""
        
        scenarios = []
        for scenario_type, template in self.scenario_templates.items():
            scenarios.append({
                "type": scenario_type,
                "title": template["title"],
                "legal_area": template["legal_area"],
                "complexity": template["complexity"],
                "estimated_duration": template["duration"],
                "cost_range": template["cost_range"],
                "description": f"Step-by-step simulation for {template['title']}"
            })
        
        return scenarios

# Export the main classes
__all__ = ['ScenarioSimulationEngine', 'LegalScenario', 'SimulationStep']