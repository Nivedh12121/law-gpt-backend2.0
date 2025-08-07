"""
Law GPT - Enhanced Production Backend
Complete version with company law support
"""

import os
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Law GPT API",
    description="AI Legal Assistant for Indian Law",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all origins for development
        "https://law-gpt-professional.web.app",  # Your Firebase hosting domain
        "https://law-gpt-professional.firebaseapp.com",  # Alternative Firebase domain
        "http://localhost:3000",  # Local development
        "http://localhost:3001",  # Alternative local port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: list = []

# Enhanced Legal knowledge base
LEGAL_RESPONSES = {
    "section 302 ipc": {
        "response": """**Section 302 IPC - Punishment for Murder**

📋 **Legal Provision:**
Section 302 of the Indian Penal Code deals with punishment for murder.

⚖️ **Definition:**
Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.

🔍 **Key Elements:**
• **Intention to cause death** - The act must be done with intention of causing death
• **Knowledge of likelihood** - Or with knowledge that the act is likely to cause death
• **Culpable homicide amounting to murder** - Must satisfy conditions under Section 300

📚 **Related Sections:**
• Section 299 - Culpable homicide
• Section 300 - Murder
• Section 301 - Culpable homicide by causing death of person other than person whose death was intended

⚠️ **Important Note:**
This is a non-bailable and cognizable offense. Always consult with a qualified criminal lawyer for specific cases.""",
        "sources": ["Indian Penal Code", "Supreme Court Judgments", "Criminal Law"]
    },
    
    "article 21": {
        "response": """**Article 21 - Right to Life and Personal Liberty**

📋 **Constitutional Provision:**
"No person shall be deprived of his life or personal liberty except according to procedure established by law."

🔍 **Scope and Interpretation:**
• **Right to Life** - Includes right to live with human dignity
• **Personal Liberty** - Encompasses various freedoms and rights
• **Procedure Established by Law** - Must be just, fair and reasonable

📚 **Landmark Cases:**
• **Maneka Gandhi v. Union of India (1978)** - Expanded interpretation
• **Francis Coralie Mullin v. Administrator (1981)** - Right to live with dignity
• **Olga Tellis v. Bombay Municipal Corporation (1985)** - Right to livelihood

🌟 **Extended Rights Under Article 21:**
• Right to Privacy
• Right to Education
• Right to Health
• Right to Clean Environment
• Right to Speedy Trial

⚖️ **Legal Significance:**
Article 21 is the heart of fundamental rights and has been expansively interpreted by the Supreme Court.""",
        "sources": ["Constitution of India", "Supreme Court Cases", "Fundamental Rights"]
    },
    
    "article 14": {
        "response": """**Article 14 - Right to Equality**

📋 **Constitutional Text:**
"The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India."

🔍 **Two Concepts:**
1. **Equality before Law** - Negative concept (no discrimination)
2. **Equal Protection of Laws** - Positive concept (equal treatment)

📚 **Key Principles:**
• **Reasonable Classification** - State can classify but must be reasonable
• **Non-Arbitrariness** - State action must not be arbitrary
• **Rule of Law** - Everyone is equal before law

⚖️ **Tests for Reasonable Classification:**
1. Classification must be based on intelligible differentia
2. Differentia must have rational relation to object sought to be achieved

🌟 **Landmark Cases:**
• **E.P. Royappa v. State of Tamil Nadu (1974)** - Arbitrariness doctrine
• **Maneka Gandhi v. Union of India (1978)** - Procedure must be fair
• **Indra Sawhney v. Union of India (1992)** - Reservation case

📋 **Exceptions:**
• President and Governors have immunity
• Foreign diplomats have immunity
• Different laws for different states (reasonable classification)""",
        "sources": ["Constitution of India", "Supreme Court Judgments", "Equality Rights"]
    },
    
    "annual returns": {
        "response": """**Annual Returns for Private Limited Companies**

📋 **Legal Requirement:**
Every private limited company must file annual returns with the Registrar of Companies (ROC) under the Companies Act, 2013.

⚖️ **Consequences of Non-Filing for 3 Years:**

🚨 **For the Company:**
• **Section 248** - Company may be struck off from ROC records
• **Penalty under Section 92(5)** - ₹5,000 + ₹50 per day of default
• **Additional penalty** - ₹1,00,000 for continuing default
• **Loss of legal status** - Company ceases to exist legally
• **Asset forfeiture** - All assets vest in the Central Government

👥 **For Directors:**
• **Section 164(2)(a)** - Disqualification for 5 years from appointment as director
• **Personal liability** - For company debts in certain cases
• **Criminal liability** - Under Section 447 (punishment up to 6 months imprisonment)
• **Penalty** - ₹25,000 to ₹5,00,000 per director

🔄 **Revival Options:**
• **Section 252** - Application for revival within 20 years
• **Compounding of offenses** - Pay penalties and file returns
• **Fresh incorporation** - Start new company (if old one struck off)

📚 **Key Sections:**
• Section 92 - Annual Return filing
• Section 248 - Striking off companies
• Section 252 - Revival of companies
• Section 164 - Director disqualification

⚠️ **Immediate Action Required:**
File all pending annual returns immediately and pay applicable penalties to avoid striking off.""",
        "sources": ["Companies Act 2013", "ROC Guidelines", "Corporate Law"]
    },
    
    "bail procedure": {
        "response": """**🚨 BAIL PROCEDURE UNDER CrPC - COMPLETE GUIDE**

📋 **Legal Framework:** Sections 436-450, Code of Criminal Procedure, 1973

## 📝 **TYPES OF BAIL:**

### **1. REGULAR BAIL (Section 437 CrPC)** ⚖️
• **When:** After arrest and during trial
• **Who can grant:** Magistrate, Sessions Judge, High Court, Supreme Court
• **Procedure:**
  - File bail application with supporting documents
  - Serve copy to prosecution
  - Court hearing with arguments
  - Decision based on merits

### **2. ANTICIPATORY BAIL (Section 438 CrPC)** 🛡️
• **When:** Before arrest (apprehension of arrest)
• **Who can grant:** Sessions Judge, High Court, Supreme Court only
• **Conditions:** Must show reasonable grounds for arrest apprehension

### **3. INTERIM BAIL** ⏰
• **When:** Temporary relief pending regular bail decision
• **Duration:** Usually 2-4 weeks
• **Purpose:** Prevent immediate custody

## 🔍 **BAIL APPLICATION PROCEDURE:**

### **STEP 1: PREPARATION** 📋
• **Draft bail application** with proper format
• **Attach documents:**
  - Copy of FIR
  - Medical certificates (if applicable)
  - Character certificates
  - Surety documents
  - Property papers for bond

### **STEP 2: FILING** 📄
• **File in appropriate court** (Magistrate/Sessions/High Court)
• **Pay court fees** as prescribed
• **Serve copy to prosecution** within stipulated time
• **Get hearing date** from court registry

### **STEP 3: HEARING** 🏛️
• **Prosecution arguments** against bail
• **Defense arguments** for bail
• **Court considers factors** for bail decision
• **Order passed** - granted/rejected/conditions imposed

## ⚖️ **FACTORS COURT CONSIDERS:**

### **FOR GRANTING BAIL:** ✅
• **Nature of offense** - Non-serious, bailable
• **Strength of evidence** - Weak prosecution case
• **Flight risk** - Accused has roots in community
• **No tampering** - Won't influence witnesses
• **Health grounds** - Medical emergency
• **Long trial** - Case likely to take years

### **AGAINST GRANTING BAIL:** ❌
• **Serious offense** - Murder, rape, terrorism
• **Strong evidence** - Clear case against accused
• **Flight risk** - May abscond
• **Witness tampering** - May influence evidence
• **Repeat offender** - History of similar crimes
• **Public safety** - Threat to society

## 📚 **IMPORTANT SECTIONS:**

• **Section 436** - Bail in non-bailable cases
• **Section 437** - When bail may be taken in non-bailable cases
• **Section 438** - Direction for grant of bail to person apprehending arrest
• **Section 439** - Special powers of High Court or Court of Session regarding bail
• **Section 440** - Amount of bond and reduction thereof

## 🏛️ **LANDMARK JUDGMENTS:**

• **Gurbaksh Singh Sibbia v. State of Punjab (1980)** - Anticipatory bail guidelines
• **Sanjay Chandra v. CBI (2012)** - Economic offenses and bail
• **Arnesh Kumar v. State of Bihar (2014)** - Arrest guidelines
• **Satender Kumar Antil v. CBI (2022)** - Bail as rule, jail as exception

## 💰 **BAIL CONDITIONS:**

### **COMMON CONDITIONS:**
• **Personal Bond** - ₹10,000 to ₹5,00,000 (varies)
• **Surety** - One or more sureties
• **Surrender passport** - In serious cases
• **Regular reporting** - Police station/court
• **No tampering** - With evidence or witnesses
• **Residence restriction** - Stay in jurisdiction

## 🚨 **WHEN BAIL IS DIFFICULT:**

• **Non-bailable offenses** - Murder (302 IPC), Rape (376 IPC)
• **NDPS cases** - Narcotic drugs offenses
• **Economic offenses** - Large fraud cases
• **Terror cases** - UAPA, NIA cases
• **Repeat offenders** - Habitual criminals

## ⏰ **TIME LIMITS:**

• **Regular bail** - No specific time limit
• **Anticipatory bail** - Before arrest occurs
• **Default bail** - 60/90 days without chargesheet (Section 167)
• **Statutory bail** - Automatic in certain conditions

## 📞 **PRACTICAL TIPS:**

• **Engage experienced criminal lawyer** immediately
• **Prepare strong grounds** for bail application
• **Arrange reliable sureties** beforehand
• **Keep all documents** ready
• **Follow bail conditions** strictly once granted

⚠️ **REMEMBER:** "Bail is the rule, jail is the exception" - Supreme Court principle

📋 **DISCLAIMER:** This is general information. Always consult a qualified criminal lawyer for specific cases and current legal position.""",
        "sources": ["Criminal Procedure Code 1973", "Supreme Court Judgments", "Bail Jurisprudence", "Criminal Law Practice"]
    },
    
    "company law": {
        "response": """**Indian Company Law Overview**

📋 **Primary Legislation:**
Companies Act, 2013 - Governs incorporation, management, and winding up of companies in India.

🏢 **Types of Companies:**
• **Private Limited Company** - Limited by shares, private
• **Public Limited Company** - Can raise funds from public
• **One Person Company (OPC)** - Single member company
• **Limited Liability Partnership (LLP)** - Hybrid structure

📚 **Key Compliance Requirements:**
• **Annual Returns** - Form MGT-7 (due within 60 days of AGM)
• **Annual Financial Statements** - Form AOC-4
• **Board Meetings** - Minimum 4 per year
• **Annual General Meeting** - Within 6 months of financial year end

⚖️ **Director Responsibilities:**
• **Fiduciary Duty** - Act in company's best interest
• **Due Diligence** - Exercise reasonable care and skill
• **Compliance** - Ensure statutory compliances
• **Disclosure** - Declare interests in contracts

🚨 **Common Violations & Penalties:**
• **Non-filing of returns** - ₹5,000 + daily penalty
• **Non-conduct of AGM** - ₹25,000 to ₹5,00,000
• **Director disqualification** - Various grounds under Section 164

📞 **Regulatory Bodies:**
• **Ministry of Corporate Affairs (MCA)** - Policy and administration
• **Registrar of Companies (ROC)** - Registration and compliance
• **National Company Law Tribunal (NCLT)** - Adjudication

⚠️ **Professional Advice:**
Always consult a Company Secretary or Corporate Lawyer for specific compliance issues.""",
        "sources": ["Companies Act 2013", "MCA Guidelines", "Corporate Governance"]
    },
    
    "private limited company": {
        "response": """**Private Limited Company in India**

📋 **Definition:**
A private limited company is a company whose shares are not freely transferable and cannot be offered to the general public.

🔍 **Key Features:**
• **Limited Liability** - Members' liability limited to unpaid share capital
• **Separate Legal Entity** - Company has its own legal identity
• **Perpetual Succession** - Continues despite changes in membership
• **Minimum 2 Directors** - Maximum 15 directors
• **Minimum 2 Members** - Maximum 200 members

📚 **Compliance Requirements:**
• **Annual Returns** - File Form MGT-7 within 60 days of AGM
• **Financial Statements** - File Form AOC-4 within 30 days of AGM
• **Board Meetings** - Minimum 4 meetings per year
• **AGM** - Must be held within 6 months of financial year end
• **Statutory Registers** - Maintain various registers at registered office

⚖️ **Advantages:**
• Limited liability protection
• Easy to raise capital
• Tax benefits and deductions
• Professional credibility
• Separate legal entity

🚨 **Disadvantages:**
• Extensive compliance requirements
• Higher cost of formation and maintenance
• Restrictions on share transfer
• Mandatory audit requirements

💰 **Penalties for Non-Compliance:**
• **Late filing of returns** - ₹5,000 + ₹50 per day
• **Non-conduct of AGM** - ₹25,000 to ₹5,00,000
• **Non-filing of financial statements** - ₹5,000 + daily penalty

📞 **Professional Help:**
Consult a Company Secretary or Chartered Accountant for proper compliance management.""",
        "sources": ["Companies Act 2013", "ROC Procedures", "Corporate Law"]
    },
    
    "fir": {
        "response": """**🚨 HOW TO FILE FIR - COMPLETE STEP-BY-STEP PROCESS**

📋 **Legal Framework:** Section 154, Code of Criminal Procedure, 1973

## 📝 **STEP-BY-STEP PROCESS TO FILE FIR:**

### **STEP 1: IMMEDIATE ACTIONS** ⏰
• **Go to nearest police station** within jurisdiction where crime occurred
• **Bring identification documents** (Aadhaar, PAN, Driving License)
• **Collect evidence** if available (photos, documents, witness details)
• **Note down time, date, location** of incident accurately

### **STEP 2: APPROACH THE POLICE** 👮‍♂️
• **Visit Station House Officer (SHO)** or duty officer
• **Inform about cognizable offense** (serious crimes like theft, assault, murder)
• **Request to file FIR** - it's your legal right under Section 154 CrPC
• **Police CANNOT refuse** to register FIR for cognizable offenses

### **STEP 3: PROVIDE COMPLETE INFORMATION** 📋
**Essential Details to Include:**
• **Your personal details** (name, address, contact number)
• **Detailed description of incident** (what, when, where, how)
• **Names of accused persons** (if known)
• **Names and addresses of witnesses**
• **Description of stolen/damaged property** (if applicable)
• **Injuries sustained** (if any)

### **STEP 4: FIR REGISTRATION PROCESS** ✍️
• **Police will write down your complaint** in FIR register
• **FIR will be read back to you** for verification
• **You must sign the FIR** after confirming accuracy
• **Get free copy of FIR** - this is your legal right
• **Note down FIR number** and date of registration

### **STEP 5: WHAT HAPPENS AFTER FIR** 🔍
• **Investigation begins immediately** under Section 156 CrPC
• **Police will visit crime scene** and collect evidence
• **Statements of witnesses** will be recorded under Section 161
• **You may be called for additional questioning**
• **Medical examination** if injuries are involved

## 📋 **DOCUMENTS REQUIRED:**
• **Identity Proof** (Aadhaar/PAN/Driving License)
• **Address Proof** (if different from ID)
• **Medical Certificate** (in case of injuries)
• **Evidence** (photos, receipts, documents related to crime)
• **Witness Details** (names, addresses, contact numbers)

## ⚖️ **YOUR LEGAL RIGHTS:**
• **Right to file FIR** - Police cannot refuse (Section 154)
• **Right to free copy** of FIR immediately
• **Right to add more information** later if remembered
• **Right to approach Magistrate** if police refuses to file FIR
• **Right to know investigation progress**

## 🚫 **WHAT IF POLICE REFUSES TO FILE FIR:**
1. **Approach Senior Police Officer** (SP/DCP)
2. **File complaint with Magistrate** under Section 156(3) CrPC
3. **Send written complaint by post** to police station
4. **Contact State Human Rights Commission**
5. **Approach High Court** under Article 226

## ⏰ **TIME LIMITS:**
• **No time limit** for filing FIR for serious offenses
• **File immediately** for better evidence collection
• **Within 24 hours** is ideal for most cases
• **Delay may affect investigation** quality

## 💰 **COST:**
• **Filing FIR is completely FREE**
• **Getting copy is FREE**
• **No fees for police investigation**

## 📞 **EMERGENCY CONTACTS:**
• **Police Emergency:** 100
• **Women Helpline:** 1091
• **Child Helpline:** 1098
• **Senior Citizen Helpline:** 14567

⚠️ **IMPORTANT NOTES:**
• FIR can only be filed for **cognizable offenses** (serious crimes)
• For **non-cognizable offenses**, file complaint under Section 155
• **False FIR** is punishable under Section 182 IPC
• Always keep **copy of FIR** safely for future reference

📚 **Legal Provisions:** Sections 154, 155, 156, 157, 161 of CrPC, 1973""",
        "sources": ["Code of Criminal Procedure 1973", "Police Manual", "Supreme Court Guidelines", "Legal Aid Handbook"]
    },
    "intellectual_property": {
        "response": """**Intellectual Property Law in India**

📋 **Primary Legislation:**
• The Copyright Act, 1957
• The Trademarks Act, 1999
• The Patents Act, 1970
• The Designs Act, 2000

⚖️ **Key Concepts:**
• **Copyright:** Protects original literary, dramatic, musical, and artistic works.
• **Trademark:** Protects brand names, logos, and slogans.
• **Patent:** Protects new and useful inventions.
• **Design:** Protects the ornamental or aesthetic aspect of an article.

🚨 **Common Issues:**
• **Infringement:** Unauthorized use of protected intellectual property.
• **Passing Off:** Misrepresenting goods or services as those of another.

⚠️ **Professional Advice:**
Always consult an Intellectual Property lawyer for specific advice on protecting and enforcing your IP rights.""",
        "sources": ["The Copyright Act 1957", "The Trademarks Act 1999", "The Patents Act 1970"]
    }
}

def get_legal_response(query: str) -> Dict[str, Any]:
    """Get legal response based on query"""
    query_lower = query.lower().strip()
    
    # Check for FIR-related queries first (highest priority)
    if any(term in query_lower for term in ["file fir", "fir filing", "how to file", "fir process", "register fir", "lodge fir", "complaint police", "police complaint", "fir step", "file complaint"]):
        return LEGAL_RESPONSES["fir"]
    
    # Check for bail-related queries (high priority)
    bail_keywords = ["bail", "anticipatory bail", "custody", "arrest", "bail procedure", "crpc bail", "section 437", "section 438", "regular bail", "interim bail", "bail application", "bail conditions", "surety", "bond"]
    if any(keyword in query_lower for keyword in bail_keywords):
        return LEGAL_RESPONSES["bail procedure"]
    
    # Enhanced keyword matching for better query recognition
    company_keywords = ["company", "private limited", "annual return", "filing", "roc", "director", "compliance"]
    constitutional_keywords = ["article", "constitution", "fundamental right"]
    criminal_keywords = ["ipc", "section", "murder", "theft", "fraud"]
    
    # Check for specific legal provisions
    for key, response_data in LEGAL_RESPONSES.items():
        if key in query_lower:
            return response_data
    
    # Enhanced company law detection
    if any(keyword in query_lower for keyword in company_keywords):
        if "annual return" in query_lower or "filing" in query_lower or "not filed" in query_lower:
            return LEGAL_RESPONSES["annual returns"]
        elif "private limited" in query_lower or "pvt ltd" in query_lower:
            return LEGAL_RESPONSES["private limited company"]
        else:
            return LEGAL_RESPONSES["company law"]
    
    # Contract law detection
    contract_keywords = ["contract", "agreement", "breach", "contract review", "contract law", "terms and conditions", "contract dispute", "contract violation", "contract enforcement"]
    if any(keyword in query_lower for keyword in contract_keywords):
        return {
            "response": """**📋 CONTRACT LAW IN INDIA - COMPREHENSIVE GUIDE**

📚 **Legal Framework:** Indian Contract Act, 1872

## 🔍 **CONTRACT REVIEW ESSENTIALS:**

### **KEY ELEMENTS TO REVIEW:** ✅
• **Parties** - Clear identification of contracting parties
• **Subject Matter** - Specific description of goods/services
• **Consideration** - Monetary or other valuable consideration
• **Terms & Conditions** - Rights, obligations, and responsibilities
• **Duration** - Start date, end date, renewal clauses
• **Termination** - Conditions for ending the contract
• **Dispute Resolution** - Arbitration, mediation, court jurisdiction

### **CRITICAL CLAUSES TO EXAMINE:** 🔍
• **Force Majeure** - Unforeseeable circumstances clause
• **Indemnity** - Protection against losses/damages
• **Confidentiality** - Non-disclosure provisions
• **Intellectual Property** - Ownership and usage rights
• **Limitation of Liability** - Caps on damages
• **Governing Law** - Which state/country laws apply
• **Payment Terms** - Due dates, penalties, interest

## ⚖️ **TYPES OF CONTRACTS:**

• **Sale of Goods** - Transfer of ownership
• **Service Agreements** - Provision of services
• **Employment Contracts** - Employer-employee relationship
• **Partnership Agreements** - Business partnerships
• **Lease Agreements** - Property rental
• **Non-Disclosure Agreements** - Confidentiality protection

## 🚨 **RED FLAGS IN CONTRACTS:**

• **Vague Terms** - Unclear obligations or deliverables
• **Unfair Penalties** - Excessive penalty clauses
• **One-sided Terms** - Heavily favoring one party
• **Missing Clauses** - No termination or dispute resolution
• **Unrealistic Deadlines** - Impossible performance timelines
• **Unlimited Liability** - No caps on damages

## 📋 **CONTRACT REVIEW CHECKLIST:**

### **BEFORE SIGNING:** ✅
• Read entire contract thoroughly
• Understand all terms and conditions
• Check for hidden fees or charges
• Verify party details and signatures
• Ensure compliance with applicable laws
• Get legal advice for complex contracts

### **KEY QUESTIONS TO ASK:** ❓
• What are my exact obligations?
• What happens if I can't perform?
• How can the contract be terminated?
• What are the penalty clauses?
• Who bears the risk of non-performance?
• Is there a cooling-off period?

## ⚖️ **BREACH OF CONTRACT:**

### **TYPES OF BREACH:**
• **Minor Breach** - Partial non-performance
• **Material Breach** - Substantial failure to perform
• **Anticipatory Breach** - Indication of future non-performance

### **REMEDIES AVAILABLE:**
• **Damages** - Monetary compensation
• **Specific Performance** - Court order to perform
• **Injunction** - Court order to stop/start action
• **Rescission** - Cancel the contract
• **Restitution** - Return to original position

## 📞 **PROFESSIONAL ADVICE:**

• **Simple Contracts** - Basic review possible
• **Complex Agreements** - Always consult lawyer
• **High-Value Contracts** - Mandatory legal review
• **International Contracts** - Specialized legal advice

⚠️ **IMPORTANT:** This is general guidance. Always consult a qualified contract lawyer for specific contract review and legal advice.

📋 **DISCLAIMER:** Contract law can be complex and fact-specific. Professional legal advice is recommended for all significant contracts.""",
            "sources": ["Indian Contract Act 1872", "Contract Law Practice", "Commercial Law", "Legal Precedents"]
        }
    
    # Constitutional law detection
    if any(keyword in query_lower for keyword in constitutional_keywords):
        # This will be handled by specific article matching below
        pass
    
    # Default response for unrecognized queries
    return {
        "response": f"""**Legal Inquiry: "{query[:100]}..."**

🙏 **Thank you for your legal query.**

📚 **I can help you with:**
• Constitutional Law (Articles 14, 19, 21, 32, etc.)
• Indian Penal Code (IPC Sections)
• Criminal Procedure Code (CrPC)
• Civil Procedure Code (CPC)
• Company Law and Corporate Compliance
• Contract Law and Property Law
• Family Law and Personal Laws

💡 **Try asking:**
• "What is Section 302 IPC?"
• "Explain Article 21 of Constitution"
• "Rights under Article 14"
• "Annual returns for private limited company"
• "Company law compliance requirements"

⚠️ **Important Disclaimer:**
This is general legal information only. For specific legal advice, please consult with a qualified advocate or legal practitioner.

📞 **For Professional Help:**
• Contact your local bar association
• Visit legal aid centers
• Consult with practicing advocates""",
        "sources": ["General Legal Knowledge", "Indian Law Database"]
    }

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "message": "⚡ Law GPT Enhanced API is running!",
        "status": "healthy",
        "version": "2.0.0"
    }

@app.options("/chat")
async def chat_options():
    """Handle CORS preflight requests for chat endpoint"""
    return {"message": "OK"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for legal queries"""
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get legal response
        response_data = get_legal_response(request.query)
        
        return ChatResponse(
            response=response_data["response"],
            sources=response_data.get("sources", [])
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def detailed_health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "service": "Law GPT Enhanced API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        },
        "legal_coverage": [
            "Constitutional Law",
            "Indian Penal Code", 
            "Criminal Procedure Code",
            "Company Law & Corporate Compliance",
            "Annual Returns & ROC Filings",
            "Director Responsibilities",
            "Bail Provisions",
            "Fundamental Rights"
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )