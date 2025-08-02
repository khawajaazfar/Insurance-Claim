import os
import re
import json
import requests
import yaml
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import LLMChain
import numpy as np
from groq import Groq

# Configuration
CONFIG_PATH = "config.yaml"

# Load configuration
def load_config():
    try:
        with open(CONFIG_PATH, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {CONFIG_PATH} not found. Using environment variables.")
        return {}

config = load_config()

# API Keys - prioritize config file, then environment variables
GROQ_API_KEY = config.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
QLOO_API_KEY = config.get('QLOO_API_KEY') or os.getenv('QLOO_API_KEY')

# Validate API keys
if not GROQ_API_KEY:
    print("WARNING: GROQ API key not found. Please set GROQ_API_KEY environment variable or add it to config.yaml")
    print("Create a config.yaml file with:\nGROQ_API_KEY: 'your_actual_api_key_here'")

if not QLOO_API_KEY:
    print("WARNING: QLOO API key not found. Please set QLOO_API_KEY environment variable or add it to config.yaml")

FAISS_PATH = "./faiss"

app = Flask(__name__)

# Initialize Groq client only if API key is available
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")

# General exclusion list
general_exclusion_list = [
    "HIV/AIDS", "Parkinson's disease", "Alzheimer's disease", "pregnancy",
    "substance abuse", "self-inflicted injuries", "sexually transmitted diseases",
    "pre-existing conditions", "cosmetic surgery", "experimental treatments",
    "dental care", "vision care", "mental health", "addiction treatment"
]

# Fraud detection patterns and thresholds
FRAUD_PATTERNS = {
    'high_amount_threshold': 50000,  # Claims above this amount need extra scrutiny
    'frequent_claims_threshold': 3,  # More than 3 claims in 30 days is suspicious
    'suspicious_keywords': [
        'urgent', 'emergency without prior consultation', 'rare disease',
        'expensive medication', 'experimental', 'abroad treatment'
    ],
    'round_number_threshold': 0.8,  # Suspicion if too many round numbers
    'weekend_emergency_weight': 2.0  # Weekend emergencies get higher scrutiny
}

# In-memory storage for fraud detection (in production, use a database)
claim_history = []

class FraudDetector:
    def __init__(self):
        self.risk_score = 0
        self.risk_factors = []
        
    def analyze_claim_amount(self, amount, disease):
        """Analyze if claim amount is suspicious"""
        try:
            amount = float(amount)
            
            # Check for unusually high amounts
            if amount > FRAUD_PATTERNS['high_amount_threshold']:
                self.risk_score += 30
                self.risk_factors.append(f"High claim amount: ${amount:,.2f}")
            
            # Check for round numbers (potential fabrication)
            if amount % 1000 == 0 and amount > 5000:
                self.risk_score += 15
                self.risk_factors.append("Suspiciously round claim amount")
                
            # Disease-amount correlation check
            if self._check_disease_amount_mismatch(disease, amount):
                self.risk_score += 25
                self.risk_factors.append("Amount inconsistent with reported condition")
                
        except ValueError:
            self.risk_score += 10
            self.risk_factors.append("Invalid claim amount format")
    
    def analyze_claim_frequency(self, patient_name, current_date):
        """Check for frequent claims from same patient"""
        current_date = datetime.strptime(current_date, '%Y-%m-%d')
        recent_claims = 0
        
        for claim in claim_history:
            if (claim['name'].lower() == patient_name.lower() and 
                (current_date - datetime.strptime(claim['date'], '%Y-%m-%d')).days <= 30):
                recent_claims += 1
        
        if recent_claims >= FRAUD_PATTERNS['frequent_claims_threshold']:
            self.risk_score += 40
            self.risk_factors.append(f"Multiple claims ({recent_claims}) in past 30 days")
    
    def analyze_claim_content(self, description, claim_reason):
        """Analyze claim description for suspicious patterns"""
        combined_text = f"{description} {claim_reason}".lower()
        
        # Check for suspicious keywords
        suspicious_count = 0
        for keyword in FRAUD_PATTERNS['suspicious_keywords']:
            if keyword.lower() in combined_text:
                suspicious_count += 1
                self.risk_factors.append(f"Suspicious keyword detected: {keyword}")
        
        if suspicious_count > 0:
            self.risk_score += suspicious_count * 15
        
        # Check for vague or inconsistent descriptions
        if len(description.strip()) < 20:
            self.risk_score += 10
            self.risk_factors.append("Vague claim description")
    
    def analyze_timing(self, claim_date, claim_type):
        """Analyze claim timing patterns"""
        try:
            date_obj = datetime.strptime(claim_date, '%Y-%m-%d')
            
            # Weekend emergency claims are more suspicious
            if date_obj.weekday() >= 5 and 'emergency' in claim_type.lower():
                self.risk_score += 20
                self.risk_factors.append("Weekend emergency claim")
                
            # Future dates are suspicious
            if date_obj > datetime.now():
                self.risk_score += 50
                self.risk_factors.append("Future dated claim")
                
        except ValueError:
            self.risk_score += 15
            self.risk_factors.append("Invalid date format")
    
    def analyze_medical_bill_consistency(self, bill_info, claim_amount, description):
        """Check consistency between bill and claim details"""
        try:
            bill_amount = float(bill_info.get('expense', 0))
            claim_amount = float(claim_amount)
            
            # Check for exact matches (potentially fabricated)
            if bill_amount == claim_amount:
                # This might actually be legitimate, so lower score
                pass
            elif abs(bill_amount - claim_amount) / max(bill_amount, claim_amount) > 0.1:
                self.risk_score += 20
                self.risk_factors.append("Significant discrepancy between bill and claim amount")
                
            # Check disease consistency
            bill_disease = bill_info.get('disease', '').lower()
            if bill_disease and bill_disease not in description.lower():
                self.risk_score += 25
                self.risk_factors.append("Disease mismatch between bill and description")
                
        except (ValueError, TypeError):
            self.risk_score += 15
            self.risk_factors.append("Inconsistent financial information")
    
    def _check_disease_amount_mismatch(self, disease, amount):
        """Check if amount is reasonable for the disease"""
        # Simple heuristic - in production, use ML model or disease-cost database
        common_minor_conditions = [
            'fever', 'cold', 'cough', 'headache', 'minor injury', 
            'routine checkup', 'vaccination'
        ]
        
        if any(condition in disease.lower() for condition in common_minor_conditions):
            return amount > 10000  # $10k seems high for minor conditions
        
        return False
    
    def get_risk_assessment(self):
        """Return final risk assessment"""
        if self.risk_score >= 80:
            risk_level = "HIGH"
            recommendation = "REJECT - High fraud probability"
        elif self.risk_score >= 50:
            risk_level = "MEDIUM"
            recommendation = "MANUAL REVIEW - Requires investigation"
        elif self.risk_score >= 20:
            risk_level = "LOW"
            recommendation = "APPROVE with monitoring"
        else:
            risk_level = "MINIMAL"
            recommendation = "APPROVE"
        
        return {
            'risk_score': self.risk_score,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'risk_factors': self.risk_factors
        }

# Helper functions
def get_document_loader():
    """Load PDF documents from the documents directory"""
    try:
        loader = DirectoryLoader('documents', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
        return loader.load()
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

def get_text_chunks(documents: list[Document]):
    """Split documents into chunks for better processing"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def get_embeddings():
    """Create FAISS embeddings using HuggingFace (free alternative to OpenAI)"""
    try:
        docs = get_document_loader()
        if not docs:
            return None
        chunks = get_text_chunks(docs)
        # Use free HuggingFace embeddings instead of OpenAI
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

def get_claim_approval_context():
    """Get context for claim approval requirements"""
    db = get_embeddings()
    if db:
        docs = db.similarity_search("What are the documents required for claim approval?", k=3)
        return "\n".join([doc.page_content for doc in docs])
    return "Standard claim approval documentation required."

def get_general_exclusion_context():
    """Get context for general exclusions"""
    db = get_embeddings()
    if db:
        docs = db.similarity_search("Give a list of all general exclusions", k=3)
        return "\n".join([doc.page_content for doc in docs])
    return "Standard exclusions apply as per policy terms."

def get_file_content(file):
    """Extract text content from uploaded PDF file"""
    text = ""
    try:
        if file and file.filename.endswith(".pdf"):
            pdf = PdfReader(file)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def get_bill_info_groq(data):
    """Extract bill information using Groq API"""
    if not groq_client:
        return {"disease": "API not configured", "expense": "0"}
    
    try:
        prompt = """Act as an expert in extracting information from medical invoices. 
        You are given invoice details of a patient. Go through the document carefully 
        and extract the 'disease' and the 'expense amount' from the data. 
        Return the data in valid JSON format: {"disease": "", "expense": ""}
        
        If you cannot find specific information, use empty strings but maintain the JSON structure."""
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"INVOICE DETAILS: {data}"}
            ],
            model="llama3-8b-8192",  # Using Llama 3 8B model
            temperature=0.1,
            max_tokens=1000
        )
        
        response_content = chat_completion.choices[0].message.content.strip()
        
        # Clean up response to ensure valid JSON
        if '```json' in response_content:
            response_content = response_content.split('```json')[1].split('```')[0].strip()
        elif '```' in response_content:
            response_content = response_content.split('```')[1].strip()
        
        return json.loads(response_content)
        
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return {"disease": "", "expense": ""}

def validate_claim_with_groq(patient_info, medical_bill_info, claim_approval_context, general_exclusion_context, fraud_assessment):
    """Validate claim using Groq LLM with fraud assessment"""
    
    if not groq_client:
        return "Error: Groq API not configured. Please set up your API key."
    
    PROMPT = f"""You are an AI assistant for verifying health insurance claims. Your role is to analyze 
    the provided information and determine whether a claim should be approved or rejected.

    FRAUD RISK ASSESSMENT:
    Risk Score: {fraud_assessment['risk_score']}/100
    Risk Level: {fraud_assessment['risk_level']}
    Risk Factors: {', '.join(fraud_assessment['risk_factors']) if fraud_assessment['risk_factors'] else 'None detected'}
    Recommendation: {fraud_assessment['recommendation']}

    CLAIM APPROVAL REQUIREMENTS:
    {claim_approval_context}

    GENERAL EXCLUSIONS:
    {general_exclusion_context}

    PATIENT INFORMATION:
    {patient_info}

    MEDICAL BILL INFORMATION:
    {medical_bill_info}

    Please provide a comprehensive analysis including:
    1. Eligibility assessment based on policy terms
    2. Fraud risk consideration
    3. Final recommendation (APPROVE/REJECT/MANUAL REVIEW)
    4. Reasoning for your decision
    5. Any additional requirements or conditions

    Format your response clearly with sections for each point."""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert insurance claim analyst with fraud detection capabilities."},
                {"role": "user", "content": PROMPT}
            ],
            model="llama3-70b-8192",  # Using larger model for complex analysis
            temperature=0.2,
            max_tokens=2000
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"Error with Groq validation: {e}")
        return f"Error processing claim validation: {str(e)}"

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Extract form data
            name = request.form.get('name', '').strip()
            address = request.form.get('address', '').strip()
            claim_type = request.form.get('claim_type', '').strip()
            claim_reason = request.form.get('claim_reason', '').strip()
            date = request.form.get('date', '').strip()
            medical_facility = request.form.get('medical_facility', '').strip()
            medical_bill = request.files.get('medical_bill')
            total_claim_amount = request.form.get('total_claim_amount', '0').strip()
            description = request.form.get('description', '').strip()

            # Validate required fields
            if not all([name, claim_type, claim_reason, date, total_claim_amount]):
                return render_template("result.html", 
                                     name=name,
                                     address=address,
                                     claim_type=claim_type,
                                     claim_reason=claim_reason,
                                     medical_facility=medical_facility,
                                     total_claim_amount=total_claim_amount,
                                     description=description,
                                     output="Error: Missing required fields. Please fill all mandatory fields.",
                                     claim_validation_message="")

            # Process medical bill
            bill_content = get_file_content(medical_bill) if medical_bill else ""
            bill_info = get_bill_info_groq(bill_content) if bill_content else {"disease": "", "expense": ""}

            # Initialize fraud detector
            fraud_detector = FraudDetector()
            
            # Run fraud detection analysis
            fraud_detector.analyze_claim_amount(total_claim_amount, bill_info.get('disease', ''))
            fraud_detector.analyze_claim_frequency(name, date)
            fraud_detector.analyze_claim_content(description, claim_reason)
            fraud_detector.analyze_timing(date, claim_type)
            if bill_info.get('expense'):
                fraud_detector.analyze_medical_bill_consistency(bill_info, total_claim_amount, description)
            
            fraud_assessment = fraud_detector.get_risk_assessment()

            # Store claim in history for fraud detection
            claim_history.append({
                'name': name,
                'date': date,
                'amount': total_claim_amount,
                'disease': bill_info.get('disease', ''),
                'timestamp': datetime.now().isoformat()
            })

            # Basic amount validation
            if bill_info.get('expense'):
                try:
                    bill_expense = float(bill_info['expense'])
                    claim_amount = float(total_claim_amount)
                    
                    if bill_expense < claim_amount:
                        claim_validation_message = f"REJECTED: Claim amount (${claim_amount:,.2f}) exceeds bill amount (${bill_expense:,.2f})"
                        return render_template("result.html", 
                                             name=name,
                                             address=address,
                                             claim_type=claim_type,
                                             claim_reason=claim_reason,
                                             medical_facility=medical_facility,
                                             total_claim_amount=total_claim_amount,
                                             description=description,
                                             claim_validation_message=claim_validation_message,
                                             fraud_assessment=fraud_assessment,
                                             output="")
                except ValueError:
                    pass

            # Prepare information for LLM analysis
            patient_info = f"""
            Name: {name}
            Address: {address}
            Claim Type: {claim_type}
            Claim Reason: {claim_reason}
            Medical Facility: {medical_facility}
            Date: {date}
            Total Claim Amount: ${total_claim_amount}
            Description: {description}
            """

            medical_bill_info = f"""
            Medical Bill Content: {bill_content[:1000]}...
            Extracted Disease: {bill_info.get('disease', 'Not specified')}
            Extracted Expense: ${bill_info.get('expense', 'Not specified')}
            """

            # Get LLM analysis with fraud assessment
            output = validate_claim_with_groq(
                patient_info, 
                medical_bill_info, 
                get_claim_approval_context(),
                get_general_exclusion_context(),
                fraud_assessment
            )

            # Format output for HTML display
            output = re.sub(r'\n', '<br>', output)
            
            return render_template("result.html", 
                                 name=name,
                                 address=address,
                                 claim_type=claim_type,
                                 claim_reason=claim_reason,
                                 medical_facility=medical_facility,
                                 total_claim_amount=total_claim_amount,
                                 description=description,
                                 output=output,
                                 fraud_assessment=fraud_assessment,
                                 bill_info=bill_info,
                                 claim_validation_message="")

        except Exception as e:
            error_message = f"An error occurred while processing your claim: {str(e)}"
            return render_template("result.html", 
                                 name=name if 'name' in locals() else '',
                                 address=address if 'address' in locals() else '',
                                 claim_type=claim_type if 'claim_type' in locals() else '',
                                 claim_reason=claim_reason if 'claim_reason' in locals() else '',
                                 medical_facility=medical_facility if 'medical_facility' in locals() else '',
                                 total_claim_amount=total_claim_amount if 'total_claim_amount' in locals() else '',
                                 description=description if 'description' in locals() else '',
                                 output=error_message,
                                 claim_validation_message="")

    return render_template("index.html")

@app.route('/fraud-stats')
def fraud_statistics():
    """Endpoint to view fraud detection statistics"""
    total_claims = len(claim_history)
    if total_claims == 0:
        return jsonify({"message": "No claims processed yet"})
    
    # Calculate basic statistics
    recent_claims = [claim for claim in claim_history 
                    if (datetime.now() - datetime.fromisoformat(claim['timestamp'])).days <= 7]
    
    stats = {
        "total_claims": total_claims,
        "recent_claims_7_days": len(recent_claims),
        "average_claim_amount": sum(float(claim['amount']) for claim in claim_history) / total_claims,
        "latest_claims": claim_history[-5:] if total_claims >= 5 else claim_history
    }
    
    return jsonify(stats)

@app.route('/insights')
def get_qloo_insights():
    """Get insights from Qloo API"""
    if not QLOO_API_KEY:
        return "Error: Qloo API key not configured. Please set up your API key."
    
    url = "https://hackathon.api.qloo.com/v2/insights/?filter.type=urn:entity:place&signal.interests.entities=FCE8B172-4795-43E4-B222-3B550DC05FD9&filter.location.query=New%20York"
    headers = {"X-Api-Key": QLOO_API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return render_template("insights.html", insights=response.json())
        else:
            return f"Qloo API Error {response.status_code}: {response.text}"
    except requests.RequestException as e:
        return f"Error connecting to Qloo API: {str(e)}"

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "groq_api_configured": bool(GROQ_API_KEY),
        "qloo_api_configured": bool(QLOO_API_KEY),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting Enhanced Insurance Claim System...")
    print(f"Groq API configured: {bool(GROQ_API_KEY)}")
    print(f"Qloo API configured: {bool(QLOO_API_KEY)}")
    
    if not GROQ_API_KEY:
        print("\n⚠️  SETUP REQUIRED:")
        print("Please create a 'config.yaml' file with your API keys:")
        print("GROQ_API_KEY: 'your_groq_api_key_here'")
        print("QLOO_API_KEY: 'your_qloo_api_key_here'")
        print("\nAlternatively, set environment variables:")
        print("export GROQ_API_KEY='your_groq_api_key_here'")
        print("export QLOO_API_KEY='your_qloo_api_key_here'")
    
    print("Fraud detection module loaded successfully")

    app.run(host='0.0.0.0', port=8085, debug=False, use_reloader=False)

