
from PyPDF2 import PdfReader
from groq import Groq
import yaml
import json
import os
from datetime import datetime

CONFIG_PATH = "config.yaml"

class BillExtractor:
    def __init__(self, config_path=CONFIG_PATH):
        self.config = self.load_config(config_path)
        self.groq_client = self.initialize_groq()
        
    def load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default settings.")
            return {"GROQ_API_KEY": os.getenv('GROQ_API_KEY', '')}
    
    def initialize_groq(self):
        """Initialize Groq client"""
        api_key = self.config.get('GROQ_API_KEY') or os.getenv('GROQ_API_KEY')
        if not api_key or api_key == 'gsk_your_actual_groq_api_key_here':
            raise ValueError("Groq API key not configured. Please update config.yaml or set GROQ_API_KEY environment variable.")
        return Groq(api_key=api_key)

    def extract_pdf_text(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist")
            
            pdf = PdfReader(file_path)
            total_pages = len(pdf.pages)
            
            print(f"Processing PDF with {total_pages} pages...")
            
            for page_num in range(total_pages):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
                
            print(f"Successfully extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            raise

    def extract_invoice_info(self, pdf_text, use_advanced_prompt=True):
        """Extract invoice information using Groq API with enhanced prompts"""
        
        if use_advanced_prompt:
            prompt = """You are an expert medical billing analyst with years of experience in processing healthcare invoices. 
            Your task is to carefully analyze the provided medical invoice/bill and extract key information.

            Please extract the following information from the medical invoice:
            1. **Disease/Condition**: The primary medical condition, diagnosis, or reason for treatment
            2. **Total Expense Amount**: The total amount billed (look for final total, grand total, amount due, etc.)
            3. **Additional Details**: Any other relevant medical or billing information

            **Important Guidelines:**
            - Look for medical terminology, ICD codes, procedure codes, diagnosis descriptions
            - For amounts, prioritize: "Total Due", "Amount Due", "Grand Total", "Final Amount"
            - If multiple amounts exist, choose the final billable amount
            - Extract actual currency values (remove currency symbols, keep only numbers)
            - If information is unclear or missing, indicate "Not clearly specified"
            - Be precise and only extract information that is clearly visible in the document

            Return the information in this exact JSON format:
            {
                "disease": "specific condition or diagnosis found",
                "expense": "numerical amount only (no currency symbols)",
                "additional_info": "any other relevant details",
                "confidence": "high/medium/low based on clarity of information"
            }

            If you cannot find specific information, use descriptive text like "Not clearly specified" but maintain the JSON structure.
            """
        else:
            prompt = """Act as an expert in extracting information from medical invoices. 
            You are given invoice details of a patient. Go through the document carefully 
            and extract the 'disease' and the 'expense amount' from the data. 
            Return the data in valid JSON format: {"disease": "", "expense": ""}"""

        try:
            print("Sending request to Groq API...")
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a highly skilled medical billing expert specializing in accurate information extraction from healthcare documents."
                    },
                    {
                        "role": "user", 
                        "content": f"{prompt}\n\nINVOICE DETAILS:\n{pdf_text}"
                    }
                ],
                model="llama3-70b-8192",  # Using the larger, more capable model
                temperature=0.1,  # Low temperature for consistent, factual extraction
                max_tokens=1500,
                top_p=0.9
            )
            
            response_content = chat_completion.choices[0].message.content.strip()
            print(f"Received response from Groq API: {response_content[:200]}...")
            
            # Clean up response to extract JSON
            json_str = self._extract_json_from_response(response_content)
            
            try:
                extracted_data = json.loads(json_str)
                
                # Validate and clean the extracted data
                cleaned_data = self._validate_and_clean_data(extracted_data)
                
                print("Successfully extracted and validated data")
                return cleaned_data
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {response_content}")
                
                # Fallback: try to extract basic info manually
                return self._fallback_extraction(response_content, pdf_text)
                
        except Exception as e:
            print(f"Error with Groq API: {str(e)}")
            return {
                "disease": "Error extracting information",
                "expense": "0",
                "additional_info": f"Error: {str(e)}",
                "confidence": "low"
            }

    def _extract_json_from_response(self, response_content):
        """Extract JSON from the response, handling various formats"""
        
        # Try to find JSON in code blocks first
        if '```json' in response_content:
            json_str = response_content.split('```json')[1].split('```')[0].strip()
        elif '```' in response_content:
            # Generic code block
            json_str = response_content.split('```')[1].strip()
        elif '{' in response_content and '}' in response_content:
            # Find JSON-like content
            start = response_content.find('{')
            end = response_content.rfind('}') + 1
            json_str = response_content[start:end]
        else:
            json_str = response_content
            
        return json_str

    def _validate_and_clean_data(self, data):
        """Validate and clean extracted data"""
        cleaned = {
            "disease": str(data.get("disease", "")).strip(),
            "expense": str(data.get("expense", "0")).strip(),
            "additional_info": str(data.get("additional_info", "")).strip(),
            "confidence": str(data.get("confidence", "medium")).strip().lower()
        }
        
      