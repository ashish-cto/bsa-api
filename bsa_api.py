import base64
import json
import logging
import os
import re
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal
from enum import Enum

import anthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Header, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Define backend types as enum for type safety
class BackendType(str, Enum):
    ANTHROPIC_CLAUDE = "ANTHROPIC_CLAUDE"
    GOOGLE_GEMINI = "GOOGLE_GEMINI"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bsa_api.log")
    ]
)
logger = logging.getLogger("bsa_api")

# Create FastAPI app
app = FastAPI(
    title="Bank Statement Analyzer API",
    description="API for analyzing Indian bank statements using Claude AI",
    version="1.0.0"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Models -----

class LeadMetadata(BaseModel):
    lead_id: str = Field(..., description="Unique identifier for the lead")
    source: Optional[str] = Field(None, description="Source of the lead")
    application_id: Optional[str] = Field(None, description="Application ID if available")
    customer_name: Optional[str] = Field(None, description="Name of the customer")
    additional_info: Optional[Dict[str, Any]] = Field(None, description="Any additional metadata")


class ProcessingRequest(BaseModel):
    metadata: LeadMetadata
    custom_prompt: Optional[str] = Field(None, description="Custom prompt to override default")
    model: Optional[str] = Field(None, description="Model to use for processing")


class SalaryTransaction(BaseModel):
    date: str
    amount: float
    description: str
    mode: str
    type: str


class FraudulentSalaryAlert(BaseModel):
    date: str
    amount: float
    description: str
    reason: str
    type: str


class ProcessingResponse(BaseModel):
    # Core response fields
    lead_id: str
    processing_id: str
    request_id: int = Field(default=-1, description="Database request ID, -1 if not stored")
    timestamp: str
    processing_time_seconds: float
    status: str
    error_message: Optional[str] = None
    
    # Account information
    account_holder_name: Optional[str] = Field(None, description="Name of the account holder")
    account_number: Optional[str] = Field(None, description="Account number from the statement")
    
    # Financial analysis results
    total_salary_received: float = 0.0
    number_of_salary_credits: int = 0
    salary_transactions: List[SalaryTransaction] = []
    average_monthly_salary: float = 0.0
    suspicious_salary_transactions: List[FraudulentSalaryAlert] = []
    
    # Loan eligibility results - NEW FIELD NAMES ONLY
    loan_eligibility: bool = False
    max_loan_eligibility: float = 0.0
    loan_eligibility_comments: str = ""
    
    # Debug and metadata fields - ONLY INCLUDED WHEN debug=True
    # These fields will be completely excluded from the model unless explicitly set
    estimated_api_cost_usd: Optional[float] = Field(default=None)
    backend_used: Optional[str] = Field(default=None)
    ai_model_used: Optional[str] = Field(default=None)
    
    class Config:
        # This config ensures better handling of None values
        use_enum_values = True
        validate_assignment = True


# ----- Configuration Functions -----

def load_config():
    """Load configuration from config.xml file"""
    config_file = Path("config.xml")

    # Default configuration
    default_config = {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "model": "claude-sonnet-4-20250514",
        # Separate model lists for each backend
        "available_claude_models": [
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022"
        ],
        "available_gemini_models": [
            "gemini-1.5-pro",
            "gemini-1.5-flash"
            # "gemini-2.5-flash" removed due to performance/stability issues
        ],
        # Keep legacy available_models for backward compatibility
        "available_models": [
            "claude-sonnet-4-20250514",
            "claude-3-7-sonnet-20250219",
            "claude-3-5-sonnet-20241022"
        ],
        "max_file_size_mb": 10,
        "supported_formats": ["pdf"],
        "api_keys": [],  # List of valid API keys
        "rate_limit": 100,  # Requests per minute
        "extraction_prompt_claude": """
You are a financial document parser. I will upload a scanned or digital bank statement PDF from an Indian bank. Your task is to calculate and return only aggregated totals for key transaction categories from the entire statement.

### CRITICAL TRANSACTION TYPE IDENTIFICATION:
NEVER use the transaction description to determine whether it is credit or debit! Always process the document / table visually and look ONLY at table headers and corresponding vertically aligned values to determine the transaction type. DO NOT ASSUME that finding words like "salary" in the description means that the transaction is a credit!!!

Look for columns labeled "Credit", "Debit", "Dr", "Cr", "Withdrawal", "Deposit" or similar. A transaction is only a credit if there is a positive amount in the Credit/Deposit column, and only a debit if there is a positive amount in the Debit/Withdrawal column.

### Output Format:
Return the result in JSON with this exact structure:
{
  "account_holder_name": "",
  "account_number": "",
  "total_salary_received": 0.0,
  "number_of_salary_credits": 0,
  "salary_transactions": [],
  "total_emi_paid": 0.0,
  "number_of_emi_debits": 0,
  "total_loans_received": 0.0,
  "number_of_loans_received": 0,
  "total_upi_credits": 0.0,
  "total_upi_debits": 0.0,
  "closing_balance": 0.0,
  "average_monthly_salary": 0.0,
  "suspicious_salary_transactions": []
}

### Classification Rules:
- **Account Information**:
  - Extract the account holder name from the statement header/top section
  - Extract the account number from the statement header/top section
  - If not clearly visible, leave as empty string ""

- **Legitimate Salary**:
  - Credit transactions only via NEFT or RTGS (never UPI)
  - Description contains words like "salary", "payroll", "HRMS", "SAL", "PAY" or is a recurring monthly credit from the same source
  - Must be bank transfer (NEFT/RTGS), not UPI transactions
  - IMPORTANT: If you find a normal-looking salary transaction with a company name in it, look for other CREDIT transactions from the same company that might also be salary payments (even if they don't explicitly mention "salary"). Consider recurring amounts from the same company as potential salary payments.
  - When you find a credit transaction that matches the company name of a known salary transaction, include it as a legitimate salary transaction even if it doesn't contain explicit salary keywords.
  - If no clear legitimate salary is found at first, look for CREDIT transactions greater than INR 30,000 with company names ending with "LTD" or "Limited" in their description. If 2 or more CREDIT transactions are found with a similar amount and roughly a gap of 1 month between them, treat all such transactions as salary even if they don't have a keyword like "SAL" or "SALARY".
  - For each legitimate salary transaction, include: {"date": "DD-MM-YYYY", "amount": 12345.67, "description": "transaction description", "mode": "NEFT/RTGS", "type": "Credit"}
  - Calculate the average_monthly_salary field by dividing total_salary_received by the number_of_salary_credits

- **Suspicious Salary Transactions**:
  - ABSOLUTELY NEVER FLAG WITHDRAWALS, DEBITS, OR OUTGOING PAYMENTS
  - ONLY UPI CREDIT transactions (money coming INTO the account) with "salary", "payroll", "SAL", "PAY" or similar terms
  - If a transaction is a debit/withdrawal/outgoing payment, DO NOT include it in suspicious_salary_transactions regardless of description
  - Only transactions where money is RECEIVED (credits) can be flagged as suspicious salary
  - For each suspicious CREDIT transaction, include: {"date": "DD-MM-YYYY", "amount": 12345.67, "description": "transaction description", "reason": "UPI credit claiming to be salary payment", "type": "Credit"}
  - The "reason" field should be a short, clear explanation like "UPI credit claiming to be salary payment" or "Unusual salary payment method"

- **Loan Disbursal**:
  - Credit transactions only
  - Description contains words like "loan", "mpokket", "finance", "NBFC", "earlysalary", "kissht", or similar.

- **EMI**:
  - Debit transactions only
  - Description contains "EMI", "instalment", "loan repayment", or recurring pattern matching loans.

- **UPI**:
  - Any transaction mentioning UPI, PhonePe, Paytm, GPay. Sum separately for credits and debits.
  - Exclude UPI transactions that are flagged as suspicious salary transactions from UPI totals.

- Exclude non-transactional info like headers, footers, page numbers.
- Extract the **final closing balance** from the statement.

### CRITICAL Requirements:
- Return **only valid JSON**, no extra text or commentary.
- All monetary amounts must be numeric values (not strings) - example: 1234.56 not "1234.56"
- All counts must be integer values (not strings) - example: 5 not "5"
- If no transactions match a category, return 0 for amounts, 0 for counts, and empty arrays for transaction lists.
- Do not include currency symbols or commas in numeric values.
- Legitimate salary transactions must be NEFT/RTGS only - never UPI.
- NEVER EVER flag debit/withdrawal transactions as suspicious salary - only credit transactions can be salary.
- Always include "type" field indicating "Credit" or "Debit" for all transaction records.
- ALWAYS determine Credit/Debit status by looking at the table structure and column headers, NEVER by transaction description content.
- For account_holder_name and account_number, extract from statement header - if not visible, use empty string "".
""",
        # Initially identical to Claude prompt but can evolve separately
        "extraction_prompt_gemini": """
You are a financial document parser. I will upload a scanned or digital bank statement PDF from an Indian bank. Your task is to calculate and return only aggregated totals for key transaction categories from the entire statement.

### CRITICAL TRANSACTION TYPE IDENTIFICATION:
NEVER use the transaction description to determine whether it is credit or debit! Always process the document / table visually and look ONLY at table headers and corresponding vertically aligned values to determine the transaction type. DO NOT ASSUME that finding words like "salary" in the description means that the transaction is a credit!!!

Look for columns labeled "Credit", "Debit", "Dr", "Cr", "Withdrawal", "Deposit" or similar. A transaction is only a credit if there is a positive amount in the Credit/Deposit column, and only a debit if there is a positive amount in the Debit/Withdrawal column.

### Output Format:
Return the result in JSON with this exact structure:
{
  "account_holder_name": "",
  "account_number": "",
  "total_salary_received": 0.0,
  "number_of_salary_credits": 0,
  "salary_transactions": [],
  "total_emi_paid": 0.0,
  "number_of_emi_debits": 0,
  "total_loans_received": 0.0,
  "number_of_loans_received": 0,
  "total_upi_credits": 0.0,
  "total_upi_debits": 0.0,
  "closing_balance": 0.0,
  "average_monthly_salary": 0.0,
  "suspicious_salary_transactions": []
}

### Classification Rules:
- **Account Information**:
  - Extract the account holder name from the statement header/top section
  - Extract the account number from the statement header/top section
  - If not clearly visible, leave as empty string ""

- **Legitimate Salary**:
  - Credit transactions only via NEFT or RTGS (never UPI)
  - Description contains words like "salary", "payroll", "HRMS", "SAL", "PAY" or is a recurring monthly credit from the same source
  - Must be bank transfer (NEFT/RTGS), not UPI transactions
  - IMPORTANT: If you find a normal-looking salary transaction with a company name in it, look for other CREDIT transactions from the same company that might also be salary payments (even if they don't explicitly mention "salary"). Consider recurring amounts from the same company as potential salary payments.
  - When you find a credit transaction that matches the company name of a known salary transaction, include it as a legitimate salary transaction even if it doesn't contain explicit salary keywords.
  - If no clear legitimate salary is found at first, look for CREDIT transactions greater than INR 30,000 with company names ending with "LTD" or "Limited" in their description. If 2 or more CREDIT transactions are found with a similar amount and roughly a gap of 1 month between them, treat all such transactions as salary even if they don't have a keyword like "SAL" or "SALARY".
  - For each legitimate salary transaction, include: {"date": "DD-MM-YYYY", "amount": 12345.67, "description": "transaction description", "mode": "NEFT/RTGS", "type": "Credit"}
  - Calculate the average_monthly_salary field by dividing total_salary_received by the number_of_salary_credits

- **Suspicious Salary Transactions**:
  - ABSOLUTELY NEVER FLAG WITHDRAWALS, DEBITS, OR OUTGOING PAYMENTS
  - ONLY UPI CREDIT transactions (money coming INTO the account) with "salary", "payroll", "SAL", "PAY" or similar terms
  - If a transaction is a debit/withdrawal/outgoing payment, DO NOT include it in suspicious_salary_transactions regardless of description
  - Only transactions where money is RECEIVED (credits) can be flagged as suspicious salary
  - For each suspicious CREDIT transaction, include: {"date": "DD-MM-YYYY", "amount": 12345.67, "description": "transaction description", "reason": "UPI credit claiming to be salary payment", "type": "Credit"}
  - The "reason" field should be a short, clear explanation like "UPI credit claiming to be salary payment" or "Unusual salary payment method"

- **Loan Disbursal**:
  - Credit transactions only
  - Description contains words like "loan", "mpokket", "finance", "NBFC", "earlysalary", "kissht", or similar.

- **EMI**:
  - Debit transactions only
  - Description contains "EMI", "instalment", "loan repayment", or recurring pattern matching loans.

- **UPI**:
  - Any transaction mentioning UPI, PhonePe, Paytm, GPay. Sum separately for credits and debits.
  - Exclude UPI transactions that are flagged as suspicious salary transactions from UPI totals.

- Exclude non-transactional info like headers, footers, page numbers.
- Extract the **final closing balance** from the statement.

### CRITICAL Requirements:
- Return **only valid JSON**, no extra text or commentary.
- All monetary amounts must be numeric values (not strings) - example: 1234.56 not "1234.56"
- All counts must be integer values (not strings) - example: 5 not "5"
- If no transactions match a category, return 0 for amounts, 0 for counts, and empty arrays for transaction lists.
- Do not include currency symbols or commas in numeric values.
- Legitimate salary transactions must be NEFT/RTGS only - never UPI.
- NEVER EVER flag debit/withdrawal transactions as suspicious salary - only credit transactions can be salary.
- Always include "type" field indicating "Credit" or "Debit" for all transaction records.
- ALWAYS determine Credit/Debit status by looking at the table structure and column headers, NEVER by transaction description content.
- For account_holder_name and account_number, extract from statement header - if not visible, use empty string "".
""",
        # Google Gemini configuration
        "google_api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "gemini_model": "gemini-1.5-pro",
    }

    def create_default_xml(config_dict):
        """Create a default XML config file"""
        root = ET.Element("config")

        # Add basic settings
        ET.SubElement(root, "anthropic_api_key").text = config_dict["anthropic_api_key"]
        ET.SubElement(root, "model").text = config_dict["model"]
        ET.SubElement(root, "max_file_size_mb").text = str(config_dict["max_file_size_mb"])
        ET.SubElement(root, "rate_limit").text = str(config_dict["rate_limit"])
        
        # Add Google API settings
        ET.SubElement(root, "google_api_key").text = config_dict["google_api_key"]
        ET.SubElement(root, "gemini_model").text = config_dict["gemini_model"]

        # Add available Claude models as a list
        claude_models_elem = ET.SubElement(root, "available_claude_models")
        for model in config_dict["available_claude_models"]:
            ET.SubElement(claude_models_elem, "model").text = model

        # Add available Gemini models as a list
        gemini_models_elem = ET.SubElement(root, "available_gemini_models")
        for model in config_dict["available_gemini_models"]:
            ET.SubElement(gemini_models_elem, "model").text = model

        # Add legacy available models for backward compatibility
        models_elem = ET.SubElement(root, "available_models")
        for model in config_dict["available_models"]:
            ET.SubElement(models_elem, "model").text = model

        # Add supported formats as a list
        formats_elem = ET.SubElement(root, "supported_formats")
        for fmt in config_dict["supported_formats"]:
            ET.SubElement(formats_elem, "format").text = fmt

        # Add API keys
        api_keys_elem = ET.SubElement(root, "api_keys_bsa")
        for key in config_dict["api_keys"]:
            ET.SubElement(api_keys_elem, "key").text = key

        # Add extraction prompts
        ET.SubElement(root, "extraction_prompt_claude").text = config_dict["extraction_prompt_claude"]
        ET.SubElement(root, "extraction_prompt_gemini").text = config_dict["extraction_prompt_gemini"]

        return root

    def parse_xml_to_dict(root):
        """Parse XML config to dictionary"""
        config = {}

        # Basic settings
        config["anthropic_api_key"] = root.find("anthropic_api_key").text or ""
        config["model"] = root.find("model").text or "claude-sonnet-4-20250514"
        config["max_file_size_mb"] = int(root.find("max_file_size_mb").text or "10")
        config["rate_limit"] = int(root.find("rate_limit").text or "100")
        
        # Google API settings
        config["google_api_key"] = root.find("google_api_key").text or ""
        config["gemini_model"] = root.find("gemini_model").text or "gemini-1.5-pro"

        # Available Claude models
        claude_models_elem = root.find("available_claude_models")
        config["available_claude_models"] = []
        if claude_models_elem is not None:
            for model_elem in claude_models_elem.findall("model"):
                if model_elem.text:
                    config["available_claude_models"].append(model_elem.text)

        # Available Gemini models
        gemini_models_elem = root.find("available_gemini_models")
        config["available_gemini_models"] = []
        if gemini_models_elem is not None:
            for model_elem in gemini_models_elem.findall("model"):
                if model_elem.text:
                    config["available_gemini_models"].append(model_elem.text)

        # Legacy available models (for backward compatibility)
        models_elem = root.find("available_models")
        config["available_models"] = []
        if models_elem is not None:
            for model_elem in models_elem.findall("model"):
                if model_elem.text:
                    config["available_models"].append(model_elem.text)

        # Default to Claude models if legacy available_models is empty
        if not config["available_models"] and config["available_claude_models"]:
            config["available_models"] = config["available_claude_models"].copy()

        # Default to available models if not specified
        if not config["available_claude_models"]:
            config["available_claude_models"] = default_config["available_claude_models"]
        
        if not config["available_gemini_models"]:
            config["available_gemini_models"] = default_config["available_gemini_models"]

        # Supported formats
        formats_elem = root.find("supported_formats")
        config["supported_formats"] = []
        if formats_elem is not None:
            for fmt_elem in formats_elem.findall("format"):
                if fmt_elem.text:
                    config["supported_formats"].append(fmt_elem.text)

        # Default to PDF if no formats specified
        if not config["supported_formats"]:
            config["supported_formats"] = ["pdf"]

        # API keys
        api_keys_elem = root.find("api_keys_bsa")
        config["api_keys"] = []
        if api_keys_elem is not None:
            for key_elem in api_keys_elem.findall("key"):
                if key_elem.text:
                    config["api_keys"].append(key_elem.text)

        # For backward compatibility, also check the old tag name
        if not config["api_keys"]:
            legacy_api_keys_elem = root.find("api_keys")
            if legacy_api_keys_elem is not None:
                for key_elem in legacy_api_keys_elem.findall("key"):
                    if key_elem.text:
                        config["api_keys"].append(key_elem.text)

        # Extraction prompts
        prompt_claude_elem = root.find("extraction_prompt_claude")
        config["extraction_prompt_claude"] = prompt_claude_elem.text or default_config["extraction_prompt_claude"]
        
        prompt_gemini_elem = root.find("extraction_prompt_gemini")
        config["extraction_prompt_gemini"] = prompt_gemini_elem.text or default_config["extraction_prompt_gemini"]
        
        # For backward compatibility
        prompt_elem = root.find("extraction_prompt")
        if prompt_elem is not None and prompt_elem.text:
            # If old format exists, copy it to both new prompts
            config["extraction_prompt_claude"] = prompt_elem.text
            config["extraction_prompt_gemini"] = prompt_elem.text

        return config

    def pretty_print_xml(element):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(element, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").split('\n', 1)[1]  # Remove XML declaration

    try:
        if config_file.exists():
            # Load existing XML config
            tree = ET.parse(config_file)
            root = tree.getroot()
            config = parse_xml_to_dict(root)

            # Merge with default config to ensure all keys exist
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value

            logger.info("Config loaded successfully from config.xml")
            return config
        else:
            # Create default XML config file
            root = create_default_xml(default_config)

            # Write pretty-printed XML
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write(pretty_print_xml(root))

            logger.info("Created default config.xml file.")
            return default_config

    except ET.ParseError as e:
        logger.error(f"Error parsing XML config file: {str(e)}")
        return default_config
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        return default_config


# Load configuration
CONFIG = load_config()

# Use environment variables for sensitive data if available
if os.environ.get("ANTHROPIC_API_KEY"):
    CONFIG["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY")
if os.environ.get("GOOGLE_API_KEY"):
    CONFIG["google_api_key"] = os.environ.get("GOOGLE_API_KEY")


# ----- Utility Functions for Model Management -----

def get_available_models_for_backend(backend: BackendType) -> List[str]:
    """Get available models for a specific backend"""
    if backend == BackendType.ANTHROPIC_CLAUDE:
        return CONFIG["available_claude_models"]
    elif backend == BackendType.GOOGLE_GEMINI:
        return CONFIG["available_gemini_models"]
    else:
        return []

def get_default_model_for_backend(backend: BackendType) -> str:
    """Get default model for a specific backend"""
    if backend == BackendType.ANTHROPIC_CLAUDE:
        return CONFIG["model"]
    elif backend == BackendType.GOOGLE_GEMINI:
        return CONFIG["gemini_model"]
    else:
        return CONFIG["model"]

def validate_model_for_backend(model: str, backend: BackendType) -> bool:
    """Validate if model is available for the specified backend"""
    available_models = get_available_models_for_backend(backend)
    return model in available_models


# ----- Security and Rate Limiting -----

class RateLimiter:
    def __init__(self, limit_per_minute=100):
        self.limit = limit_per_minute
        self.window_size = 60  # seconds
        self.requests = {}  # {ip: [timestamps]}
        
    def is_allowed(self, ip):
        current_time = time.time()
        if ip not in self.requests:
            self.requests[ip] = []
            
        # Clean up old requests
        self.requests[ip] = [t for t in self.requests[ip] if current_time - t < self.window_size]
        
        # Check if allowed
        if len(self.requests[ip]) < self.limit:
            self.requests[ip].append(current_time)
            return True
        return False


# Initialize rate limiter
rate_limiter = RateLimiter(CONFIG["rate_limit"])


# API key validation dependency
async def validate_api_key(x_api_key: str = Header(...)):
    """Validate the API key from the X-API-Key header"""
    if not CONFIG["api_keys"]:
        # If no API keys defined, skip validation (development mode)
        logger.warning("No API keys defined in config - running in development mode without API key validation")
        return
        
    if x_api_key not in CONFIG["api_keys"]:
        logger.warning(f"Invalid API key attempted: {x_api_key[:5]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key


# Rate limiting middleware with enhanced error handling
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    try:
        # Skip rate limiting for certain paths
        if request.url.path in ["/docs", "/redoc", "/openapi.json", "/health"]:
            return await call_next(request)
        
        if not rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        
        return await call_next(request)
        
    except Exception as e:
        logger.critical(f"CRITICAL: Rate limiting middleware failed for IP {client_ip}: {str(e)}", exc_info=True)
        # Continue processing even if rate limiting fails
        return await call_next(request)


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch any unhandled exceptions"""
    request_id = f"req_{int(time.time() * 1000)}_{os.urandom(4).hex()}"
    
    logger.critical(f"CRITICAL: Unhandled exception in request {request_id} for {request.url.path}: {str(exc)}", exc_info=True)
    
    # Return a safe error response
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url.path)
        }
    )


# ----- Bank Statement Processing -----

class BankStatementProcessor:
    def __init__(self, model=None, backend=BackendType.ANTHROPIC_CLAUDE):
        self.client = None
        self.model = model or get_default_model_for_backend(backend)
        self.backend = backend
        self.setup_client()

    def setup_client(self):
        """Initialize the appropriate client based on the backend"""
        if self.backend == BackendType.ANTHROPIC_CLAUDE:
            self.setup_anthropic_client()
        elif self.backend == BackendType.GOOGLE_GEMINI:
            self.setup_gemini_client()
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            raise ValueError(f"Unsupported backend: {self.backend}")

    def setup_anthropic_client(self):
        """Initialize Anthropic client with API key"""
        try:
            api_key = CONFIG["anthropic_api_key"]
            if not api_key:
                logger.error("Anthropic API key not configured")
                raise ValueError("Anthropic API key not configured. Please set API key in config.xml file or environment variable.")

            # Initialize client
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Anthropic client initialized with model: {self.model}")
            
        except ValueError:
            # Re-raise configuration errors
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize Anthropic client: {str(e)}")
            
    def setup_gemini_client(self):
        """Initialize Google Gemini client with API key"""
        try:
            api_key = CONFIG["google_api_key"]
            if not api_key:
                logger.error("Google API key not configured")
                raise ValueError("Google API key not configured. Please set API key in config.xml file or environment variable.")
                
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            # For Gemini, use the model passed in or default
            if not self.model or not validate_model_for_backend(self.model, BackendType.GOOGLE_GEMINI):
                self.model = CONFIG["gemini_model"]
            logger.info(f"Google Gemini client initialized with model: {self.model}")
            
        except ValueError:
            # Re-raise configuration errors
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini client: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to initialize Google Gemini client: {str(e)}")

    def validate_file(self, file_content, filename) -> bool:
        """Validate uploaded file"""
        if not file_content:
            return False, "No file content provided"

        # Check file size
        file_size_mb = len(file_content) / (1024 * 1024)
        if file_size_mb > CONFIG["max_file_size_mb"]:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({CONFIG['max_file_size_mb']}MB)"

        # Check file format
        file_extension = filename.split('.')[-1].lower()
        if file_extension not in CONFIG["supported_formats"]:
            return False, f"Unsupported file format. Please upload a PDF file."

        return True, "File is valid"

    def clean_json_response(self, raw_response: str) -> str:
        """Clean and normalize JSON response from LLM"""
        try:
            # 1. Remove markdown code blocks and common prefixes
            cleaned = re.sub(r'```json\s*', '', raw_response, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*```', '', cleaned)

            # Remove common LLM response prefixes
            cleaned = re.sub(r'^[^{]*(?=\{)', '', cleaned, flags=re.DOTALL)

            # 2. Find and extract the JSON object
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}')

            if json_start == -1 or json_end == -1 or json_end <= json_start:
                logger.warning("No valid JSON object found in response")
                return cleaned

            json_str = cleaned[json_start:json_end + 1]

            # 3. Clean up the extracted JSON
            json_str = json_str.strip()

            # Remove any invisible characters at the start
            while json_str and json_str[0] not in '{[':
                json_str = json_str[1:]

            # 4. Fix common JSON formatting issues
            # Fix trailing commas
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Fix unescaped newlines, tabs, etc.
            json_str = json_str.replace('\\n', '\\\\n')
            json_str = json_str.replace('\\t', '\\\\t')
            json_str = json_str.replace('\\r', '\\\\r')

            # Remove control characters and BOM
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            json_str = json_str.replace('\ufeff', '')  # BOM
            json_str = json_str.replace('\u200b', '')  # Zero-width space

            # 5. Ensure proper JSON structure
            if not json_str.startswith('{'):
                json_str = '{' + json_str
            if not json_str.endswith('}'):
                json_str = json_str + '}'

            return json_str

        except Exception as e:
            logger.warning(f"Error during JSON cleanup: {str(e)}")
            return raw_response

    def parse_json_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response with consistent cleanup"""
        # Always apply cleanup first
        cleaned_json = self.clean_json_response(raw_response)

        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed after cleanup: {str(e)}")
            logger.debug(f"Position {e.pos}: '{cleaned_json[e.pos] if e.pos < len(cleaned_json) else 'EOF'}'")
            logger.debug(f"Error location: line {getattr(e, 'lineno', 'unknown')}, column {getattr(e, 'colno', 'unknown')}")
            return None

    def encode_pdf_to_base64(self, pdf_bytes) -> str:
        """Convert PDF file to base64 string"""
        try:
            return base64.b64encode(pdf_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding PDF: {str(e)}")
            return None

    def calculate_estimated_cost(self, pdf_base64: str, prompt: str, output_tokens: int = None) -> float:
        """Calculate estimated cost for API usage based on backend"""
        if self.backend == BackendType.ANTHROPIC_CLAUDE:
            return self.calculate_anthropic_cost(pdf_base64, prompt, output_tokens)
        elif self.backend == BackendType.GOOGLE_GEMINI:
            return self.calculate_gemini_cost(pdf_base64, prompt, output_tokens)
        else:
            logger.warning(f"Unknown backend for cost calculation: {self.backend}")
            return 0.0
            
    def calculate_anthropic_cost(self, pdf_base64: str, prompt: str, output_tokens: int = None) -> float:
        """Calculate estimated cost for Claude API usage"""
        # Claude Sonnet 4 pricing (as of 2025)
        input_cost_per_1k_tokens = 0.003  # $0.003 per 1K input tokens
        output_cost_per_1k_tokens = 0.015  # $0.015 per 1K output tokens

        # Rough estimation: 1 character ≈ 0.25 tokens for text, images are more complex
        prompt_tokens = len(prompt) * 0.25

        # PDF tokens are harder to estimate, but roughly:
        # Base64 encoded PDF: every 4 chars = 3 bytes of original data
        # Anthropic processes images/PDFs more efficiently, rough estimate
        pdf_size_chars = len(pdf_base64)
        pdf_tokens = pdf_size_chars * 0.1  # Conservative estimate for document processing

        total_input_tokens = prompt_tokens + pdf_tokens

        # Estimate output tokens (if not provided from actual response)
        estimated_output_tokens = output_tokens or 500  # Conservative estimate

        # Calculate costs
        input_cost = (total_input_tokens / 1000) * input_cost_per_1k_tokens
        output_cost = (estimated_output_tokens / 1000) * output_cost_per_1k_tokens

        total_cost = input_cost + output_cost
        return total_cost
        
    def calculate_gemini_cost(self, pdf_base64: str, prompt: str, output_tokens: int = None) -> float:
        """Calculate estimated cost for Google Gemini API usage"""
        # Gemini 1.5 Pro pricing (as of 2025)
        input_cost_per_1k_tokens = 0.00125  # $0.00125 per 1K input tokens
        output_cost_per_1k_tokens = 0.00375  # $0.00375 per 1K output tokens

        # Rough estimation: 1 character ≈ 0.25 tokens for text, images are more complex
        prompt_tokens = len(prompt) * 0.25

        # PDF tokens are harder to estimate, but roughly:
        # For Gemini, we'll use a similar estimation as Claude
        pdf_size_chars = len(pdf_base64)
        pdf_tokens = pdf_size_chars * 0.1  # Conservative estimate for document processing

        total_input_tokens = prompt_tokens + pdf_tokens

        # Estimate output tokens (if not provided from actual response)
        estimated_output_tokens = output_tokens or 500  # Conservative estimate

        # Calculate costs
        input_cost = (total_input_tokens / 1000) * input_cost_per_1k_tokens
        output_cost = (estimated_output_tokens / 1000) * output_cost_per_1k_tokens

        total_cost = input_cost + output_cost
        return total_cost

    def extract_transactions(self, pdf_base64: str, custom_prompt: str = None) -> tuple[
        Optional[Dict[str, Any]], float, float, str, str]:
        """Extract financial summary from PDF using the selected backend"""
        if self.backend == BackendType.ANTHROPIC_CLAUDE:
            return self.extract_transactions_claude(pdf_base64, custom_prompt)
        elif self.backend == BackendType.GOOGLE_GEMINI:
            return self.extract_transactions_gemini(pdf_base64, custom_prompt)
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            raise ValueError(f"Unsupported backend: {self.backend}")
            
    def extract_transactions_claude(self, pdf_base64: str, custom_prompt: str = None) -> tuple[
        Optional[Dict[str, Any]], float, float, str, str]:
        """Extract financial summary from PDF using Claude API - Messages API for v0.6.0"""
        prompt = custom_prompt or CONFIG["extraction_prompt_claude"]

        start_time = datetime.now()

        try:
            logger.info(f"Starting Claude API call with model {self.model}")
            
            # Use the Messages API (supported in v0.6.0+)
            message = self.client.messages.create(
                model=self.model,
                max_tokens=8000,
                temperature=0.0,  # Set temperature to 0 for deterministic output
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_base64
                                }
                            }
                        ]
                    }
                ]
            )

            response_text = message.content[0].text
            logger.info("Claude API call completed successfully")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Calculate estimated cost using actual output token count
            actual_output_tokens = len(response_text) * 0.25  # Rough estimation
            estimated_cost = self.calculate_estimated_cost(pdf_base64, prompt, actual_output_tokens)

            # Always apply consistent JSON cleanup and parse
            result = self.parse_json_response(response_text)
            return result, processing_time, estimated_cost, prompt, response_text

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            estimated_cost = self.calculate_estimated_cost(pdf_base64, prompt)
            logger.error(f"Error calling Claude API: {str(e)}")
            return None, processing_time, estimated_cost, prompt, str(e)
            
    def extract_transactions_gemini(self, pdf_base64: str, custom_prompt: str = None) -> tuple[
        Optional[Dict[str, Any]], float, float, str, str]:
        """Extract financial summary from PDF using Google Gemini API"""
        prompt = custom_prompt or CONFIG["extraction_prompt_gemini"]

        start_time = datetime.now()

        try:
            logger.info(f"Starting Gemini API call with model {self.model}")
            
            # Create Gemini model instance
            model = genai.GenerativeModel(self.model)
            
            # Configure safety settings - use more permissive settings for document analysis
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Set max_output_tokens based on the specific Gemini model
            # All Gemini models now use 8000 tokens for consistency
            max_output_tokens = 8000
            
            # Generate content
            response = model.generate_content(
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {"mime_type": "application/pdf", "data": pdf_base64}}
                        ]
                    }
                ],
                safety_settings=safety_settings,
                generation_config={"temperature": 0.0, "max_output_tokens": max_output_tokens}
                # Note: Gemini API doesn't support custom timeout in generate_content
                # The 120s timeout will be handled by the underlying HTTP client
            )
            
            response_text = response.text
            logger.info("Gemini API call completed successfully")
            
            # Debug logging for Gemini responses
            logger.info(f"Raw response length: {len(response_text) if response_text else 0} characters")
            logger.info(f"Gemini response preview (first 200 chars): {response_text[:200] if response_text else 'EMPTY_RESPONSE'}")

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Calculate estimated cost using actual output token count
            actual_output_tokens = len(response_text) * 0.25  # Rough estimation
            estimated_cost = self.calculate_estimated_cost(pdf_base64, prompt, actual_output_tokens)

            # Always apply consistent JSON cleanup and parse
            result = self.parse_json_response(response_text)
            return result, processing_time, estimated_cost, prompt, response_text

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            estimated_cost = self.calculate_estimated_cost(pdf_base64, prompt)
            logger.error(f"Error calling Gemini API: {str(e)}")
            return None, processing_time, estimated_cost, prompt, str(e)

    def determine_loan_eligibility(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate loan eligibility and amount based on analysis results"""
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                return float(value) if value is not None else default
            except (ValueError, TypeError):
                return default

        # Helper function to safely convert to int
        def safe_int(value, default=0):
            try:
                return int(float(value)) if value is not None else default
            except (ValueError, TypeError):
                return default
        
        # Helper function to round to next multiple of 100
        def round_to_next_hundred(value):
            return math.ceil(value / 100) * 100

        # Extract key values
        salary_amount = safe_float(data.get('total_salary_received', 0))
        salary_count = safe_int(data.get('number_of_salary_credits', 0))
        avg_salary = safe_float(data.get('average_monthly_salary', 0))
        suspicious_count = len(data.get('suspicious_salary_transactions', []))

        # Calculate average salary if not provided by the API
        if avg_salary == 0 and salary_count > 0:
            avg_salary = salary_amount / salary_count

        # Determine loan eligibility and comments
        is_eligible = False
        max_loan_amount = 0.0
        eligibility_comments = ""

        # Check for fraud first
        if suspicious_count > 0:
            is_eligible = False
            max_loan_amount = 0.0
            eligibility_comments = "Not Eligible: Suspicious salary transactions detected"
        
        # Check salary credit count
        elif salary_count < 3:
            is_eligible = False
            max_loan_amount = 0.0
            eligibility_comments = f"Not Eligible: Insufficient salary credits (need 3, found {salary_count})"
        
        # Check if eligible
        else:
            is_eligible = True
            eligibility_comments = "Eligible"
            
            # Calculate loan amount based on salary bands
            loan_rate = 0
            
            # Define the salary bands and their rates
            salary_bands = [
                {"range": "₹25K - ₹30K", "min": 25000, "max": 30000, "rate": 30},
                {"range": "₹30K+ - ₹50K", "min": 30001, "max": 50000, "rate": 35},
                {"range": "₹50K+ - ₹51K", "min": 50001, "max": 51000, "rate": 50},
                {"range": "₹51K+ - ₹1L", "min": 51001, "max": 100000, "rate": 55},
            ]
            
            # Find which band the salary falls into
            for band in salary_bands:
                if band["min"] <= avg_salary <= band["max"]:
                    loan_rate = band["rate"]
                    break
                    
            # If salary is below the lowest band or above the highest band, handle accordingly
            if loan_rate == 0:
                if avg_salary < salary_bands[0]["min"]:
                    loan_rate = salary_bands[0]["rate"]
                else:
                    loan_rate = salary_bands[-1]["rate"]
                    
            # Calculate loan amount
            max_loan_amount = avg_salary * (loan_rate / 100)
            max_loan_amount = round_to_next_hundred(max_loan_amount)

        return {
            "loan_eligibility": is_eligible,
            "average_monthly_salary": avg_salary,
            "max_loan_eligibility": max_loan_amount,
            "loan_eligibility_comments": eligibility_comments
        }


# ----- API Endpoints -----

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "api": "Bank Statement Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/models")
async def get_available_models(api_key: str = Depends(validate_api_key)):
    """Get list of available models for all backends"""
    return {
        "claude_models": CONFIG["available_claude_models"],
        "gemini_models": CONFIG["available_gemini_models"],
        "default_claude_model": CONFIG["model"],
        "default_gemini_model": CONFIG["gemini_model"],
        # Legacy support
        "models": CONFIG["available_models"],
        "default_model": CONFIG["model"]
    }


@app.post("/analyze")
async def analyze_bank_statement(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    metadata: str = Form(...),
    custom_prompt: str = Form(None),
    model: str = Form(None),
    debug: bool = Form(False),
    backend: str = Form(BackendType.ANTHROPIC_CLAUDE.value),
    api_key: str = Depends(validate_api_key)
):
    """
    Analyze a bank statement PDF and return financial summary
    
    - **file**: PDF bank statement to analyze
    - **metadata**: JSON string with lead information (lead_id required)
    - **custom_prompt**: Optional custom prompt for the LLM
    - **model**: Optional model to use (must be compatible with selected backend)
    - **debug**: Set to true to include additional debug information
    - **backend**: AI backend to use (ANTHROPIC_CLAUDE or GOOGLE_GEMINI)
    """
    processing_id = f"proc_{int(time.time() * 1000)}_{os.urandom(6).hex()}"
    start_time = time.time()
    lead_id = "unknown"  # Initialize for error handling
    backend_enum = BackendType.ANTHROPIC_CLAUDE  # Default for error handling
    
    try:
        logger.info(f"Starting analysis request {processing_id}")
        
        # Parse metadata - wrapped in try-catch
        try:
            metadata_dict = json.loads(metadata)
            lead_metadata = LeadMetadata(**metadata_dict)
            lead_id = lead_metadata.lead_id  # Update for error handling
            logger.info(f"Processing request for lead_id={lead_id}, processing_id={processing_id}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in metadata parsing: {str(e)}, raw_metadata='{metadata[:200]}...'")
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Metadata validation error: {str(e)}, metadata_dict={locals().get('metadata_dict', 'N/A')}")
            raise HTTPException(status_code=400, detail=f"Invalid metadata format: {str(e)}")
        
        # Validate lead_id exists
        if not lead_metadata.lead_id:
            logger.error(f"Missing lead_id in request {processing_id}")
            raise HTTPException(status_code=400, detail="lead_id is required in metadata")
            
        # Validate backend - wrapped in try-catch
        try:
            backend_enum = BackendType(backend)
            logger.info(f"Using backend: {backend_enum.value}")
        except ValueError as e:
            logger.warning(f"Invalid backend '{backend}' for {processing_id}, defaulting to Claude. Error: {str(e)}")
            backend_enum = BackendType.ANTHROPIC_CLAUDE  # Default to Claude
        
        # NEW: Strict model validation based on backend
        if model:
            available_models = get_available_models_for_backend(backend_enum)
            if model not in available_models:
                logger.error(f"Invalid model '{model}' for backend {backend_enum.value} in request {processing_id}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Model '{model}' not available for backend {backend_enum.value}. Available models: {available_models}"
                )
            selected_model = model
            logger.info(f"Using requested model: {selected_model}")
        else:
            selected_model = get_default_model_for_backend(backend_enum)
            logger.info(f"Using default model for {backend_enum.value}: {selected_model}")
        
        # Read and validate file - wrapped in try-catch
        try:
            file_content = await file.read()
            logger.info(f"File read successfully: {len(file_content)} bytes, filename='{file.filename}'")
        except Exception as e:
            logger.error(f"File reading error for {processing_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
        
        # Validate file without creating a processor instance - wrapped in try-catch
        def validate_file_content(file_content, filename):
            """Validate uploaded file"""
            try:
                if not file_content:
                    return False, "No file content provided"

                # Check file size
                file_size_mb = len(file_content) / (1024 * 1024)
                if file_size_mb > CONFIG["max_file_size_mb"]:
                    return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({CONFIG['max_file_size_mb']}MB)"

                # Check file format
                file_extension = filename.split('.')[-1].lower()
                if file_extension not in CONFIG["supported_formats"]:
                    return False, f"Unsupported file format. Please upload a PDF file."

                return True, "File is valid"
            except Exception as e:
                logger.error(f"File validation error: {str(e)}")
                return False, f"File validation failed: {str(e)}"
        
        try:
            is_valid, validation_message = validate_file_content(file_content, file.filename)
            if not is_valid:
                logger.error(f"File validation failed for {processing_id}: {validation_message}")
                raise HTTPException(status_code=400, detail=validation_message)
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Unexpected error during file validation for {processing_id}: {str(e)}")
            raise HTTPException(status_code=400, detail=f"File validation error: {str(e)}")
            
        # Initialize processor with selected model and backend - wrapped in try-catch
        try:
            processor = BankStatementProcessor(selected_model, backend_enum)
            logger.info(f"Processor initialized successfully for {backend_enum.value} with model {selected_model}")
        except Exception as e:
            logger.error(f"Processor initialization failed for {processing_id}: {str(e)}")
            return create_error_response(
                lead_id, processing_id, start_time, 
                f"Failed to initialize {backend_enum.value} processor: {str(e)}", 
                backend_enum, debug
            )
        
        # Process PDF - wrapped in try-catch
        try:
            pdf_base64 = processor.encode_pdf_to_base64(file_content)
            if not pdf_base64:
                raise ValueError("Failed to encode PDF to base64")
            logger.info(f"PDF encoded successfully: {len(pdf_base64)} characters")
        except Exception as e:
            logger.error(f"PDF encoding failed for {processing_id}: {str(e)}")
            return create_error_response(
                lead_id, processing_id, start_time, 
                f"Failed to encode PDF file: {str(e)}", 
                backend_enum, debug
            )
            
        # Extract transactions - wrapped in try-catch
        try:
            result, processing_time, estimated_cost, prompt_used, raw_response = processor.extract_transactions(
                pdf_base64, custom_prompt
            )
            logger.info(f"Transaction extraction completed for {processing_id} in {processing_time:.2f}s")
        except Exception as e:
            logger.error(f"Transaction extraction failed for {processing_id}: {str(e)}")
            processing_time = time.time() - start_time
            return create_error_response(
                lead_id, processing_id, start_time, 
                f"LLM processing failed: {str(e)}", 
                backend_enum, debug, processing_time, 0.0
            )
        
        if not result:
            # LLM call failed - create detailed error response but don't raise HTTPException
            logger.warning(f"LLM extraction returned no result for {processing_id}")
            return create_error_response(
                lead_id, processing_id, start_time, 
                f"LLM API Error: {raw_response}", 
                backend_enum, debug, processing_time, estimated_cost
            )
            
        # Calculate loan eligibility - wrapped in try-catch
        try:
            loan_info = processor.determine_loan_eligibility(result)
            
            # Add new loan eligibility fields
            result.update(loan_info)
            logger.info(f"Loan eligibility calculated for {processing_id}: eligible={loan_info.get('loan_eligibility', False)}")
        except Exception as e:
            logger.error(f"Loan eligibility calculation failed for {processing_id}: {str(e)}")
            # Continue without loan eligibility rather than failing the entire request
            logger.warning(f"Continuing without loan eligibility for {processing_id}")
        
        # Create response - wrapped in try-catch
        try:
            # Ensure old fields are not included in response_data
            result_cleaned = {k: v for k, v in result.items() if k not in ['loan_amount', 'loan_rate', 'salary_band']}
            
            response_data = {
                "lead_id": lead_metadata.lead_id,
                "processing_id": processing_id,
                "request_id": -1,  # Placeholder for future database integration
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "status": "success",
                **result_cleaned
            }
            
            # FIXED: Only add debug fields if debug mode is enabled
            if debug:
                response_data.update({
                    "estimated_api_cost_usd": estimated_cost,
                    "backend_used": backend_enum.value,
                    "ai_model_used": selected_model
                })
            
            # Create a dynamic response class that only includes the fields we want
            if debug:
                # When debug is enabled, use the full model
                response = ProcessingResponse(**response_data)
                response_dict = response.dict(exclude_none=True)
            else:
                # When debug is disabled, create response without debug fields
                # Don't create ProcessingResponse object at all - return dict directly
                
                # Validate required fields are present (basic validation)
                required_fields = [
                    "lead_id", "processing_id", "timestamp", "processing_time_seconds", "status"
                ]
                for field in required_fields:
                    if field not in response_data:
                        raise ValueError(f"Missing required field: {field}")
                
                response_dict = response_data
            
            # Log processing details
            logger.info(f"Successfully processed bank statement for lead_id={lead_metadata.lead_id}, " 
                       f"backend={backend_enum.value}, model={selected_model}, "
                       f"processing_time={processing_time:.2f}s, cost=${estimated_cost:.4f}")
                       
            return response_dict
            
        except Exception as e:
            logger.error(f"Response creation failed for {processing_id}: {str(e)}")
            processing_time = time.time() - start_time
            return create_error_response(
                lead_id, processing_id, start_time, 
                f"Failed to create response: {str(e)}", 
                backend_enum, debug, processing_time, estimated_cost
            )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions (these are expected/handled errors)
        logger.warning(f"HTTP exception for {processing_id}: {e.status_code} - {e.detail}")
        raise
        
    except Exception as e:
        # Catch-all for any unexpected errors - NEVER let the process crash
        processing_time = time.time() - start_time
        logger.critical(f"CRITICAL: Unexpected error in analyze_bank_statement for {processing_id}: {str(e)}", exc_info=True)
        
        # Return a safe error response even if response creation might fail
        try:
            return create_error_response(
                lead_id, processing_id, start_time, 
                f"Unexpected system error: {str(e)}", 
                backend_enum, debug, processing_time, 0.0
            )
        except Exception as inner_e:
            # Last resort - return minimal response to prevent total crash
            logger.critical(f"CRITICAL: Error response creation failed for {processing_id}: {str(inner_e)}", exc_info=True)
            return {
                "lead_id": lead_id,
                "processing_id": processing_id,
                "request_id": -1,
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_time,
                "status": "critical_error",
                "error_message": f"Critical system error: {str(e)}",
                "total_salary_received": 0.0,
                "number_of_salary_credits": 0,
                "salary_transactions": [],
                "average_monthly_salary": 0.0,
                "suspicious_salary_transactions": [],
                "loan_eligibility": False,
                "max_loan_eligibility": 0.0,
                "loan_eligibility_comments": "Not Eligible: System error"
            }


def create_error_response(lead_id: str, processing_id: str, start_time: float, 
                         error_message: str, backend_enum: BackendType, debug: bool,
                         processing_time: float = None, estimated_cost: float = 0.0) -> dict:
    """Create a standardized error response - helper function to prevent code duplication"""
    try:
        if processing_time is None:
            processing_time = time.time() - start_time
            
        error_response_data = {
            "lead_id": lead_id,
            "processing_id": processing_id,
            "request_id": -1,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time,
            "status": "error",
            "error_message": error_message,
            "account_holder_name": None,
            "account_number": None,
            "total_salary_received": 0.0,
            "number_of_salary_credits": 0,
            "salary_transactions": [],
            "average_monthly_salary": 0.0,
            "suspicious_salary_transactions": [],
            "loan_eligibility": False,
            "max_loan_eligibility": 0.0,
            "loan_eligibility_comments": "Not Eligible: Processing error"
        }
        
        # FIXED: Only add debug information if requested
        if debug:
            error_response_data.update({
                "estimated_api_cost_usd": estimated_cost,
                "backend_used": backend_enum.value,
                "ai_model_used": "N/A"
            })
        
        # Return the dict directly when debug is False, or create ProcessingResponse when debug is True
        if debug:
            response = ProcessingResponse(**error_response_data)
            return response.dict(exclude_none=True)
        else:
            # Return dict directly without creating ProcessingResponse object
            # Remove None values manually
            return {k: v for k, v in error_response_data.items() if v is not None}
        
    except Exception as e:
        logger.critical(f"CRITICAL: create_error_response failed: {str(e)}", exc_info=True)
        # Return absolute minimal response as last resort
        minimal_response_data = {
            "lead_id": lead_id,
            "processing_id": processing_id,
            "request_id": -1,
            "timestamp": datetime.now().isoformat(),
            "processing_time_seconds": processing_time or 0.0,
            "status": "critical_error",
            "error_message": f"System error during error handling: {str(e)}",
            "total_salary_received": 0.0,
            "number_of_salary_credits": 0,
            "salary_transactions": [],
            "average_monthly_salary": 0.0,
            "suspicious_salary_transactions": [],
            "loan_eligibility": False,
            "max_loan_eligibility": 0.0,
            "loan_eligibility_comments": "Not Eligible: Critical error"
        }
        
        # Remove None values and return dict directly
        return {k: v for k, v in minimal_response_data.items() if v is not None}


@app.get("/status/{processing_id}")
async def get_processing_status(processing_id: str, api_key: str = Depends(validate_api_key)):
    """
    Get status of a processing job
    
    This is a placeholder for future implementation with proper job queue.
    Currently returns a stub response.
    """
    # In a real implementation, this would check a database or queue for the job status
    return {
        "processing_id": processing_id,
        "status": "completed",  # or "pending", "processing", "failed"
        "message": "Processing status would be retrieved from a database in production"
    }


# ----- Main Application Entry Point -----

if __name__ == "__main__":
    try:
        # Check if Anthropic API key is configured
        if not CONFIG["anthropic_api_key"]:
            logger.error("ANTHROPIC_API_KEY not set. Please add it to config.xml or set the environment variable.")
            print("Error: ANTHROPIC_API_KEY not set. Please add it to config.xml or set the environment variable.")
            exit(1)
        
        # Set log level based on environment
        log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        try:
            logging.getLogger().setLevel(log_level)
            logger.info(f"Log level set to: {log_level}")
        except Exception as e:
            logger.warning(f"Failed to set log level '{log_level}', using INFO: {str(e)}")
            logging.getLogger().setLevel(logging.INFO)
        
        # Add a default API key if none defined (for development)
        if not CONFIG["api_keys"] and os.environ.get("ENVIRONMENT", "").lower() != "production":
            try:
                default_key = "dev_" + os.urandom(16).hex()
                CONFIG["api_keys"].append(default_key)
                logger.warning(f"No API keys defined. Using generated development key: {default_key}")
                print(f"Development API Key: {default_key}")
            except Exception as e:
                logger.error(f"Failed to generate development API key: {str(e)}")
                print("Warning: No API keys configured and failed to generate development key")
        
        # Start the FastAPI server
        try:
            port = int(os.environ.get("PORT", 8000))
            logger.info(f"Starting Bank Statement Analyzer API on port {port}")
            print(f"Starting server on port {port}...")
            
            uvicorn.run("bsa_api:app", host="0.0.0.0", port=port, reload=False)
            
        except ValueError as e:
            logger.error(f"Invalid port configuration: {str(e)}")
            print(f"Error: Invalid port configuration - {str(e)}")
            exit(1)
        except Exception as e:
            logger.critical(f"Failed to start server: {str(e)}", exc_info=True)
            print(f"Critical Error: Failed to start server - {str(e)}")
            exit(1)
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        print("\nShutting down gracefully...")
        
    except Exception as e:
        logger.critical(f"CRITICAL: Application startup failed: {str(e)}", exc_info=True)
        print(f"Critical Error: Application failed to start - {str(e)}")
        exit(1)