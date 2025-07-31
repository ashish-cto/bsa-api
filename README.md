# Bank Statement Analyzer API

A comprehensive FastAPI-based service for analyzing Indian bank statements using AI-powered document processing. The system extracts financial insights including salary patterns, loan eligibility, and fraud detection from PDF bank statements.

## 🚀 Features

- **Multi-AI Backend Support**: Choose between Anthropic Claude and Google Gemini models
- **Financial Analysis**: Comprehensive salary transaction analysis and categorization
- **Fraud Detection**: Identifies suspicious salary transactions (UPI-based fake salaries)
- **Loan Eligibility**: Automated loan eligibility assessment with amount calculation
- **Secure Processing**: Bank-grade security with API key authentication and rate limiting
- **Scalable Architecture**: Built for production deployment on Azure App Service

## 📋 API Capabilities

### Core Analysis Features
- **Account Information Extraction**: Account holder name and account number
- **Salary Analysis**: Legitimate salary credits via NEFT/RTGS transactions
- **Transaction Categorization**: EMI payments, loan disbursals, UPI transactions
- **Fraud Alerts**: Detection of suspicious UPI-based salary claims
- **Loan Assessment**: Eligibility determination based on salary history and patterns

### Supported Transaction Types
- ✅ **Legitimate Salaries**: NEFT/RTGS transfers from employers
- ⚠️ **Suspicious Transactions**: UPI credits claiming to be salary payments
- 💰 **Loan Disbursals**: Credits from NBFCs and financial institutions
- 📊 **EMI Payments**: Recurring loan repayments and installments
- 💳 **UPI Transactions**: Digital payment credits and debits

## 🛠️ Technology Stack

- **Backend**: FastAPI with uvicorn server
- **AI Models**: Anthropic Claude 4, Google Gemini 1.5
- **Document Processing**: PDF parsing with base64 encoding
- **Configuration**: XML-based configuration management
- **Client Interface**: Streamlit web application
- **Deployment**: Azure App Service ready

## 📦 Installation

### Prerequisites
- Python 3.11+
- AI API Keys (Anthropic Claude and/or Google Gemini)

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bank-statement-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   - Set environment variables:
     ```bash
     export ANTHROPIC_API_KEY="your-anthropic-key"
     export GOOGLE_API_KEY="your-google-key"
     ```
   - Or update `config.xml` with your configuration

4. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 🔧 Configuration

### config.xml Structure
```xml
<?xml version="1.0" encoding="UTF-8"?>
<config>
  <!-- AI Provider Settings -->
  <anthropic_api_key></anthropic_api_key>
  <google_api_key></google_api_key>
  <model>claude-sonnet-4-20250514</model>
  <gemini_model>gemini-1.5-pro</gemini_model>
  
  <!-- API Configuration -->
  <max_file_size_mb>10</max_file_size_mb>
  <rate_limit>100</rate_limit>
  
  <!-- Authentication -->
  <api_keys_bsa>
    <key>your-api-key-here</key>
  </api_keys_bsa>
  
  <!-- Available Models -->
  <available_claude_models>
    <model>claude-sonnet-4-20250514</model>
    <model>claude-3-7-sonnet-20250219</model>
    <model>claude-3-5-sonnet-20241022</model>
  </available_claude_models>
  
  <available_gemini_models>
    <model>gemini-1.5-pro</model>
    <model>gemini-1.5-flash</model>
  </available_gemini_models>
</config>
```

### Environment Variables
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `GOOGLE_API_KEY`: Your Google AI API key
- `ENVIRONMENT`: Deployment environment (development/production)

## 📖 API Documentation

### Base URL
- **Local**: `http://localhost:8000`
- **Production**: `https://your-app.azurewebsites.net`

### Authentication
All API endpoints require authentication via the `X-API-Key` header:
```
X-API-Key: your-api-key
```

### Main Endpoints

#### POST /analyze
Analyze a bank statement PDF and return financial summary.

**Request:**
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Headers**: `X-API-Key: your-api-key`

**Form Data:**
```
file: <PDF file> (max 10MB)
metadata: {
  "lead_id": "LEAD_123456789",
  "customer_name": "John Doe",
  "application_id": "APP_001",
  "source": "Web Portal"
}
backend: "ANTHROPIC_CLAUDE" | "GOOGLE_GEMINI"
model: "claude-sonnet-4-20250514" | "gemini-1.5-pro"
```

**Response:**
```json
{
  "lead_id": "LEAD_123456789",
  "processing_id": "proc_1234567890_abcdef",
  "timestamp": "2025-01-31T20:42:43.128289",
  "processing_time_seconds": 8.62,
  "status": "success",
  
  "account_holder_name": "JOHN DOE",
  "account_number": "149901508829",
  
  "total_salary_received": 138366.0,
  "number_of_salary_credits": 3,
  "average_monthly_salary": 46122.0,
  
  "salary_transactions": [
    {
      "date": "30-04-2025",
      "amount": 46158.0,
      "description": "NEFT-APOLLO HOSPITALS ENTERPRISE LTD",
      "mode": "NEFT",
      "type": "Credit"
    },
    {
      "date": "28-03-2025",
      "amount": 46054.0,
      "description": "NEFT-APOLLO HOSPITALS ENTERPRISE LTD",
      "mode": "NEFT",
      "type": "Credit"
    },
    {
      "date": "28-02-2025",
      "amount": 46154.0,
      "description": "NEFT-APOLLO HOSPITALS ENTERPRISE LTD",
      "mode": "NEFT",
      "type": "Credit"
    }
  ],
  
  "suspicious_salary_transactions": [],
  
  "loan_eligibility": true,
  "max_loan_eligibility": 16200.0,
  "loan_eligibility_comments": "Eligible"
}
```

#### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-31T20:42:43.128289",
  "version": "1.0.0"
}
```

#### GET /models
Get available AI models for all backends.

**Response:**
```json
{
  "claude_models": ["claude-sonnet-4-20250514", "claude-3-7-sonnet-20250219"],
  "gemini_models": ["gemini-1.5-pro", "gemini-1.5-flash"],
  "default_claude_model": "claude-sonnet-4-20250514",
  "default_gemini_model": "gemini-1.5-pro"
}
```

## 🎯 Loan Eligibility Logic

### Eligibility Criteria
- **Minimum Salary Credits**: 3 legitimate salary transactions required
- **No Suspicious Activity**: Zero tolerance for fraudulent salary transactions
- **NEFT/RTGS Only**: Only bank transfers considered as legitimate salaries

### Loan Amount Calculation
Based on average monthly salary bands:

| Salary Range | Loan Rate | Example |
|-------------|-----------|---------|
| ₹25K - ₹30K | 30% | ₹28K → ₹8,400 |
| ₹30K+ - ₹50K | 35% | ₹40K → ₹14,000 |
| ₹50K+ - ₹51K | 50% | ₹50.5K → ₹25,300 |
| ₹51K+ - ₹1L | 55% | ₹75K → ₹41,300 |

*Amounts rounded up to nearest ₹100*

## 🔒 Security Features

- **API Key Authentication**: Secure endpoint access
- **Rate Limiting**: 100 requests per minute per IP
- **File Validation**: Size limits (10MB) and format restrictions (PDF only)
- **Environment Variable Security**: Sensitive credentials via environment variables
- **CORS Protection**: Configurable cross-origin request handling

## 🖥️ Client Interface

### Streamlit Web Application
A user-friendly web interface for testing and interacting with the API.

**Features:**
- AI backend selection (Claude/Gemini)
- Model selection per backend
- File upload with validation
- Real-time results display
- Export functionality (JSON/CSV)

**Running the Client:**
```bash
streamlit run streamlit_client.py
```

Access at: `http://localhost:8501`

## ☁️ Deployment

### Azure App Service (Recommended)
The application is optimized for Azure App Service deployment with uvicorn.

**Key Benefits:**
- Native Python FastAPI support
- Automatic uvicorn detection
- Built-in SSL and custom domains
- Integrated CI/CD with GitHub
- Environment variable management

**Deployment Steps:**
1. Push code to GitHub repository
2. Create Azure App Service (Python 3.11 runtime)
3. Configure GitHub deployment integration
4. Set environment variables for API keys
5. Deploy and test

### Environment Variables for Production
```
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
ENVIRONMENT=production
```

## 📊 Response Codes

- **200**: Successful analysis
- **400**: Invalid request (bad file, missing metadata, invalid model)
- **401**: Invalid API key
- **413**: File too large (>10MB)
- **429**: Rate limit exceeded
- **500**: Internal server error

## 🔍 Error Handling

The API provides detailed error responses for troubleshooting:

```json
{
  "lead_id": "LEAD_123456789",
  "processing_id": "proc_1234567890_abcdef",
  "status": "error",
  "error_message": "File size exceeds maximum allowed size (10MB)",
  "timestamp": "2025-01-31T20:42:43.128289"
}
```

## 📈 Monitoring

### Built-in Logging
- Comprehensive request/response logging
- Error tracking with stack traces
- Performance metrics (processing time)
- Security event logging

### Health Monitoring
- `/health` endpoint for uptime monitoring
- Processing time tracking
- Error rate monitoring

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Configure API keys in environment variables
5. Run tests and ensure all endpoints work

### Code Standards
- Follow PEP 8 style guidelines
- Comprehensive error handling
- Detailed logging for debugging
- Security-first approach

## 📄 License

This project is proprietary software. All rights reserved.

## 📞 Support

For technical support and questions:
- Review API documentation at `/docs` endpoint
- Check logs for detailed error information
- Ensure all API keys are properly configured

---

**Bank Statement Analyzer API v1.0.0** - Powered by AI for accurate financial document analysis.