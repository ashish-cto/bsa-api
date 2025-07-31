import streamlit as st
import requests
import json
import os
import pandas as pd
from datetime import datetime
import time
import tempfile

# Page configuration
st.set_page_config(
    page_title="Bank Statement Analyzer Client",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration in sidebar
with st.sidebar:
    st.title("🏦 Bank Statement Analyzer")
    st.markdown("---")

    # API Configuration
    st.subheader("API Settings")
    api_url = st.text_input("API URL", value="http://localhost:8000")
    # Pre-populate with the specified API key
    api_key = st.text_input("API Key", value="83d8f6f2-a7c8-4cf6-9b9c-453776444896", type="password")

    # Backend selection with radio buttons
    st.subheader("AI Provider")
    backend_display = st.radio(
        "Select AI Provider:",
        options=["Anthropic Claude", "Google Gemini"],
        index=0,  # Claude selected by default
        help="Select which AI provider to use for analysis"
    )

    # Convert display names to backend enum values
    backend_mapping = {
        "Anthropic Claude": "ANTHROPIC_CLAUDE",
        "Google Gemini": "GOOGLE_GEMINI"
    }
    backend = backend_mapping[backend_display]

    # Model selection based on backend
    st.subheader("Model Selection")

    # Define available models for each backend
    claude_models = [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022"
    ]

    gemini_models = [
        "gemini-1.5-pro",
        "gemini-1.5-flash"
        # "gemini-2.5-flash" removed due to performance/stability issues
    ]

    # Show model dropdown based on selected backend
    if backend == "ANTHROPIC_CLAUDE":
        available_models = claude_models
        default_model = claude_models[0]  # claude-sonnet-4-20250514
    else:
        available_models = gemini_models
        default_model = "gemini-1.5-pro"  # Specific default for Gemini (not first in list)

    # Persist model selection across backend changes
    model_key = f"selected_model_{backend}"
    if model_key not in st.session_state:
        st.session_state[model_key] = default_model

    # Ensure the stored model is still valid for current backend
    if st.session_state[model_key] not in available_models:
        st.session_state[model_key] = default_model

    selected_model = st.selectbox(
        f"Select {backend_display} Model:",
        options=available_models,
        index=available_models.index(st.session_state[model_key]),
        help=f"Choose which {backend_display} model to use for analysis"
    )

    # Update session state
    st.session_state[model_key] = selected_model

    # Debug mode toggle
    debug_mode = st.checkbox("Debug Mode", value=False,
                             help="Enable to include additional debug information and estimated API cost")

    # Auto-save settings to session state (no button needed)
    st.session_state.api_url = api_url
    st.session_state.api_key = api_key
    st.session_state.backend = backend
    st.session_state.selected_model = selected_model
    st.session_state.debug_mode = debug_mode

    st.markdown("---")
    st.info("Upload a bank statement PDF and get a comprehensive financial analysis.")
    st.markdown("---")

    # Display connection status
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=3)
        if response.status_code == 200:
            st.success("✅ Connected to API")
        else:
            st.error(f"❌ API returned status code: {response.status_code}")
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API")


# Main function to analyze bank statement
def analyze_bank_statement(file, lead_id, additional_metadata=None):
    """Send bank statement to API for analysis"""
    # Prepare API request
    url = f"{st.session_state.api_url}/analyze"

    headers = {
        "X-API-Key": st.session_state.api_key
    }

    # Prepare metadata
    metadata = {
        "lead_id": lead_id
    }

    # Add any additional metadata
    if additional_metadata and isinstance(additional_metadata, dict):
        metadata.update(additional_metadata)

    # Initialize request_details before the try block
    request_details = {
        'url': url,
        'method': 'POST',
        'headers': headers,
        'metadata': json.dumps(metadata, indent=2),
        'file_details': {
            'filename': getattr(file, 'name', 'unknown'),
            'content_type': 'application/pdf'
        }
    }

    try:
        # Prepare the multipart form data
        # Check if the file is a BufferedReader (opened with open()) or a Streamlit UploadedFile
        if hasattr(file, 'getvalue'):
            # It's a Streamlit UploadedFile
            file_size_kb = len(file.getvalue()) / 1024
            files = {
                'file': (file.name, file, 'application/pdf')
            }
            request_details['file_details']['size_kb'] = round(file_size_kb, 2)
        else:
            # It's a BufferedReader from open()
            file.seek(0, 2)  # Go to the end of the file
            file_size = file.tell()  # Get current position (file size)
            file.seek(0)  # Go back to the beginning
            file_size_kb = file_size / 1024

            # For BufferedReader, we need the filename
            filename = getattr(file, 'name', 'unknown').split('/')[-1].split('\\')[-1]

            files = {
                'file': (filename, file, 'application/pdf')
            }
            request_details['file_details']['size_kb'] = round(file_size_kb, 2)
            request_details['file_details']['filename'] = filename

        data = {
            'metadata': json.dumps(metadata),
            'debug': str(st.session_state.debug_mode).lower(),  # Add debug parameter
            'backend': st.session_state.backend,  # Add backend parameter
            'model': st.session_state.selected_model  # Add model parameter
        }

        # Make the API request
        with st.spinner("Analyzing bank statement... This may take a minute."):
            response = requests.post(url, headers=headers, files=files, data=data)

        # Check for errors
        if response.status_code != 200:
            st.error(f"Error: API returned status code {response.status_code}")
            st.code(response.text)
            return None, request_details

        # Parse and return the JSON response
        result = response.json()
        return result, request_details

    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {str(e)}")
        return None, request_details
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return None, request_details


# Display financial summary
def display_financial_summary(data):
    """Display financial summary in a clean, organized format"""
    if not data:
        st.error("❌ No data to display")
        return

    # Skip display if this is an error response
    if data.get("status") == "error":
        return

    st.subheader("📊 Financial Summary")

    # Extract key values with safe defaults and new field names
    account_holder_name = data.get('account_holder_name', '') or 'Not Available'
    account_number = data.get('account_number', '') or 'Not Available'
    salary_amount = data.get('total_salary_received', 0) or 0
    salary_count = data.get('number_of_salary_credits', 0) or 0
    salary_transactions = data.get('salary_transactions', []) or []
    suspicious_transactions = data.get('suspicious_salary_transactions', []) or []
    avg_salary = data.get('average_monthly_salary', 0) or 0

    # Display account information
    st.markdown("#### 🏦 Account Information")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Account Holder**\n{account_holder_name}")
    with col2:
        st.info(f"**Account Number**\n{account_number}")

    st.markdown("---")

    # Create two columns for better organization
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Income")
        # Salary Information
        st.metric(
            "Total Legitimate Salary Received",
            f"₹{salary_amount:,.2f}",
            delta=f"{salary_count} legitimate credits"
        )

        # Average Salary
        st.metric(
            "Average Monthly Salary",
            f"₹{avg_salary:,.2f}",
            delta=f"Based on {salary_count} transactions"
        )

    with col2:
        st.markdown("#### 📈 Account Status")
        # Salary Credit Count
        st.metric(
            "Legitimate Salary Credits",
            salary_count,
            delta="Total transactions"
        )

        # Suspicious Alerts Count
        suspicious_count = len(suspicious_transactions)
        st.metric(
            "Suspicious Salary Alerts",
            suspicious_count,
            delta="Suspicious transactions" if suspicious_count > 0 else "Clean record"
        )

    # Summary Statistics
    st.markdown("---")
    st.subheader("📋 Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Number of Salary Credits**\n{salary_count} legitimate salary transactions")

    with col2:
        if salary_count >= 3:
            st.success(f"**Salary History**\nSufficient for loan eligibility")
        else:
            st.warning(f"**Salary History**\nNeed {3 - salary_count} more for eligibility")

    with col3:
        if suspicious_count == 0:
            st.success(f"**Fraud Detection**\nNo suspicious transactions")
        else:
            st.error(f"**Fraud Detection**\n{suspicious_count} suspicious transactions found")


# Display loan eligibility
def display_loan_eligibility(data):
    """Display loan eligibility information"""
    if not data:
        return

    st.markdown("---")
    st.subheader("🏦 Loan Eligibility")

    # Extract loan eligibility information with safe defaults and new field names
    is_eligible = data.get('loan_eligibility', False) or False
    salary_count = data.get('number_of_salary_credits', 0) or 0
    avg_salary = data.get('average_monthly_salary', 0) or 0
    max_loan_amount = data.get('max_loan_eligibility', 0) or 0
    eligibility_comments = data.get('loan_eligibility_comments', '') or ''

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Loan Eligibility Status",
            "Eligible" if is_eligible else "Not Eligible",
            delta="Based on salary history"
        )

        if not is_eligible:
            st.warning(f"Need at least 3 clear salary credits. Currently have {salary_count}.")
        else:
            st.success("✅ Sufficient salary history to qualify for loan consideration.")

    with col2:
        st.metric(
            "Average Monthly Salary",
            f"₹{avg_salary:,.2f}",
            delta=f"Based on {salary_count} transactions"
        )

    # Display loan amount tiers if eligible
    if is_eligible and avg_salary > 0:
        st.markdown("### 💸 Loan Eligibility")

        # Display eligibility status with comments
        if eligibility_comments:
            st.success(f"✅ {eligibility_comments}")

        # Display the calculated loan amount
        st.markdown(f"""
        <div style="margin-top:15px; background-color:#F0F8FF; padding:15px; border-radius:5px; border-left:4px solid #4285F4;">
            <h4 style="margin:0; color:#333;">Maximum Loan Eligibility</h4>
            <div style="font-size:1.5em; font-weight:bold; margin:10px 0;">₹{max_loan_amount:,.2f}</div>
            <div style="font-size:0.85em; color:#666;">
            Based on your average monthly salary of ₹{avg_salary:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif not is_eligible:
        st.markdown("### 💸 Loan Eligibility")

        # Display rejection reason
        if eligibility_comments:
            st.error(f"❌ {eligibility_comments}")
        else:
            st.error("❌ Not eligible for loan")

        # Show what's needed for eligibility
        if salary_count < 3:
            st.info(
                f"💡 **Next Steps**: You need {3 - salary_count} more clear salary transactions to become eligible for a loan.")

        st.markdown(f"""
        <div style="margin-top:15px; background-color:#FFF5F5; padding:15px; border-radius:5px; border-left:4px solid #E53E3E;">
            <h4 style="margin:0; color:#333;">Maximum Loan Eligibility</h4>
            <div style="font-size:1.5em; font-weight:bold; margin:10px 0;">₹0.00</div>
            <div style="font-size:0.85em; color:#666;">
            Not eligible for loan at this time
            </div>
        </div>
        """, unsafe_allow_html=True)


# Display detailed salary transactions
def display_salary_transactions(data):
    """Display detailed salary transactions"""
    salary_transactions = data.get('salary_transactions', []) or []

    if salary_transactions:
        st.markdown("---")
        st.subheader("💼 Legitimate Salary Transactions")

        # Convert to DataFrame for display
        salary_df = pd.DataFrame(salary_transactions)
        if not salary_df.empty:
            # Format amount column for better display
            salary_df['amount'] = salary_df['amount'].apply(lambda x: f"₹{x:,.2f}" if x is not None else "₹0.00")
            st.dataframe(
                salary_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Date",
                    "amount": "Amount",
                    "description": "Description",
                    "mode": "Transfer Mode",
                    "type": "Transaction Type"
                }
            )


# Display suspicious salary alerts
def display_fraudulent_alerts(data):
    """Display suspicious salary alerts"""
    suspicious_transactions = data.get('suspicious_salary_transactions', []) or []

    if suspicious_transactions:
        st.markdown("---")
        st.subheader("⚠️ Suspicious Salary Transaction Alerts")
        st.error(
            "The following transactions claim to be salary payments but are flagged as suspicious:")

        # Convert to DataFrame for display
        suspicious_df = pd.DataFrame(suspicious_transactions)
        if not suspicious_df.empty:
            # Format amount column for better display
            suspicious_df['amount'] = suspicious_df['amount'].apply(
                lambda x: f"₹{x:,.2f}" if x is not None else "₹0.00")
            st.dataframe(
                suspicious_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "date": "Date",
                    "amount": "Amount",
                    "description": "Description",
                    "reason": "Reason for Alert",
                    "type": "Transaction Type"
                }
            )

            st.warning(
                "🚨 **Security Tip**: Legitimate salary payments are typically made via NEFT or RTGS by employers, never through UPI. Please verify these transactions with your employer.")


# Display processing details
def display_processing_details(data, upload_time, file_size_kb):
    """Display processing details"""
    st.markdown("---")
    processing_time = data.get('processing_time_seconds', 0) or 0

    # Basic info in a 3-column layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"<small>**Upload Time:** {upload_time or 'Unknown'}</small>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<small>**File Size:** {file_size_kb:.1f} KB</small>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<small>**Processing Time:** {processing_time:.1f} seconds</small>", unsafe_allow_html=True)

    # Additional debug info if available - only show if not None
    backend_used = data.get("backend_used")
    if backend_used is not None and backend_used != "":
        st.markdown(f"<small>**AI Backend:** {backend_used}</small>", unsafe_allow_html=True)

    # Show model used if available
    ai_model_used = data.get("ai_model_used")
    if ai_model_used is not None and ai_model_used != "":
        st.markdown(f"<small>**Model Used:** {ai_model_used}</small>", unsafe_allow_html=True)

    # Only show estimated cost if debug mode is enabled and the cost is in the response and not None
    estimated_cost_usd = data.get('estimated_api_cost_usd')
    estimated_cost = data.get('estimated_api_cost')

    if estimated_cost_usd is not None:
        st.markdown(f"<small>**Estimated API Cost:** ${estimated_cost_usd:.4f} USD</small>", unsafe_allow_html=True)
    elif estimated_cost is not None:
        st.markdown(f"<small>**Estimated API Cost:** ${estimated_cost:.4f} USD</small>", unsafe_allow_html=True)


# Display debug information
def display_debug_info(data, request_details=None):
    """Display debug information"""
    st.markdown("---")
    with st.expander("🔍 Debug Information"):
        # Show request details first
        if request_details:
            st.subheader("📤 Request Details")

            # URL and method
            st.markdown(f"**URL**: `{request_details.get('url', 'N/A')}`")
            st.markdown(f"**Method**: `{request_details.get('method', 'N/A')}`")

            # Headers (with API key masked)
            st.markdown("**Headers**:")
            headers = request_details.get('headers', {})
            masked_headers = headers.copy()
            if 'X-API-Key' in masked_headers:
                masked_headers['X-API-Key'] = masked_headers['X-API-Key'][:5] + '...' if masked_headers[
                    'X-API-Key'] else 'None'
            st.code(json.dumps(masked_headers, indent=2), language="json")

            # Metadata
            st.markdown("**Metadata**:")
            st.code(request_details.get('metadata', '{}'), language="json")

            # File details
            st.markdown("**File Details**:")
            file_details = request_details.get('file_details', {})
            st.code(json.dumps(file_details, indent=2), language="json")

            st.markdown("---")

        # Show response
        st.subheader("📥 API Response")
        st.json(data)


# Main app logic
def main():
    # App header
    st.title("🏦 Bank Statement Analyzer Client")
    st.markdown("Upload your PDF bank statement and get a comprehensive financial summary using Claude AI")

    # File upload section
    st.header("📄 Upload Bank Statement")

    # Use columns for a better layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("**📁 Upload PDF Bank Statement (Max: 10MB)**")
        st.info("ℹ️ **File Size Limit**: 10MB maximum. Larger files will be rejected.")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Files larger than 10MB will be rejected",
            label_visibility="collapsed"
        )

        # Check file size immediately after upload
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            if file_size_mb > 10:
                st.error(
                    f"❌ **File too large!** Your file is {file_size_mb:.1f}MB. Please upload a file smaller than 10MB.")
                uploaded_file = None  # Reset the file to prevent processing

    with col2:
        # Lead information
        st.subheader("Lead Information")
        lead_id = st.text_input("Lead ID", value=f"LEAD_{int(time.time())}")

        # Optional metadata fields
        with st.expander("Additional Metadata (Optional)"):
            customer_name = st.text_input("Customer Name")
            application_id = st.text_input("Application ID")
            source = st.text_input("Source", value="Streamlit Client")

    # Process button
    if uploaded_file is not None:
        # Display file info
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.info(f"📁 File: {uploaded_file.name} ({file_size_kb:.1f}KB)")

        # Create additional metadata
        additional_metadata = {
            "source": source or "Streamlit Client"
        }

        if customer_name:
            additional_metadata["customer_name"] = customer_name

        if application_id:
            additional_metadata["application_id"] = application_id

        # Process button
        if st.button("🚀 Analyze Bank Statement", type="primary"):
            # Record upload time
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Process the temporary file
                with open(tmp_file_path, 'rb') as f:
                    # Analyze bank statement
                    result, request_details = analyze_bank_statement(f, lead_id, additional_metadata)

                # Clean up the temporary file
                os.unlink(tmp_file_path)

                if result:
                    # Check if this is an error response
                    if result.get("status") == "error":
                        st.error("❌ Bank statement analysis failed")

                        # Show the detailed error message
                        error_msg = result.get("error_message", "Unknown error occurred")
                        st.error(f"**Error Details:** {error_msg}")

                        # Add helpful information based on error type
                        if "credit balance" in error_msg.lower() or "billing" in error_msg.lower():
                            st.warning(
                                "💳 **Action Required:** This appears to be an API billing issue. Please check your API credits/billing.")
                        elif "api key" in error_msg.lower():
                            st.warning(
                                "🔑 **Action Required:** This appears to be an API key issue. Please verify your API configuration.")
                        else:
                            st.info("🔍 **Next Steps:** Check the debug information below for more details.")
                    else:
                        st.success("✅ Bank statement analyzed successfully!")

                    # Add result to session state for persistence
                    st.session_state.analysis_result = result
                    st.session_state.upload_time = upload_time
                    st.session_state.file_size_kb = file_size_kb
                    st.session_state.request_details = request_details

                    # Separator before analysis display
                    st.markdown("---")

                    # Display sections - but only if status is success
                    if result.get("status") == "success":
                        display_financial_summary(result)
                        display_loan_eligibility(result)
                        display_salary_transactions(result)
                        display_fraudulent_alerts(result)

                    # Always show processing details and debug info (regardless of success/error)
                    display_processing_details(result, upload_time, file_size_kb)
                    display_debug_info(result, request_details)

                    # Add download options
                    st.markdown("---")
                    st.subheader("📥 Download Summary")

                    col1, col2 = st.columns(2)

                    with col1:
                        # JSON download
                        json_data = json.dumps(result, indent=2)
                        st.download_button(
                            label="Download as JSON",
                            data=json_data,
                            file_name=f"financial_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

                    with col2:
                        # CSV download (convert to single row)
                        df = pd.DataFrame([result])
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv_data,
                            file_name=f"financial_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ Failed to extract financial summary")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                # Add detailed error information for debugging
                st.error(f"❌ Error details: {type(e).__name__}: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

                # Clean up the temporary file in case of error
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

    # Display previous results if available
    elif 'analysis_result' in st.session_state:
        st.info("Showing previous analysis results. Upload a new statement to analyze.")

        # Display all sections
        display_financial_summary(st.session_state.analysis_result)
        display_loan_eligibility(st.session_state.analysis_result)
        display_salary_transactions(st.session_state.analysis_result)
        display_fraudulent_alerts(st.session_state.analysis_result)
        display_processing_details(
            st.session_state.analysis_result,
            st.session_state.upload_time,
            st.session_state.file_size_kb
        )
        display_debug_info(st.session_state.analysis_result,
                           st.session_state.get('request_details', None))

    # Footer
    st.markdown("---")
    st.caption("Bank Statement Analyzer Client © 2025")


if __name__ == "__main__":
    main()