# streamlit_app_with_db.py
import streamlit as st
import os
from dotenv import load_dotenv
import sys
import uuid
import pandas as pd
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db_operations import initialize_database
from app.prompt_management import add_prompt_section, ensure_prompt_names
from app.api_utils import (
    call_api,
    suggest_prompt_from_response,
    format_request_template,
    show_system_prompt_preference_dialog,
    _apply_query_only_format,
    _apply_combined_format
)
from app.modes.individual import render_individual_testing
from app.modes.chaining import render_prompt_chaining
from app.modes.combination import render_prompt_combination
from app.export_with_db import render_export_section
from app.auth_with_db import render_auth_page

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="BainGan",
    page_icon="🍆",
    layout="wide"
)

# Initialize database
if 'db_initialized' not in st.session_state:
    success = initialize_database(password=os.getenv("DB_PASSWORD"))
    st.session_state.db_initialized = success
    if success:
        st.info("Database initialized successfully")
    else:
        st.error("Failed to initialize database")

# Initialize session state defaults with consistent columns
defaults = {
    "prompts": [],
    "prompt_names": [],
    "test_results": pd.DataFrame(columns=[
        'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
        'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited', 'step',
        'combination_strategy', 'combination_temperature'
    ]).astype({'rating': 'Int64'}),
    "chain_results": [],
    "combination_results": {},
    "slider_weights": {},
    "last_selected_prompts": [],
    "response_ratings": {},
    "export_data": pd.DataFrame(columns=[
        'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
        'status', 'status_code', 'timestamp', 'edited', 'step',
        'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark'
    ]).astype({'rating': 'Int64'}),
    "prompt_input_key_suffix": str(uuid.uuid4()),  # Unique key for prompt input widgets
    "body_template": """{
"query": "{system_prompt}\\n\\n{query}",
"top_k": 5
}""",  # Default template in session state
    "body_template_key": str(uuid.uuid4())  # Unique key for body template widget
}

# ✅ Ensure defaults are only set once
if "initialized_defaults" not in st.session_state:
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    st.session_state.initialized_defaults = True

# ✅ Restore last formatted template if available
if "persisted_body_template" in st.session_state:
    st.session_state.body_template = st.session_state.persisted_body_template

# Check if user is authenticated
if not render_auth_page():
    st.stop()  # Stop rendering the rest of the app if not authenticated

# Title and Logout section
col_title, col_logout = st.columns([6, 1])
with col_title:
    st.title("🔮 BainGan 🍆")
    st.markdown("Test **system prompts**, create **prompt chains**, and **combine prompts** with AI assistance")
with col_logout:
    if st.button("🚪 Logout"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Sidebar: API configuration
st.sidebar.header("🔧 API Configuration")

# 1st: API Endpoint URL
st.sidebar.text_input(
    "API Endpoint URL",
    key="api_url",
    placeholder="https://api.example.com/chat/rag",
    help="Enter the full URL of your chat API endpoint"
)

# 2nd: Request Body Template (collapsible)
st.sidebar.subheader("📋 Request Body Template")
# Ensure the current template is preserved between reruns
if "display_body_template" not in st.session_state:
    st.session_state.display_body_template = st.session_state.get("body_template", defaults["body_template"])

with st.sidebar.expander("Request Body Template", expanded=True):
    st.markdown("Paste your API request body from Swagger/Postman and click Submit to auto-format it.")

    # Show the dialog if needed
    dialog_shown = show_system_prompt_preference_dialog()

    if not dialog_shown:
        # Only show template editor if dialog is not active
        body_template = st.text_area(
            "JSON Template",
            value=st.session_state.display_body_template,
            height=150,
            help="Paste your API request body template here",
            key=f"body_template_{st.session_state.body_template_key}"
        )

        # Submit button for template formatting
        if st.button("Submit", key="submit_template"):
            if body_template.strip():
                with st.spinner("Formatting template with AI..."):
                    formatted_template, changes_made, needs_dialog = format_request_template(body_template)

                    if needs_dialog:
                        # Set flag to show dialog and store the template
                        st.session_state.show_prompt_preference_dialog = True
                        st.session_state.pending_template = body_template
                        st.rerun()
                    elif formatted_template != body_template:
                        # ✅ Save and persist formatted template
                        st.session_state.body_template = formatted_template.strip()
                        st.session_state.display_body_template = formatted_template.strip()
                        st.session_state.persisted_body_template = formatted_template.strip()
                        st.session_state.body_template_key = str(uuid.uuid4())
                        st.success(f"✅ Template auto-formatted successfully! **Changes made:** {changes_made}")
                        st.rerun()
                    else:
                        st.info(f"✅ {changes_made}")
            else:
                st.error("Please enter a request body template first.")

    if 'format_message' in st.session_state:
        st.success(st.session_state.format_message)
        del st.session_state.format_message

# 3rd: Query input
query_text = st.sidebar.text_area(
    "Query (the actual user question)",
    placeholder="e.g. What is RAG?",
    height=100
)

# 4th: Test mode selection
test_mode = st.sidebar.selectbox(
    "🎯 Test Mode",
    ["Individual Testing", "Prompt Chaining", "Prompt Combination"],
    help="Choose how to test your prompts"
)

# 5th: Headers (Authentication Type)
st.sidebar.subheader("Headers")
auth_type = st.sidebar.selectbox("Authentication Type", ["None", "Bearer Token", "API Key", "Custom"])

headers = {}
if auth_type == "Bearer Token":
    token = st.sidebar.text_input("Bearer Token", type="password")
    if token:
        headers["Authorization"] = f"Bearer {token}"
elif auth_type == "API Key":
    api_key = st.sidebar.text_input("API Key", type="password")
    key_header = st.sidebar.text_input("Key Header Name", value="X-API-Key")
    if api_key:
        headers[key_header] = api_key
elif auth_type == "Custom":
    num_headers = st.sidebar.number_input("Number of custom headers", min_value=1, max_value=5, value=1)
    for i in range(num_headers):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            key = st.text_input(f"Header {i+1} Key", key=f"header_key_{i}")
        with col2:
            value = st.text_input(f"Header {i+1} Value", key=f"header_value_{i}")
        if key and value:
            headers[key] = value

headers["Content-Type"] = "application/json"

# 6th: Response path
response_path = st.sidebar.text_input(
    "Response Text Path",
    value="response",
    help="JSON path to extract response text (e.g., 'response' or 'data.message')"
)

# 7th: Gemini API configuration
st.sidebar.subheader("🤖 Gemini API (for prompt combination)")
env_gemini_key = os.getenv('GEMINI_API_KEY')
if env_gemini_key:
    st.sidebar.success("✅ Gemini API key loaded from environment")
    st.session_state.gemini_api_key = env_gemini_key
else:
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Optional: Enter manually if not in environment",
        key="gemini_api_key"
    )
    if gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            st.sidebar.success("✅ Gemini API configured")
        except Exception as e:
            st.sidebar.error(f"❌ Gemini API error: {str(e)}")

st.markdown("---")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    add_prompt_section()

with col2:
    api_url = st.session_state.api_url
    user_name = st.session_state.get("user_name", "Unknown")  # Get user_name from session state
    if test_mode == "Individual Testing":
        render_individual_testing(api_url, query_text, st.session_state.body_template, headers, response_path, call_api, suggest_prompt_from_response, user_name=user_name)
    elif test_mode == "Prompt Chaining":
        render_prompt_chaining(api_url, query_text, st.session_state.body_template, headers, response_path, call_api, suggest_prompt_from_response, user_name=user_name)
    elif test_mode == "Prompt Combination":
        render_prompt_combination(api_url, query_text, st.session_state.body_template, headers, response_path, call_api, suggest_prompt_from_response, st.session_state.get("gemini_api_key", ""), user_name=user_name)

render_export_section(query_text)

# Footer info
st.markdown("---")
st.markdown("""
💡 **Enhanced Features:**
- **✏️ Prompt Management:** Add, edit, and name prompts in all test modes  
- **🧪 Individual Testing:** Test multiple system prompts independently with editable responses  
- **🔗 Prompt Chaining:** Chain prompts where each step uses the previous output  
- **🤝 Prompt Combination:** Use AI to intelligently combine multiple prompts with auto-adjusting weights  
- **🎚️ Slider Strategy:** Custom influence weights for prompt combination, auto-adjusted to sum to 100%  
- **🌡️ Temperature Control:** 0-100% slider to control AI creativity for prompt combination  
- **🔮 Smart Suggestions:** Generate prompt suggestions with options to save, save and run, or edit  
- **⭐ Response Rating:** Rate all responses (0-10, stored as percentage in export)  
- **📊 Comprehensive Export:** All results including individual, chain, and combination data with ratings and remarks  
- **💾 Response Editing:** Edit and save responses, with reverse prompt engineering  
- **🔄 Auto-Format Templates:** Paste API templates from Swagger/Postman and auto-format with AI  
- **🗄️ Database Integration:** Persistent storage of user data and export results in MySQL
""")