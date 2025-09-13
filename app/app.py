import streamlit as st
import os
from dotenv import load_dotenv
from app.prompt_management import add_prompt_section, ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response
from app.modes.individual import render_individual_testing
from app.modes.chaining import render_prompt_chaining
from app.modes.combination import render_prompt_combination
from app.export import render_export_section

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="BainGan",
    page_icon="🍆",
    layout="wide"
)

# Initialize session state
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'prompt_names' not in st.session_state:
    st.session_state.prompt_names = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'chain_results' not in st.session_state:
    st.session_state.chain_results = []
if 'combination_results' not in st.session_state:
    st.session_state.combination_results = []
if 'slider_weights' not in st.session_state:
    st.session_state.slider_weights = {}
if 'last_selected_prompts' not in st.session_state:
    st.session_state.last_selected_prompts = []
if 'response_ratings' not in st.session_state:
    st.session_state.response_ratings = {}
if 'export_data' not in st.session_state:
    st.session_state.export_data = []

st.title("🔮 BainGan 🍆")
st.markdown("Test **system prompts**, create **prompt chains**, and **combine prompts** with AI assistance")

# Sidebar for API configuration
st.sidebar.header("🔧 API Configuration")

api_url = st.sidebar.text_input(
    "API Endpoint URL",
    placeholder="https://api.example.com/chat/rag",
    help="Enter the full URL of your chat API endpoint"
)

# Gemini API Key configuration
st.sidebar.subheader("🤖 Gemini API (for prompt combination)")
env_gemini_key = os.getenv('GEMINI_API_KEY')
if env_gemini_key:
    st.sidebar.success("✅ Gemini API key loaded from environment")
    gemini_api_key = env_gemini_key
else:
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Optional: Enter manually if not in environment"
    )

if gemini_api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=gemini_api_key)
        if not env_gemini_key:
            st.sidebar.success("✅ Gemini API configured manually")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini API error: {str(e)}")

# Headers configuration
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

# Single query input
query_text = st.sidebar.text_area(
    "Query (the actual user question)",
    placeholder="e.g. What is RAG?",
    height=100
)

# Request body template
st.sidebar.subheader("Request Body Template")
body_template = st.sidebar.text_area(
    "JSON Template (use {system_prompt} and {query} as placeholders)",
    value="""{
  "query": "{system_prompt}\\n\\nQuestion: {query}\\nAnswer:",
  "top_k": 5
}""",
    height=150,
    help="Use {system_prompt} for the system instructions and {query} for the user query"
)

# Response path for extracting the actual response
response_path = st.sidebar.text_input(
    "Response Text Path",
    value="response",
    help="JSON path to extract response text (e.g., 'response' or 'data.message')"
)

# Test mode selection
test_mode = st.sidebar.selectbox(
    "🎯 Test Mode",
    ["Individual Testing", "Prompt Chaining", "Prompt Combination"],
    help="Choose how to test your prompts"
)

st.markdown("---")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    add_prompt_section()

with col2:
    if test_mode == "Individual Testing":
        render_individual_testing(api_url, query_text, body_template, headers, response_path, call_api, suggest_prompt_from_response)
    elif test_mode == "Prompt Chaining":
        render_prompt_chaining(api_url, query_text, body_template, headers, response_path, call_api, suggest_prompt_from_response)
    elif test_mode == "Prompt Combination":
        render_prompt_combination(api_url, query_text, body_template, headers, response_path, call_api, suggest_prompt_from_response, gemini_api_key)

render_export_section(query_text)

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
""")

st.markdown("""
📦 **Requirements:**
```bash
pip install streamlit requests pandas openpyxl google-generativeai python-dotenv
GEMINI_API_KEY=your_gemini_api_key_here
""")