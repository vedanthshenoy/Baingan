import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import io

# Set page config
st.set_page_config(
    page_title="Chat API Tester",
    page_icon="📮",
    layout="wide"
)

# Initialize session state
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

st.title("📮 Chat API Tester")
st.markdown("Test multiple **system prompts** against your chat API with a single query and export results")

# Sidebar for API configuration
st.sidebar.header("🔧 API Configuration")

# API URL input
api_url = st.sidebar.text_input(
    "API Endpoint URL",
    placeholder="https://api.example.com/chat/rag",
    help="Enter the full URL of your chat API endpoint"
)

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

# Content-Type
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

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("✍️ System Prompt Manager")
    
    # Add new system prompt
    st.subheader("Add New System Prompt")
    new_prompt = st.text_area("Enter your system prompt:", height=100, key="new_prompt_input")
    
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add System Prompt", type="primary"):
            if new_prompt.strip():
                st.session_state.prompts.append(new_prompt.strip())
                st.success(f"Added system prompt #{len(st.session_state.prompts)}")
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear All Prompts"):
            st.session_state.prompts = []
            st.session_state.test_results = []
            st.success("Cleared all prompts and results")
            st.rerun()
    
    # Display current prompts
    if st.session_state.prompts:
        st.subheader(f"📝 Current System Prompts ({len(st.session_state.prompts)})")
        for i, prompt in enumerate(st.session_state.prompts):
            with st.expander(f"Prompt {i+1}: {prompt[:50]}..."):
                st.text(prompt)
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.prompts.pop(i)
                    if i < len(st.session_state.test_results):
                        st.session_state.test_results.pop(i)
                    st.rerun()

with col2:
    st.header("🧪 Testing & Results")
    
    if st.button("🚀 Test All Prompts", type="primary", disabled=not (api_url and st.session_state.prompts and query_text)):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.prompts:
            st.error("Please add at least one system prompt")
        elif not query_text:
            st.error("Please enter a query")
        else:
            st.session_state.test_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, system_prompt in enumerate(st.session_state.prompts):
                status_text.text(f"Testing prompt {i+1}/{len(st.session_state.prompts)}...")
                try:
                    safe_system = system_prompt.replace("\n", "\\n").replace("\"", "\\\"")
                    safe_query = query_text.replace("\n", "\\n").replace("\"", "\\\"")

                    body = body_template.replace("{system_prompt}", safe_system).replace("{query}", safe_query)
                    body_json = json.loads(body)

                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=body_json,
                        timeout=30
                    )

                    if response.status_code == 200:
                        response_data = response.json()
                        response_text = response_data
                        for key in response_path.split('.'):
                            if key in response_text:
                                response_text = response_text[key]
                            else:
                                response_text = str(response_data)
                                break
                        result = {
                            'system_prompt': system_prompt,
                            'query': query_text,
                            'response': str(response_text),
                            'status': 'Success',
                            'status_code': response.status_code,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    else:
                        result = {
                            'system_prompt': system_prompt,
                            'query': query_text,
                            'response': f"Error: {response.text}",
                            'status': 'Error',
                            'status_code': response.status_code,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                except Exception as e:
                    result = {
                        'system_prompt': system_prompt,
                        'query': query_text,
                        'response': f"Error: {str(e)}",
                        'status': 'Unknown Error',
                        'status_code': 'N/A',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                st.session_state.test_results.append(result)
                progress_bar.progress((i + 1) / len(st.session_state.prompts))
            
            status_text.text("Testing completed!")
            st.success(f"Tested {len(st.session_state.prompts)} prompts!")

    if st.session_state.test_results:
        st.subheader("📊 Test Results")
        success_count = sum(1 for r in st.session_state.test_results if r['status'] == 'Success')
        st.metric("Successful Tests", f"{success_count}/{len(st.session_state.test_results)}")
        
        for i, result in enumerate(st.session_state.test_results):
            status_color = "🟢" if result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} Test {i+1} - {result['status']}"):
                st.write("**System Prompt:**")
                st.text(result['system_prompt'])
                st.write("**Query:**")
                st.text(result['query'])
                st.write("**Response:**")
                st.text(result['response'])
                st.write("**Details:**")
                st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']}")

# Export section
if st.session_state.test_results:
    st.header("💾 Export Results")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Excel", type="primary"):
            df = pd.DataFrame(st.session_state.test_results)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Test_Results', index=False)
                worksheet = writer.sheets['Test_Results']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
            excel_data = output.getvalue()
            st.download_button(
                label="Download Excel File",
                data=excel_data,
                file_name=f"chat_api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    with col2:
        if st.button("📋 Show DataFrame"):
            df = pd.DataFrame(st.session_state.test_results)
            st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("💡 **Tips:** Add multiple system prompts, enter one query, and test how each prompt affects the model's response.")
