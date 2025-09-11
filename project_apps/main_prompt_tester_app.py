import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import io
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Set page config
st.set_page_config(
    page_title="Chat API Tester",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state
if 'prompts' not in st.session_state:
    st.session_state.prompts = []
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'chain_results' not in st.session_state:
    st.session_state.chain_results = []
if 'combination_results' not in st.session_state:
    st.session_state.combination_results = []

st.title("🔮 Enhanced Chat API Tester")
st.markdown("Test **system prompts**, create **prompt chains**, and **combine prompts** with AI assistance")

# Sidebar for API configuration
st.sidebar.header("🔧 API Configuration")

# API URL input
api_url = st.sidebar.text_input(
    "API Endpoint URL",
    placeholder="https://api.example.com/chat/rag",
    help="Enter the full URL of your chat API endpoint"
)

# Gemini API Key for prompt combination
st.sidebar.subheader("🤖 Gemini API (for prompt combination)")
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Required for intelligent prompt combination feature"
)

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        st.sidebar.success("✅ Gemini API configured")
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

# Test mode selection
test_mode = st.sidebar.selectbox(
    "🎯 Test Mode",
    ["Individual Testing", "Prompt Chaining", "Prompt Combination"],
    help="Choose how to test your prompts"
)

# Main content area
if test_mode == "Individual Testing":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("✏️ System Prompt Manager")
        
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

elif test_mode == "Prompt Chaining":
    st.header("🔗 Prompt Chaining")
    st.markdown("Create a chain where each prompt uses the output of the previous one")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Chain Configuration")
        
        # Display prompts with ordering
        if st.session_state.prompts:
            st.write("**Current Chain Order:**")
            for i, prompt in enumerate(st.session_state.prompts):
                st.write(f"**Step {i+1}:** {prompt[:60]}...")
                
            # Reorder prompts
            if len(st.session_state.prompts) > 1:
                st.subheader("Reorder Chain")
                new_order = st.multiselect(
                    "Select prompts in desired order:",
                    options=list(range(len(st.session_state.prompts))),
                    format_func=lambda x: f"Step {x+1}: {st.session_state.prompts[x][:50]}...",
                    default=list(range(len(st.session_state.prompts)))
                )
                
                if st.button("🔄 Apply New Order") and len(new_order) == len(st.session_state.prompts):
                    st.session_state.prompts = [st.session_state.prompts[i] for i in new_order]
                    st.success("Chain order updated!")
                    st.rerun()
        else:
            st.info("Add system prompts first to create a chain")
    
    with col2:
        st.subheader("Chain Execution")
        
        if st.button("⛓️ Execute Chain", type="primary", disabled=not (api_url and st.session_state.prompts and query_text)):
            if not api_url:
                st.error("Please enter an API endpoint URL")
            elif not st.session_state.prompts:
                st.error("Please add at least one system prompt")
            elif not query_text:
                st.error("Please enter a query")
            else:
                st.session_state.chain_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                current_query = query_text
                
                for i, system_prompt in enumerate(st.session_state.prompts):
                    status_text.text(f"Executing chain step {i+1}/{len(st.session_state.prompts)}...")
                    
                    try:
                        safe_system = system_prompt.replace("\n", "\\n").replace("\"", "\\\"")
                        safe_query = current_query.replace("\n", "\\n").replace("\"", "\\\"")

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
                                'step': i + 1,
                                'system_prompt': system_prompt,
                                'input_query': current_query,
                                'response': str(response_text),
                                'status': 'Success',
                                'status_code': response.status_code,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Use this response as input for next step
                            current_query = str(response_text)
                            
                        else:
                            result = {
                                'step': i + 1,
                                'system_prompt': system_prompt,
                                'input_query': current_query,
                                'response': f"Error: {response.text}",
                                'status': 'Error',
                                'status_code': response.status_code,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            # Stop chain on error
                            st.session_state.chain_results.append(result)
                            break
                            
                    except Exception as e:
                        result = {
                            'step': i + 1,
                            'system_prompt': system_prompt,
                            'input_query': current_query,
                            'response': f"Error: {str(e)}",
                            'status': 'Unknown Error',
                            'status_code': 'N/A',
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        # Stop chain on error
                        st.session_state.chain_results.append(result)
                        break

                    st.session_state.chain_results.append(result)
                    progress_bar.progress((i + 1) / len(st.session_state.prompts))
                
                status_text.text("Chain execution completed!")
                st.success(f"Executed {len(st.session_state.chain_results)} chain steps!")

    # Display chain results
    if st.session_state.chain_results:
        st.subheader("🔗 Chain Results")
        
        # Final result summary
        final_result = st.session_state.chain_results[-1]
        if final_result['status'] == 'Success':
            st.success("✅ Chain completed successfully!")
            st.subheader("🎯 Final Result")
            st.text_area("Final Output:", value=final_result['response'], height=150, disabled=True)
        else:
            st.error(f"❌ Chain failed at step {final_result['step']}")
        
        # Individual step results
        st.subheader("📋 Step-by-Step Results")
        for result in st.session_state.chain_results:
            status_color = "🟢" if result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} Step {result['step']} - {result['status']}"):
                st.write("**System Prompt:**")
                st.text(result['system_prompt'])
                st.write("**Input Query:**")
                st.text(result['input_query'])
                st.write("**Response:**")
                st.text(result['response'])
                st.write("**Details:**")
                st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']}")

elif test_mode == "Prompt Combination":
    st.header("🤝 Prompt Combination")
    st.markdown("Use AI to intelligently combine multiple prompts into one optimized prompt")
    
    if not gemini_api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to use prompt combination")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Select Prompts to Combine")
        
        if st.session_state.prompts:
            selected_prompts = st.multiselect(
                "Choose prompts to combine:",
                options=list(range(len(st.session_state.prompts))),
                format_func=lambda x: f"Prompt {x+1}: {st.session_state.prompts[x][:50]}...",
                default=list(range(min(2, len(st.session_state.prompts))))
            )
            
            if selected_prompts:
                st.subheader("Selected Prompts Preview")
                for i, idx in enumerate(selected_prompts):
                    with st.expander(f"Prompt {idx+1}"):
                        st.text(st.session_state.prompts[idx])
                
                # Combination strategy
                combination_strategy = st.selectbox(
                    "Combination Strategy:",
                    [
                        "Merge and optimize for clarity",
                        "Combine while preserving all instructions",
                        "Create a hierarchical prompt structure",
                        "Synthesize into a concise unified prompt"
                    ]
                )
        else:
            st.info("Add system prompts first to combine them")
    
    with col2:
        st.subheader("AI Combination")
        
        if st.button("🤖 Combine Prompts with AI", type="primary", disabled=not (gemini_api_key and st.session_state.prompts and 'selected_prompts' in locals() and selected_prompts)):
            if not gemini_api_key:
                st.error("Please enter your Gemini API key")
            elif not selected_prompts:
                st.error("Please select at least 2 prompts to combine")
            else:
                try:
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    selected_prompt_texts = [st.session_state.prompts[i] for i in selected_prompts]
                    
                    combination_prompt = f"""
Please combine the following system prompts into one optimized, coherent system prompt.

Strategy: {combination_strategy}

Individual Prompts:
{chr(10).join([f'Prompt {i+1}: {prompt}' for i, prompt in enumerate(selected_prompt_texts)])}

Requirements:
1. Preserve the core intent and functionality of each individual prompt
2. Eliminate redundancy and conflicts
3. Create a well-structured, clear, and actionable combined prompt
4. Maintain the essential instructions and constraints from all prompts
5. Ensure the combined prompt is more effective than using the prompts separately

Return only the combined system prompt without additional explanation.
"""
                    
                    with st.spinner("AI is combining prompts..."):
                        response = model.generate_content(combination_prompt)
                        combined_prompt = response.text
                        
                        st.session_state.combination_results = {
                            'individual_prompts': selected_prompt_texts,
                            'combined_prompt': combined_prompt,
                            'strategy': combination_strategy,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'individual_results': [],
                            'combined_result': None
                        }
                        
                        st.success("✅ Prompts combined successfully!")
                        
                except Exception as e:
                    st.error(f"Error combining prompts: {str(e)}")
        
        # Test combined prompt
        if st.session_state.combination_results and st.button("🧪 Test Combined vs Individual", type="secondary", disabled=not (api_url and query_text)):
            if not api_url or not query_text:
                st.error("Please configure API endpoint and enter a query")
            else:
                with st.spinner("Testing individual prompts..."):
                    individual_results = []
                    
                    # Test individual prompts
                    for i, prompt in enumerate(st.session_state.combination_results['individual_prompts']):
                        try:
                            safe_system = prompt.replace("\n", "\\n").replace("\"", "\\\"")
                            safe_query = query_text.replace("\n", "\\n").replace("\"", "\\\"")

                            body = body_template.replace("{system_prompt}", safe_system).replace("{query}", safe_query)
                            body_json = json.loads(body)

                            response = requests.post(api_url, headers=headers, json=body_json, timeout=30)

                            if response.status_code == 200:
                                response_data = response.json()
                                response_text = response_data
                                for key in response_path.split('.'):
                                    if key in response_text:
                                        response_text = response_text[key]
                                    else:
                                        response_text = str(response_data)
                                        break
                                
                                individual_results.append({
                                    'prompt_index': i + 1,
                                    'response': str(response_text),
                                    'status': 'Success'
                                })
                            else:
                                individual_results.append({
                                    'prompt_index': i + 1,
                                    'response': f"Error: {response.text}",
                                    'status': 'Error'
                                })
                        except Exception as e:
                            individual_results.append({
                                'prompt_index': i + 1,
                                'response': f"Error: {str(e)}",
                                'status': 'Error'
                            })
                
                with st.spinner("Testing combined prompt..."):
                    # Test combined prompt
                    try:
                        combined_prompt = st.session_state.combination_results['combined_prompt']
                        safe_system = combined_prompt.replace("\n", "\\n").replace("\"", "\\\"")
                        safe_query = query_text.replace("\n", "\\n").replace("\"", "\\\"")

                        body = body_template.replace("{system_prompt}", safe_system).replace("{query}", safe_query)
                        body_json = json.loads(body)

                        response = requests.post(api_url, headers=headers, json=body_json, timeout=30)

                        if response.status_code == 200:
                            response_data = response.json()
                            response_text = response_data
                            for key in response_path.split('.'):
                                if key in response_text:
                                    response_text = response_text[key]
                                else:
                                    response_text = str(response_data)
                                    break
                            
                            combined_result = {
                                'response': str(response_text),
                                'status': 'Success'
                            }
                        else:
                            combined_result = {
                                'response': f"Error: {response.text}",
                                'status': 'Error'
                            }
                    except Exception as e:
                        combined_result = {
                            'response': f"Error: {str(e)}",
                            'status': 'Error'
                        }
                
                st.session_state.combination_results['individual_results'] = individual_results
                st.session_state.combination_results['combined_result'] = combined_result
                
                st.success("✅ Testing completed!")

    # Display combination results
    if st.session_state.combination_results:
        st.subheader("🎯 Combination Results")
        
        # Show combined prompt
        st.subheader("🤖 AI-Generated Combined Prompt")
        st.text_area(
            "Combined Prompt:",
            value=st.session_state.combination_results['combined_prompt'],
            height=200,
            disabled=True
        )
        
        # Show test results if available
        if st.session_state.combination_results.get('individual_results') and st.session_state.combination_results.get('combined_result'):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Individual Prompt Results")
                for result in st.session_state.combination_results['individual_results']:
                    status_color = "🟢" if result['status'] == 'Success' else "🔴"
                    with st.expander(f"{status_color} Individual Prompt {result['prompt_index']}"):
                        st.text(result['response'])
            
            with col2:
                st.subheader("🤝 Combined Prompt Result")
                combined_result = st.session_state.combination_results['combined_result']
                status_color = "🟢" if combined_result['status'] == 'Success' else "🔴"
                
                st.markdown(f"**Status:** {status_color} {combined_result['status']}")
                st.text_area("Combined Response:", value=combined_result['response'], height=300, disabled=True)

# Export section
export_data = []
if st.session_state.test_results:
    export_data.extend([{**result, 'test_type': 'Individual'} for result in st.session_state.test_results])

if st.session_state.chain_results:
    export_data.extend([{**result, 'test_type': 'Chain'} for result in st.session_state.chain_results])

if st.session_state.combination_results and st.session_state.combination_results.get('combined_result'):
    export_data.append({
        'system_prompt': st.session_state.combination_results['combined_prompt'],
        'query': query_text,
        'response': st.session_state.combination_results['combined_result']['response'],
        'status': st.session_state.combination_results['combined_result']['status'],
        'timestamp': st.session_state.combination_results['timestamp'],
        'test_type': 'Combined'
    })

if export_data:
    st.header("💾 Export Results")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Excel", type="primary"):
            df = pd.DataFrame(export_data)
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
                file_name=f"enhanced_api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    with col2:
        if st.button("📋 Show DataFrame"):
            df = pd.DataFrame(export_data)
            st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown("""
💡 **Enhanced Features:**
- **Individual Testing:** Test multiple system prompts independently
- **Prompt Chaining:** Chain prompts where each step uses the previous output
- **Prompt Combination:** Use AI to intelligently combine multiple prompts into one optimized prompt
""")