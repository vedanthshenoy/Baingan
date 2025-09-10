import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import io
import google.generativeai as genai
import os
from dotenv import load_dotenv

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

st.title("🔮 Baingan 🍆")
st.markdown("Test **system prompts**, create **prompt chains**, and **combine prompts** with AI assistance")

# Sidebar for API configuration
st.sidebar.header("🔧 API Configuration")

# API URL input
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

# Utility functions
def ensure_prompt_names():
    """Ensure prompt_names list matches prompts list length"""
    while len(st.session_state.prompt_names) < len(st.session_state.prompts):
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompt_names) + 1}")
    while len(st.session_state.prompt_names) > len(st.session_state.prompts):
        st.session_state.prompt_names.pop()

def add_prompt_section():
    """Common section for adding/editing prompts"""
    st.subheader("✏️ Prompt Management")
    
    # Add new prompt
    col1, col2 = st.columns([3, 1])
    with col1:
        new_prompt = st.text_area("Enter your system prompt:", height=100, key=f"new_prompt_input_{test_mode}")
    with col2:
        st.write("") # spacing
        new_prompt_name = st.text_input("Prompt Name:", placeholder=f"Prompt {len(st.session_state.prompts) + 1}", key=f"new_prompt_name_{test_mode}")
    
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add System Prompt", type="primary", key=f"add_prompt_{test_mode}"):
            if new_prompt.strip():
                st.session_state.prompts.append(new_prompt.strip())
                prompt_name = new_prompt_name.strip() if new_prompt_name.strip() else f"Prompt {len(st.session_state.prompts)}"
                st.session_state.prompt_names.append(prompt_name)
                st.success(f"Added: {prompt_name}")
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Clear All Prompts", key=f"clear_prompts_{test_mode}"):
            st.session_state.prompts = []
            st.session_state.prompt_names = []
            st.session_state.test_results = []
            st.session_state.chain_results = []
            st.session_state.combination_results = []
            st.success("Cleared all prompts and results")
            st.rerun()
    
    # Display and edit current prompts
    if st.session_state.prompts:
        ensure_prompt_names()
        st.subheader(f"📋 Current Prompts ({len(st.session_state.prompts)})")
        
        for i in range(len(st.session_state.prompts)):
            with st.expander(f"{st.session_state.prompt_names[i]}: {st.session_state.prompts[i][:50]}..."):
                # Edit prompt name
                new_name = st.text_input("Name:", value=st.session_state.prompt_names[i], key=f"edit_name_{i}_{test_mode}")
                if new_name != st.session_state.prompt_names[i]:
                    if st.button(f"💾 Update Name", key=f"update_name_{i}_{test_mode}"):
                        st.session_state.prompt_names[i] = new_name
                        st.success(f"Updated name to: {new_name}")
                        st.rerun()
                
                # Edit prompt content
                edited_prompt = st.text_area("Content:", value=st.session_state.prompts[i], height=100, key=f"edit_prompt_{i}_{test_mode}")
                if edited_prompt != st.session_state.prompts[i]:
                    if st.button(f"💾 Update Content", key=f"update_content_{i}_{test_mode}"):
                        st.session_state.prompts[i] = edited_prompt
                        st.success("Updated prompt content")
                        st.rerun()
                
                # Remove prompt
                if st.button(f"🗑️ Remove", key=f"remove_{i}_{test_mode}"):
                    st.session_state.prompts.pop(i)
                    st.session_state.prompt_names.pop(i)
                    # Clean up results
                    if i < len(st.session_state.test_results):
                        st.session_state.test_results.pop(i)
                    if i < len(st.session_state.chain_results):
                        st.session_state.chain_results.pop(i)
                    st.rerun()

def call_api(system_prompt, query):
    """Make API call and return result"""
    try:
        safe_system = system_prompt.replace("\n", "\\n").replace("\"", "\\\"")
        safe_query = query.replace("\n", "\\n").replace("\"", "\\\"")

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
            return {
                'response': str(response_text),
                'status': 'Success',
                'status_code': response.status_code
            }
        else:
            return {
                'response': f"Error: {response.text}",
                'status': 'Error',
                'status_code': response.status_code
            }
    except Exception as e:
        return {
            'response': f"Error: {str(e)}",
            'status': 'Unknown Error',
            'status_code': 'N/A'
        }

def suggest_prompt_from_response(target_response, query):
    """Use Gemini to suggest a prompt that could generate the target response"""
    if not gemini_api_key:
        return "Gemini API key required for prompt suggestion"
    
    try:
        # Convert temperature from 0-100 to 0-2 scale for Gemini
        gemini_temperature = (st.session_state.get('temperature', 50) / 100.0) * 2.0
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        suggestion_prompt = f"""
Based on the following desired response/output, suggest a system prompt that could generate this type of response when given appropriate queries.

Target Response:
{target_response}

Original Query Context: {query}

Please analyze the response style, tone, structure, and content approach, then suggest a comprehensive system prompt that would guide an AI to produce similar responses. Focus on:
1. Response format and structure
2. Tone and style guidelines
3. Content depth and approach
4. Any specific instructions that seem evident from the output

Return only the suggested system prompt without additional explanation.
"""
        
        generation_config = genai.types.GenerationConfig(
            temperature=gemini_temperature
        )
        
        response = model.generate_content(suggestion_prompt, generation_config=generation_config)
        return response.text
        
    except Exception as e:
        return f"Error generating prompt suggestion: {str(e)}"

st.markdown("---")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    add_prompt_section()

with col2:
    if test_mode == "Individual Testing":
        st.header("🧪 Individual Testing")
        
        if st.button("🚀 Test All Prompts", type="primary", disabled=not (api_url and st.session_state.prompts and query_text)):
            if not api_url:
                st.error("Please enter an API endpoint URL")
            elif not st.session_state.prompts:
                st.error("Please add at least one system prompt")
            elif not query_text:
                st.error("Please enter a query")
            else:
                ensure_prompt_names()
                st.session_state.test_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                    status_text.text(f"Testing {prompt_name}...")
                    
                    result = call_api(system_prompt, query_text)
                    result.update({
                        'prompt_name': prompt_name,
                        'system_prompt': system_prompt,
                        'query': query_text,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

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
                with st.expander(f"{status_color} {result['prompt_name']} - {result['status']}"):
                    st.write("**System Prompt:**")
                    st.text(result['system_prompt'])
                    st.write("**Query:**")
                    st.text(result['query'])
                    st.write("**Response:**")
                    
                    # Editable response
                    edited_response = st.text_area(
                        "Response (editable):", 
                        value=result['response'], 
                        height=150, 
                        key=f"edit_response_{i}"
                    )
                    
                    if edited_response != result['response']:
                        col_save, col_reverse = st.columns(2)
                        with col_save:
                            if st.button(f"💾 Save Edited Response", key=f"save_response_{i}"):
                                st.session_state.test_results[i]['response'] = edited_response
                                st.session_state.test_results[i]['edited'] = True
                                st.success("Response updated!")
                                st.rerun()
                        with col_reverse:
                            if st.button(f"🔄 Reverse Prompt", key=f"reverse_{i}"):
                                with st.spinner("Generating updated prompt..."):
                                    suggestion = suggest_prompt_from_response(edited_response, result['query'])
                                    st.session_state.prompts[i] = suggestion
                                    st.session_state.test_results[i]['system_prompt'] = suggestion
                                    st.success("Prompt updated based on edited response!")
                                    st.rerun()
                    
                    # Prompt suggestion button
                    if st.button(f"🔮 Suggest Prompt for This Response", key=f"suggest_{i}"):
                        with st.spinner("Generating prompt suggestion..."):
                            suggestion = suggest_prompt_from_response(edited_response, result['query'])
                            st.write("**Suggested System Prompt:**")
                            st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_{i}")
                    
                    st.write("**Details:**")
                    st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']}")

    elif test_mode == "Prompt Chaining":
        st.header("🔗 Prompt Chaining")
        
        if st.session_state.prompts:
            ensure_prompt_names()
            st.write("**Current Chain Order:**")
            for i, (prompt, name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                st.write(f"**Step {i+1}:** {name}")
                
            # Reorder prompts
            if len(st.session_state.prompts) > 1:
                st.subheader("Reorder Chain")
                new_order = st.multiselect(
                    "Select prompts in desired order:",
                    options=list(range(len(st.session_state.prompts))),
                    format_func=lambda x: f"Step {x+1}: {st.session_state.prompt_names[x]}",
                    default=list(range(len(st.session_state.prompts)))
                )
                
                if st.button("🔄 Apply New Order") and len(new_order) == len(st.session_state.prompts):
                    st.session_state.prompts = [st.session_state.prompts[i] for i in new_order]
                    st.session_state.prompt_names = [st.session_state.prompt_names[i] for i in new_order]
                    st.success("Chain order updated!")
                    st.rerun()
        else:
            st.info("Add system prompts first to create a chain")
        
        if st.button("⛓️ Execute Chain", type="primary", disabled=not (api_url and st.session_state.prompts and query_text)):
            if not api_url:
                st.error("Please enter an API endpoint URL")
            elif not st.session_state.prompts:
                st.error("Please add at least one system prompt")
            elif not query_text:
                st.error("Please enter a query")
            else:
                ensure_prompt_names()
                st.session_state.chain_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                current_query = query_text
                
                for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                    status_text.text(f"Executing step {i+1}: {prompt_name}...")
                    
                    result = call_api(system_prompt, current_query)
                    result.update({
                        'step': i + 1,
                        'prompt_name': prompt_name,
                        'system_prompt': system_prompt,
                        'input_query': current_query,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.session_state.chain_results.append(result)
                    
                    if result['status'] != 'Success':
                        break
                    
                    # Use this response as input for next step
                    current_query = result['response']
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
                
                # Editable final response
                edited_final = st.text_area("Final Output (editable):", value=final_result['response'], height=150, key="edit_final_chain")
                
                if edited_final != final_result['response']:
                    col_save, col_reverse = st.columns(2)
                    with col_save:
                        if st.button("💾 Save Final Response"):
                            st.session_state.chain_results[-1]['response'] = edited_final
                            st.session_state.chain_results[-1]['edited'] = True
                            st.success("Final response updated!")
                            st.rerun()
                    with col_reverse:
                        if st.button("🔄 Reverse Prompt for Final"):
                            with st.spinner("Generating updated prompt..."):
                                suggestion = suggest_prompt_from_response(edited_final, final_result['input_query'])
                                last_index = len(st.session_state.prompts) - 1
                                st.session_state.prompts[last_index] = suggestion
                                st.session_state.chain_results[-1]['system_prompt'] = suggestion
                                st.success("Final prompt updated based on edited response!")
                                st.rerun()
                
                if st.button("🔮 Suggest Prompt for Final Response"):
                    with st.spinner("Generating prompt suggestion..."):
                        suggestion = suggest_prompt_from_response(edited_final, final_result['input_query'])
                        st.write("**Suggested System Prompt:**")
                        st.text_area("Suggested Prompt:", value=suggestion, height=100, key="suggested_final_chain")
                        
            else:
                st.error(f"❌ Chain failed at step {final_result['step']}: {final_result['prompt_name']}")
            
            # Individual step results
            st.subheader("📋 Step-by-Step Results")
            for j, result in enumerate(st.session_state.chain_results):
                status_color = "🟢" if result['status'] == 'Success' else "🔴"
                with st.expander(f"{status_color} Step {result['step']}: {result['prompt_name']} - {result['status']}"):
                    st.write("**System Prompt:**")
                    st.text(result['system_prompt'])
                    st.write("**Input Query:**")
                    st.text(result['input_query'])
                    st.write("**Response:**")
                    
                    # Editable response
                    edited_step_response = st.text_area(
                        "Response (editable):", 
                        value=result['response'], 
                        height=150, 
                        key=f"edit_chain_response_{j}"
                    )
                    
                    if edited_step_response != result['response']:
                        col_save, col_reverse = st.columns(2)
                        with col_save:
                            if st.button(f"💾 Save Response", key=f"save_chain_response_{j}"):
                                st.session_state.chain_results[j]['response'] = edited_step_response
                                st.session_state.chain_results[j]['edited'] = True
                                st.success("Step response updated!")
                                st.rerun()
                        with col_reverse:
                            if st.button(f"🔄 Reverse Prompt", key=f"reverse_chain_{j}"):
                                with st.spinner("Generating updated prompt..."):
                                    suggestion = suggest_prompt_from_response(edited_step_response, result['input_query'])
                                    st.session_state.prompts[j] = suggestion
                                    st.session_state.chain_results[j]['system_prompt'] = suggestion
                                    st.success("Prompt updated based on edited response!")
                                    st.rerun()
                    
                    # Prompt suggestion button
                    if st.button(f"🔮 Suggest Prompt for This Response", key=f"suggest_chain_{j}"):
                        with st.spinner("Generating prompt suggestion..."):
                            suggestion = suggest_prompt_from_response(edited_step_response, result['input_query'])
                            st.write("**Suggested System Prompt:**")
                            st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_chain_{j}")
                    
                    st.write("**Details:**")
                    st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']}")

    elif test_mode == "Prompt Combination":
        st.header("🤝 Prompt Combination")
        
        if not gemini_api_key:
            st.warning("⚠️ Please configure Gemini API key to use prompt combination")
        
        # Temperature slider (only for Prompt Combination mode)
        temperature = st.slider(
            "🌡️ AI Temperature (Creativity)",
            min_value=0,
            max_value=100,
            value=50,
            help="Controls creativity of AI responses. Lower = more focused, Higher = more creative"
        )
        st.session_state.temperature = temperature
        
        if st.session_state.prompts:
            ensure_prompt_names()
            selected_prompts = st.multiselect(
                "Choose prompts to combine:",
                options=list(range(len(st.session_state.prompts))),
                format_func=lambda x: f"{st.session_state.prompt_names[x]}: {st.session_state.prompts[x][:50]}...",
                default=list(range(min(2, len(st.session_state.prompts))))
            )
            
            if selected_prompts:
                st.subheader("Selected Prompts Preview")
                for idx in selected_prompts:
                    with st.expander(f"{st.session_state.prompt_names[idx]}"):
                        st.text(st.session_state.prompts[idx])
                
                # Combination strategy
                combination_strategy = st.selectbox(
                    "Combination Strategy:",
                    [
                        "Merge and optimize for clarity",
                        "Combine while preserving all instructions",
                        "Create a hierarchical prompt structure",
                        "Synthesize into a concise unified prompt",
                        "Slider - Custom influence weights"
                    ]
                )
                
                # Slider strategy
                slider_weights = {}
                if combination_strategy == "Slider - Custom influence weights":
                    st.subheader("🎚️ Influence Weights")
                    st.write("Set how much influence each prompt should have (0-100%):")
                    
                    for idx in selected_prompts:
                        weight = st.slider(
                            f"{st.session_state.prompt_names[idx]}:",
                            min_value=0,
                            max_value=100,
                            value=100 // len(selected_prompts),
                            key=f"weight_{idx}"
                        )
                        slider_weights[idx] = weight
                    
                    total_weight = sum(slider_weights.values())
                    if total_weight > 0:
                        st.write(f"**Total Weight:** {total_weight}% (weights will be normalized)")
                    else:
                        st.warning("Please set at least one prompt weight > 0%")
        else:
            st.info("Add system prompts first to combine them")
        
        if st.button("🤖 Combine Prompts with AI", type="primary", disabled=not (gemini_api_key and st.session_state.prompts and 'selected_prompts' in locals() and selected_prompts)):
            if not gemini_api_key:
                st.error("Please configure Gemini API key")
            elif not selected_prompts:
                st.error("Please select at least 2 prompts to combine")
            elif combination_strategy == "Slider - Custom influence weights" and sum(slider_weights.values()) == 0:
                st.error("Please set at least one prompt weight > 0%")
            else:
                try:
                    # Convert temperature from 0-100 to 0-2 scale for Gemini
                    gemini_temperature = (temperature / 100.0) * 2.0
                    
                    model = genai.GenerativeModel('gemini-2.0-flash-exp')
                    
                    selected_prompt_texts = [st.session_state.prompts[i] for i in selected_prompts]
                    selected_prompt_names = [st.session_state.prompt_names[i] for i in selected_prompts]
                    
                    if combination_strategy == "Slider - Custom influence weights":
                        # Normalize weights
                        total_weight = sum(slider_weights.values())
                        normalized_weights = {k: (v/total_weight)*100 for k, v in slider_weights.items()}
                        
                        weight_info = "\n".join([
                            f"{st.session_state.prompt_names[idx]} ({normalized_weights[idx]:.1f}% influence): {st.session_state.prompts[idx]}" 
                            for idx in selected_prompts
                        ])
                        
                        combination_prompt = f"""
Please combine the following system prompts into one optimized prompt, using the specified influence weights to determine how much each prompt should contribute to the final result.

Weighted Prompts:
{weight_info}

Requirements:
1. Use the influence percentages to determine how much each prompt contributes
2. Higher weight prompts should have more prominent influence on the final structure and content
3. Lower weight prompts should contribute supporting elements or nuances
4. Create a coherent, unified prompt that balances all influences appropriately
5. Preserve the most important aspects from higher-weighted prompts
6. Eliminate redundancy and conflicts intelligently

Return only the combined system prompt without additional explanation.
"""
                    else:
                        prompt_info = "\n".join([f'{name}: {prompt}' for name, prompt in zip(selected_prompt_names, selected_prompt_texts)])
                        
                        combination_prompt = f"""
Please combine the following system prompts into one optimized, coherent system prompt.

Strategy: {combination_strategy}

Individual Prompts:
{prompt_info}

Requirements:
1. Preserve the core intent and functionality of each individual prompt
2. Eliminate redundancy and conflicts
3. Create a well-structured, clear, and actionable combined prompt
4. Maintain the essential instructions and constraints from all prompts
5. Ensure the combined prompt is more effective than using the prompts separately

Return only the combined system prompt without additional explanation.
"""
                    
                    generation_config = genai.types.GenerationConfig(
                        temperature=gemini_temperature
                    )
                    
                    with st.spinner("AI is combining prompts..."):
                        response = model.generate_content(combination_prompt, generation_config=generation_config)
                        combined_prompt = response.text
                        
                        st.session_state.combination_results = {
                            'individual_prompts': selected_prompt_texts,
                            'individual_names': selected_prompt_names,
                            'selected_indices': selected_prompts,
                            'combined_prompt': combined_prompt,
                            'strategy': combination_strategy,
                            'temperature': temperature,
                            'slider_weights': slider_weights if combination_strategy == "Slider - Custom influence weights" else None,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'individual_results': [],
                            'combined_result': None
                        }
                        
                        st.success("✅ Prompts combined successfully!")
                        
                except Exception as e:
                    st.error(f"Error combining prompts: {str(e)}")
        
        # Test combined prompt
        if st.session_state.combination_results:
            if st.button("🧪 Test Combined vs Individual Prompts", type="primary", disabled=not (api_url and query_text)):
                if not api_url or not query_text:
                    st.error("Please configure API endpoint and enter a query")
                else:
                    with st.spinner("Testing individual prompts..."):
                        individual_results = []
                        
                        # Test individual prompts
                        for i, (prompt, name) in enumerate(zip(st.session_state.combination_results['individual_prompts'], st.session_state.combination_results['individual_names'])):
                            result = call_api(prompt, query_text)
                            result.update({
                                'prompt_index': i + 1,
                                'prompt_name': name,
                                'system_prompt': prompt
                            })
                            individual_results.append(result)
                    
                    with st.spinner("Testing combined prompt..."):
                        # Test combined prompt
                        combined_result = call_api(st.session_state.combination_results['combined_prompt'], query_text)
                        combined_result['system_prompt'] = st.session_state.combination_results['combined_prompt']
                    
                    st.session_state.combination_results['individual_results'] = individual_results
                    st.session_state.combination_results['combined_result'] = combined_result
                    
                    st.success("✅ Testing completed!")

        # Display combination results
        if st.session_state.combination_results:
            st.subheader("🎯 Combination Results")
            
            # Show combined prompt
            st.subheader("🤖 AI-Generated Combined Prompt")
            combined_prompt_text = st.text_area(
                "Combined Prompt (editable):",
                value=st.session_state.combination_results['combined_prompt'],
                height=200,
                key="edit_combined_prompt"
            )
            
            if combined_prompt_text != st.session_state.combination_results['combined_prompt']:
                if st.button("💾 Save Combined Prompt"):
                    st.session_state.combination_results['combined_prompt'] = combined_prompt_text
                    st.success("Combined prompt updated!")
                    st.rerun()
            
            # Show strategy, temperature and weights if applicable
            st.write(f"**Strategy:** {st.session_state.combination_results.get('strategy')}")
            st.write(f"**Temperature:** {st.session_state.combination_results.get('temperature', 50)}%")
            
            if st.session_state.combination_results.get('slider_weights'):
                st.write("**Influence Weights Used:**")
                for idx, weight in st.session_state.combination_results['slider_weights'].items():
                    name = st.session_state.combination_results['individual_names'][list(st.session_state.combination_results['slider_weights'].keys()).index(idx)]
                    st.write(f"- {name}: {weight}%")
            
            # Show test results if available
            if st.session_state.combination_results.get('individual_results') and st.session_state.combination_results.get('combined_result'):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📄 Individual Prompt Results")
                    for j, result in enumerate(st.session_state.combination_results['individual_results']):
                        status_color = "🟢" if result['status'] == 'Success' else "🔴"
                        with st.expander(f"{status_color} {result['prompt_name']}"):
                            edited_individual_response = st.text_area(
                                "Response (editable):", 
                                value=result['response'], 
                                height=150, 
                                key=f"edit_individual_{j}"
                            )
                            
                            if edited_individual_response != result['response']:
                                col_save, col_reverse = st.columns(2)
                                with col_save:
                                    if st.button(f"💾 Save Response", key=f"save_individual_{j}"):
                                        st.session_state.combination_results['individual_results'][j]['response'] = edited_individual_response
                                        st.session_state.combination_results['individual_results'][j]['edited'] = True
                                        st.success("Response updated!")
                                        st.rerun()
                                with col_reverse:
                                    if st.button(f"🔄 Reverse Prompt", key=f"reverse_individual_{j}"):
                                        with st.spinner("Generating updated prompt..."):
                                            suggestion = suggest_prompt_from_response(edited_individual_response, query_text)
                                            source_idx = st.session_state.combination_results['selected_indices'][j]
                                            st.session_state.prompts[source_idx] = suggestion
                                            st.session_state.combination_results['individual_prompts'][j] = suggestion
                                            st.session_state.combination_results['individual_results'][j]['system_prompt'] = suggestion
                                            st.success("Prompt updated based on edited response!")
                                            st.rerun()
                            
                            if st.button(f"🔮 Suggest Prompt", key=f"suggest_individual_{j}"):
                                with st.spinner("Generating prompt suggestion..."):
                                    suggestion = suggest_prompt_from_response(edited_individual_response, query_text)
                                    st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_individual_{j}")
                
                with col2:
                    st.subheader("🤝 Combined Prompt Result")
                    combined_result = st.session_state.combination_results['combined_result']
                    status_color = "🟢" if combined_result['status'] == 'Success' else "🔴"
                    
                    st.markdown(f"**Status:** {status_color} {combined_result['status']}")
                    
                    edited_combined_response = st.text_area(
                        "Combined Response (editable):", 
                        value=combined_result['response'], 
                        height=300, 
                        key="edit_combined_response"
                    )
                    
                    if edited_combined_response != combined_result['response']:
                        col_save, col_reverse = st.columns(2)
                        with col_save:
                            if st.button("💾 Save Combined Response"):
                                st.session_state.combination_results['combined_result']['response'] = edited_combined_response
                                st.session_state.combination_results['combined_result']['edited'] = True
                                st.success("Combined response updated!")
                                st.rerun()
                        with col_reverse:
                            if st.button("🔄 Reverse Prompt for Combined"):
                                with st.spinner("Generating updated prompt..."):
                                    suggestion = suggest_prompt_from_response(edited_combined_response, query_text)
                                    st.session_state.combination_results['combined_prompt'] = suggestion
                                    st.session_state.combination_results['combined_result']['system_prompt'] = suggestion
                                    st.success("Combined prompt updated based on edited response!")
                                    st.rerun()
                    
                    if st.button("🔮 Suggest Prompt for Combined Response"):
                        with st.spinner("Generating prompt suggestion..."):
                            suggestion = suggest_prompt_from_response(edited_combined_response, query_text)
                            st.text_area("Suggested Prompt:", value=suggestion, height=100, key="suggested_combined")

# Export section
export_data = []

# Individual test results
if st.session_state.test_results:
    for result in st.session_state.test_results:
        export_data.append({
            'test_type': 'Individual',
            'prompt_name': result.get('prompt_name', 'Unknown'),
            'system_prompt': result['system_prompt'],
            'query': result['query'],
            'response': result['response'],
            'status': result['status'],
            'status_code': result.get('status_code', 'N/A'),
            'timestamp': result['timestamp'],
            'edited': result.get('edited', False),
            'step': None,
            'input_query': None
        })

# Chain results
if st.session_state.chain_results:
    for result in st.session_state.chain_results:
        export_data.append({
            'test_type': 'Chain',
            'prompt_name': result.get('prompt_name', 'Unknown'),
            'system_prompt': result['system_prompt'],
            'query': query_text,
            'response': result['response'],
            'status': result['status'],
            'status_code': result.get('status_code', 'N/A'),
            'timestamp': result['timestamp'],
            'edited': result.get('edited', False),
            'step': result.get('step'),
            'input_query': result.get('input_query')
        })

# Combination results
if st.session_state.combination_results:
    combination_data = st.session_state.combination_results
    
    # Individual results from combination
    if combination_data.get('individual_results'):
        for result in combination_data['individual_results']:
            export_data.append({
                'test_type': 'Combination_Individual',
                'prompt_name': result.get('prompt_name', 'Unknown'),
                'system_prompt': result.get('system_prompt', ''),
                'query': query_text,
                'response': result['response'],
                'status': result['status'],
                'status_code': result.get('status_code', 'N/A'),
                'timestamp': combination_data['timestamp'],
                'edited': result.get('edited', False),
                'step': None,
                'input_query': None,
                'combination_strategy': combination_data.get('strategy'),
                'combination_temperature': combination_data.get('temperature'),
                'slider_weights': str(combination_data.get('slider_weights')) if combination_data.get('slider_weights') else None
            })
    
    # Combined result
    if combination_data.get('combined_result'):
        export_data.append({
            'test_type': 'Combination_Combined',
            'prompt_name': 'AI_Combined',
            'system_prompt': combination_data['combined_prompt'],
            'query': query_text,
            'response': combination_data['combined_result']['response'],
            'status': combination_data['combined_result']['status'],
            'status_code': combination_data['combined_result'].get('status_code', 'N/A'),
            'timestamp': combination_data['timestamp'],
            'edited': combination_data['combined_result'].get('edited', False),
            'step': None,
            'input_query': None,
            'combination_strategy': combination_data.get('strategy'),
            'combination_temperature': combination_data.get('temperature'),
            'slider_weights': str(combination_data.get('slider_weights')) if combination_data.get('slider_weights') else None
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
- **✏️ Prompt Management:** Add, edit, and name prompts in all test modes
- **🧪 Individual Testing:** Test multiple system prompts independently with editable responses
- **🔗 Prompt Chaining:** Chain prompts where each step uses the previous output
- **🤝 Prompt Combination:** Use AI to intelligently combine multiple prompts
- **🎚️ Slider Strategy:** Custom influence weights for prompt combination
- **🌡️ Temperature Control:** 0-100% slider to control AI creativity for prompt combination
- **🔮 Smart Suggestions:** Generate prompt suggestions from desired responses
- **📊 Comprehensive Export:** All results including individual, chain, and combination data
- **💾 Response Editing:** Edit and save responses, maintain edit history
""")

# Requirements note
st.markdown("""
📦 **Requirements:**
```bash
pip install streamlit requests pandas openpyxl google-generativeai python-dotenv
```

🔑 **Environment Variables:**
Create a `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```
""")