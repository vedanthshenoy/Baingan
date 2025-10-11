import streamlit as st
import pandas as pd
import yaml
import json
import requests
import importlib.util
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO
import tempfile
import traceback

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class PromptTestingAgent:
    """Agent that analyzes code and tests prompts intelligently."""
    
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    def analyze_code(self, code_content: str, file_type: str) -> dict:
        """Analyze the RAG/Agent code to understand how to test it."""
        
        prompt = f"""
You are a code analysis expert. Analyze this {file_type} code and provide a JSON response with the following structure:

{{
    "is_api": true/false,
    "api_endpoint": "FULL endpoint URL with scheme (http://localhost:port/route) if API, else null",
    "http_method": "POST/GET/etc if API",
    "function_name": "main function name to call if not API",
    "required_params": ["list", "of", "required", "parameters"],
    "system_prompt_param": "name of system prompt parameter",
    "query_param": "name of query parameter",
    "additional_params": {{"param": "default_value"}},
    "execution_type": "api" or "function" or "module",
    "import_needed": "module path if function needs to be imported"
}}

CRITICAL RULES FOR API ENDPOINTS:
- For api_endpoint, provide the COMPLETE URL including scheme (http:// or https://)
- ALWAYS use "localhost" or "127.0.0.1" as the host, NEVER use "0.0.0.0"
- If code mentions "0.0.0.0" as host, replace it with "localhost"
- Example: "http://localhost:8000/chat/rag" NOT "http://0.0.0.0:8000/chat/rag"
- Extract the full URL from the code including correct host and port

Code to analyze:
```{file_type}
{code_content}
```

Respond ONLY with valid JSON, no markdown or explanation.
"""
        
        try:
            response = self.model.generate_content(prompt)
            analysis = json.loads(response.text.strip().strip('`').replace('json\n', ''))
            
            # Additional safety check: Replace 0.0.0.0 with localhost if present
            if analysis.get('api_endpoint') and '0.0.0.0' in analysis['api_endpoint']:
                analysis['api_endpoint'] = analysis['api_endpoint'].replace('0.0.0.0', 'localhost')
            
            return analysis
        except Exception as e:
            st.error(f"Error analyzing code: {e}")
            return None
    
    def create_test_payload(self, analysis: dict, system_prompt: str, query: str) -> dict:
        """Create appropriate payload based on code analysis."""
        
        payload = {}
        
        if analysis['system_prompt_param']:
            payload[analysis['system_prompt_param']] = system_prompt
        
        if analysis['query_param']:
            payload[analysis['query_param']] = query
        
        # Add additional parameters
        if analysis.get('additional_params'):
            payload.update(analysis['additional_params'])
        
        return payload
    
    def execute_test(self, analysis: dict, payload: dict, code_content: str = None) -> str:
        """Execute the test based on analysis."""
        
        try:
            if analysis['execution_type'] == 'api':
                # API call
                endpoint = analysis['api_endpoint']
                method = analysis['http_method'].upper()
                
                if method == 'POST':
                    response = requests.post(endpoint, json=payload, timeout=30)
                else:
                    response = requests.get(endpoint, params=payload, timeout=30)
                
                response.raise_for_status()
                result = response.json()
                
                # Try to extract response text intelligently
                if isinstance(result, dict):
                    return result.get('response', result.get('answer', result.get('text', str(result))))
                return str(result)
            
            elif analysis['execution_type'] == 'function':
                # Function call
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code_content)
                    temp_file = f.name
                
                try:
                    spec = importlib.util.spec_from_file_location("test_module", temp_file)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["test_module"] = module
                    spec.loader.exec_module(module)
                    
                    func = getattr(module, analysis['function_name'])
                    result = func(**payload)
                    
                    return str(result)
                finally:
                    os.unlink(temp_file)
            
            else:
                return "ERROR: Unsupported execution type"
        
        except Exception as e:
            return f"ERROR: {str(e)}\n{traceback.format_exc()}"


def load_from_file(uploaded_file) -> list:
    """Load content from various file formats."""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension in ['.yaml', '.yml']:
            # YAML file
            content = yaml.safe_load(uploaded_file)
            return flatten_yaml(content)
        
        elif file_extension == '.txt':
            # Plain text file (one item per line)
            content = uploaded_file.read().decode('utf-8')
            return [line.strip() for line in content.split('\n') if line.strip()]
        
        elif file_extension == '.csv':
            # CSV file (first column or specified column)
            df = pd.read_csv(uploaded_file)
            # Use first column by default
            return df.iloc[:, 0].dropna().astype(str).tolist()
        
        elif file_extension in ['.xlsx', '.xls']:
            # Excel file (first column of first sheet)
            df = pd.read_excel(uploaded_file)
            return df.iloc[:, 0].dropna().astype(str).tolist()
        
        else:
            st.error(f"Unsupported file format: {file_extension}")
            return []
    
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []


def flatten_yaml(obj):
    """Recursively flatten YAML structure to list of values."""
    values = []
    if isinstance(obj, dict):
        for v in obj.values():
            values.extend(flatten_yaml(v))
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                values.extend(flatten_yaml(item))
            else:
                values.append(str(item))
    else:
        values.append(str(obj))
    return values


def main():
    st.set_page_config(page_title="Prompt Testing Agent", layout="wide", page_icon="🧪")
    
    st.title("🧪 Prompt Testing Agent")
    st.markdown("### Intelligent RAG/Agent Prompt Testing with Gemini")
    
    # Initialize session state
    if 'test_results' not in st.session_state:
        st.session_state.test_results = None
    if 'analysis' not in st.session_state:
        st.session_state.analysis = None
    if 'system_prompts' not in st.session_state:
        st.session_state.system_prompts = []
    if 'queries' not in st.session_state:
        st.session_state.queries = []
    if 'code_analyzed' not in st.session_state:
        st.session_state.code_analyzed = False
    
    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # RAG/Agent code upload
        code_file = st.file_uploader(
            "Upload RAG/Agent Code",
            type=['py'],
            help="Upload your RAG or agent Python code (API or function-based)"
        )
        
        st.divider()
        
        # API key check
        if os.getenv("GOOGLE_API_KEY"):
            st.success("✅ Gemini API Key loaded")
        else:
            st.error("❌ GOOGLE_API_KEY not found in .env")
    
    # Main area - Code Analysis (Auto-analyze on upload)
    st.subheader("📝 Code Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if code_file:
            code_content = code_file.read().decode('utf-8')
            
            with st.expander("View Code", expanded=False):
                st.code(code_content, language='python')
            
            # Auto-analyze when file is uploaded
            if not st.session_state.code_analyzed or st.session_state.get('last_code_file') != code_file.name:
                with st.spinner("🔍 Analyzing code with Gemini..."):
                    agent = PromptTestingAgent()
                    analysis = agent.analyze_code(code_content, 'python')
                    
                    if analysis:
                        st.session_state.analysis = analysis
                        st.session_state.code_content = code_content
                        st.session_state.code_analyzed = True
                        st.session_state.last_code_file = code_file.name
                        st.success("✅ Code analyzed successfully!")
                        st.rerun()
        else:
            st.info("👆 Upload your RAG/Agent code to begin")
            st.session_state.code_analyzed = False
    
    with col2:
        if st.session_state.analysis:
            st.markdown("**Analysis Results**")
            analysis = st.session_state.analysis
            
            st.json(analysis)
            
            # Execution details
            if analysis['execution_type'] == 'api':
                st.info(f"**Type:** API Endpoint\n\n**URL:** `{analysis['api_endpoint']}`\n\n**Method:** `{analysis['http_method']}`")
            else:
                st.info(f"**Type:** Function Call\n\n**Function:** `{analysis['function_name']}`")
    
    st.divider()
    
    # System Prompts Section
    st.subheader("💬 System Prompts")
    
    tab1, tab2 = st.tabs(["📤 Upload File", "✏️ Manual Input"])
    
    with tab1:
        prompts_file = st.file_uploader(
            "Upload System Prompts",
            type=['yaml', 'yml', 'txt', 'csv', 'xlsx', 'xls'],
            help="Supported formats: YAML, TXT, CSV, Excel",
            key="prompts_file"
        )
        
        # Auto-load when file is uploaded
        if prompts_file and st.session_state.get('last_prompts_file') != prompts_file.name:
            with st.spinner("Loading prompts..."):
                loaded_prompts = load_from_file(prompts_file)
                if loaded_prompts:
                    st.session_state.system_prompts = loaded_prompts
                    st.session_state.last_prompts_file = prompts_file.name
                    st.success(f"✅ Loaded {len(loaded_prompts)} prompts")
    
    with tab2:
        st.markdown("**Add prompts manually (one per line)**")
        manual_prompts = st.text_area(
            "Enter System Prompts",
            height=200,
            placeholder="You are a helpful AI assistant.\nYou are an expert in the domain.\nYou are a creative writer.",
            key="manual_prompts"
        )
        
        col5, col6 = st.columns([1, 1])
        with col5:
            if st.button("➕ Add Manual Prompts", type="primary"):
                if manual_prompts.strip():
                    new_prompts = [line.strip() for line in manual_prompts.split('\n') if line.strip()]
                    st.session_state.system_prompts.extend(new_prompts)
                    st.success(f"✅ Added {len(new_prompts)} prompts")
        
        with col6:
            if st.button("🗑️ Clear All Prompts"):
                st.session_state.system_prompts = []
                st.session_state.pop('last_prompts_file', None)
                st.success("✅ Cleared all prompts")
                st.rerun()
    
    # Display current prompts
    if st.session_state.system_prompts:
        st.success(f"**Current System Prompts: {len(st.session_state.system_prompts)}**")
        with st.expander("View All System Prompts"):
            for i, prompt in enumerate(st.session_state.system_prompts, 1):
                st.text_area(f"Prompt {i}", prompt, height=80, disabled=True, key=f"view_prompt_{i}")
    else:
        st.warning("⚠️ No system prompts loaded")
    
    st.divider()
    
    # Queries Section
    st.subheader("❓ Test Queries")
    
    tab3, tab4 = st.tabs(["📤 Upload File", "✏️ Manual Input"])
    
    with tab3:
        queries_file = st.file_uploader(
            "Upload Queries",
            type=['yaml', 'yml', 'txt', 'csv', 'xlsx', 'xls'],
            help="Supported formats: YAML, TXT, CSV, Excel",
            key="queries_file"
        )
        
        # Auto-load when file is uploaded
        if queries_file and st.session_state.get('last_queries_file') != queries_file.name:
            with st.spinner("Loading queries..."):
                loaded_queries = load_from_file(queries_file)
                if loaded_queries:
                    st.session_state.queries = loaded_queries
                    st.session_state.last_queries_file = queries_file.name
                    st.success(f"✅ Loaded {len(loaded_queries)} queries")
    
    with tab4:
        st.markdown("**Add queries manually (one per line)**")
        manual_queries = st.text_area(
            "Enter Test Queries",
            height=200,
            placeholder="What is the capital of France?\nExplain quantum computing.\nHow does photosynthesis work?",
            key="manual_queries"
        )
        
        col9, col10 = st.columns([1, 1])
        with col9:
            if st.button("➕ Add Manual Queries", type="primary"):
                if manual_queries.strip():
                    new_queries = [line.strip() for line in manual_queries.split('\n') if line.strip()]
                    st.session_state.queries.extend(new_queries)
                    st.success(f"✅ Added {len(new_queries)} queries")
        
        with col10:
            if st.button("🗑️ Clear All Queries"):
                st.session_state.queries = []
                st.session_state.pop('last_queries_file', None)
                st.success("✅ Cleared all queries")
                st.rerun()
    
    # Display current queries
    if st.session_state.queries:
        st.success(f"**Current Test Queries: {len(st.session_state.queries)}**")
        with st.expander("View All Queries"):
            for i, query in enumerate(st.session_state.queries, 1):
                st.text(f"{i}. {query}")
    else:
        st.warning("⚠️ No test queries loaded")
    
    st.divider()
    
    # Run tests button
    if st.session_state.analysis and st.session_state.system_prompts and st.session_state.queries:
        st.subheader("🚀 Run Tests")
        
        total_tests = len(st.session_state.system_prompts) * len(st.session_state.queries)
        st.info(f"**Ready to run {total_tests} tests** ({len(st.session_state.system_prompts)} prompts × {len(st.session_state.queries)} queries)")
        
        col11, col12, col13 = st.columns([1, 2, 1])
        
        with col12:
            if st.button("▶️ Run All Tests", type="primary", use_container_width=True):
                agent = PromptTestingAgent()
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                test_num = 0
                
                for i, system_prompt in enumerate(st.session_state.system_prompts):
                    for j, query in enumerate(st.session_state.queries):
                        test_num += 1
                        status_text.text(f"Running test {test_num}/{total_tests}...")
                        
                        payload = agent.create_test_payload(
                            st.session_state.analysis,
                            system_prompt,
                            query
                        )
                        
                        response = agent.execute_test(
                            st.session_state.analysis,
                            payload,
                            st.session_state.code_content
                        )
                        
                        results.append({
                            'System Prompt #': i + 1,
                            'System Prompt': system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt,
                            'Query': query,
                            'Response': response[:500] + "..." if len(response) > 500 else response,
                            'Full Response': response
                        })
                        
                        progress_bar.progress(test_num / total_tests)
                
                status_text.text("✅ All tests completed!")
                
                df = pd.DataFrame(results)
                st.session_state.test_results = df
    
    # Display results
    if st.session_state.test_results is not None:
        st.divider()
        st.subheader("📊 Test Results")
        
        df = st.session_state.test_results
        
        # Display summary
        col14, col15, col16 = st.columns(3)
        with col14:
            st.metric("Total Tests", len(df))
        with col15:
            errors = df['Response'].str.contains('ERROR:', case=False, na=False).sum()
            st.metric("Errors", errors)
        with col16:
            success = len(df) - errors
            st.metric("Successful", success)
        
        # Display dataframe (without Full Response column for cleaner view)
        st.dataframe(
            df[['System Prompt #', 'System Prompt', 'Query', 'Response']],
            use_container_width=True,
            height=400
        )
        
        # Download buttons
        col17, col18 = st.columns(2)
        
        with col17:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="prompt_test_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col18:
            # Excel download with BytesIO instead of StringIO
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Test Results')
            excel_buffer.seek(0)
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer,
                file_name="prompt_test_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        # Detailed view
        with st.expander("🔍 View Full Responses"):
            for idx, row in df.iterrows():
                st.markdown(f"**Test {idx + 1}**")
                st.markdown(f"*System Prompt:* {row['System Prompt']}")
                st.markdown(f"*Query:* {row['Query']}")
                st.text_area(f"Response {idx + 1}", row['Full Response'], height=150, key=f"full_response_{idx}")
                st.divider()


if __name__ == "__main__":
    main()