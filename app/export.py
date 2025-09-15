import streamlit as st
import pandas as pd
import io
import uuid
from datetime import datetime

def save_export_entry(
    prompt_name,
    system_prompt,
    query,
    response,
    mode,
    remark,
    status,
    status_code,
    combination_strategy=None,
    combination_temperature=None,
    slider_weights=None,
    edited=False,
    step=None,
    input_query=None,
    rating=0
):
    if 'export_data' not in st.session_state:
        st.session_state.export_data = pd.DataFrame(columns=[
            'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'edited', 'step', 'input_query', 
            'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark'
        ])
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unique_id = f"{mode}_{prompt_name}_{timestamp}_{uuid.uuid4()}"
    
    new_entry = pd.DataFrame([{
        'unique_id': unique_id,
        'test_type': mode,
        'prompt_name': prompt_name,
        'system_prompt': system_prompt,
        'query': query,
        'response': response,
        'status': status,
        'status_code': str(status_code),
        'timestamp': timestamp,
        'edited': edited,
        'step': step if step is not None else '',
        'input_query': input_query if input_query is not None else '',
        'combination_strategy': combination_strategy if combination_strategy is not None else '',
        'combination_temperature': combination_temperature if combination_temperature is not None else '',
        'slider_weights': str(slider_weights) if slider_weights is not None else '',
        'rating': rating,
        'remark': remark
    }])
    
    st.session_state.export_data = pd.concat([st.session_state.export_data, new_entry], ignore_index=True)
    st.write(f"Added {mode} result: {unique_id}")
    return unique_id

def render_export_section(query_text):
    st.header("📊 Export Results")
    
    if 'export_data' not in st.session_state:
        st.session_state.export_data = pd.DataFrame(columns=[
            'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'edited', 'step', 'input_query', 
            'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark'
        ])
    
    if not st.session_state.export_data.empty:
        st.subheader("📋 DataFrame Preview")
        st.dataframe(st.session_state.export_data, width='stretch')
        
        csv = st.session_state.export_data.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="prompt_results.csv",
            mime="text/csv"
        )
        
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            st.session_state.export_data.to_excel(writer, index=False, sheet_name="Results")
        st.download_button(
            label="📥 Download as Excel",
            data=excel_buffer.getvalue(),
            file_name="prompt_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No results to export yet. Run some tests first!")
    
    if st.button("🗑️ Clear All Results"):
        st.session_state.export_data = pd.DataFrame(columns=[
            'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'edited', 'step', 'input_query', 
            'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark'
        ])
        st.session_state.test_results = pd.DataFrame(columns=[
            'unique_id', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited'
        ])
        st.session_state.chain_results = []
        st.session_state.combination_results = {}
        st.session_state.response_ratings = {}
        st.success("All results cleared!")
        st.rerun()