import streamlit as st
import pandas as pd
import uuid
from datetime import datetime

def add_result_row(
    test_type,
    prompt_name,
    system_prompt,
    query,
    response,
    status,
    status_code,
    remark,
    rating=0,
    edited=False,
    step=None,
    input_query=None,
    combination_strategy=None,
    combination_temperature=None,
    user_name="Unknown"
):
    """Adds a new row to the test_results DataFrame with a consistent schema."""
    if 'test_results' not in st.session_state or not isinstance(st.session_state.test_results, pd.DataFrame):
        # Define a consistent schema with all possible columns
        st.session_state.test_results = pd.DataFrame(columns=[
            'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query',
            'response', 'status', 'status_code', 'timestamp', 'rating',
            'remark', 'edited', 'step', 'input_query',
            'combination_strategy', 'combination_temperature'
        ])

    unique_id = f"{test_type}_{prompt_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"

    new_entry = pd.DataFrame([{
        'user_name': user_name,
        'unique_id': unique_id,
        'test_type': test_type,
        'prompt_name': prompt_name,
        'system_prompt': system_prompt,
        'query': query,
        'response': response,
        'status': status,
        'status_code': status_code,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'rating': rating,
        'remark': remark,
        'edited': edited,
        'step': step,
        'input_query': input_query,
        'combination_strategy': combination_strategy,
        'combination_temperature': combination_temperature,
    }])

    st.session_state.test_results = pd.concat([st.session_state.test_results, new_entry], ignore_index=True)