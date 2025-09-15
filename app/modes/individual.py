import streamlit as st
import google.generativeai as genai
import pandas as pd
from datetime import datetime
from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response
from app.export import save_export_entry

def render_individual_testing(api_url, query_text, body_template, headers, response_path, call_api_func, suggest_func):
    st.header("🧪 Individual Testing")
    
    # Ensure test_results is a DataFrame
    if 'test_results' not in st.session_state or not isinstance(st.session_state.test_results, pd.DataFrame):
        st.session_state.test_results = pd.DataFrame(columns=[
            'unique_id', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited'
        ])
    
    if st.button("🚀 Test All Prompts", type="primary", disabled=not (api_url and st.session_state.prompts and query_text)):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.prompts:
            st.error("Please add at least one system prompt")
        elif not query_text:
            st.error("Please enter a query")
        else:
            ensure_prompt_names()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                status_text.text(f"Testing {prompt_name}...")
                
                result = call_api_func(system_prompt, query_text, body_template, headers, response_path)
                unique_id = save_export_entry(
                    prompt_name=prompt_name,
                    system_prompt=system_prompt,
                    query=query_text,
                    response=result['response'] if 'response' in result else None,
                    mode="Individual",
                    remark="Saved and ran",
                    status=result['status'],
                    status_code=result.get('status_code', 'N/A')
                )
                new_result = pd.DataFrame([{
                    'unique_id': unique_id,
                    'prompt_name': prompt_name,
                    'system_prompt': system_prompt,
                    'query': query_text,
                    'response': result['response'] if 'response' in result else None,
                    'status': result['status'],
                    'status_code': str(result.get('status_code', 'N/A')),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rating': st.session_state.response_ratings.get(f"test_{i}", 0),
                    'remark': 'Saved and ran',
                    'edited': False
                }])
                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                progress_bar.progress((i + 1) / len(st.session_state.prompts))
            
            status_text.text("Testing completed!")
            st.success(f"Tested {len(st.session_state.prompts)} prompts!")

    if not st.session_state.test_results.empty:
        st.subheader("📊 Test Results")
        success_count = sum(1 for _, row in st.session_state.test_results.iterrows() if row['status'] == 'Success')
        st.metric("Successful Tests", f"{success_count}/{len(st.session_state.test_results)}")
        
        for i, result in st.session_state.test_results.iterrows():
            status_color = "🟢" if result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} {result['prompt_name']} - {result['status']}"):
                st.write("**System Prompt:**")
                st.text(result['system_prompt'])
                st.write("**Query:**")
                st.text(result['query'])
                st.write("**Response:**")
                
                edited_response = st.text_area(
                    "Response (editable):", 
                    value=result['response'], 
                    height=150, 
                    key=f"edit_response_{i}"
                )
                
                rating = st.slider(
                    "Rate this response (0-10):",
                    min_value=0,
                    max_value=10,
                    value=int(st.session_state.response_ratings.get(f"test_{i}", 5)),
                    key=f"rating_test_{i}"
                )
                st.session_state.response_ratings[f"test_{i}"] = rating
                
                if edited_response != result['response']:
                    col_save, col_reverse = st.columns(2)
                    with col_save:
                        if st.button(f"💾 Save Edited Response", key=f"save_response_{i}"):
                            st.session_state.test_results.at[i, 'response'] = edited_response
                            st.session_state.test_results.at[i, 'edited'] = True
                            unique_id = save_export_entry(
                                prompt_name=result['prompt_name'],
                                system_prompt=result['system_prompt'],
                                query=query_text,
                                response=edited_response,
                                mode="Individual",
                                remark="Edited and saved",
                                status=result['status'],
                                status_code=result.get('status_code', 'N/A'),
                                edited=True,
                                rating=rating
                            )
                            st.session_state.test_results.at[i, 'unique_id'] = unique_id
                            st.session_state.test_results.at[i, 'remark'] = 'Edited and saved'
                            st.success("Response updated!")
                            st.rerun()
                    with col_reverse:
                        if st.button(f"🔄 Reverse Prompt", key=f"reverse_{i}") and st.session_state.get('gemini_api_key'):
                            with st.spinner("Generating updated prompt..."):
                                genai.configure(api_key=st.session_state.get('gemini_api_key'))
                                suggestion = suggest_func(edited_response, result['query'])
                                st.session_state.prompts[i] = suggestion
                                st.session_state.test_results.at[i, 'system_prompt'] = suggestion
                                st.session_state.test_results.at[i, 'edited'] = True
                                st.session_state.test_results.at[i, 'remark'] = 'Reverse prompt generated'
                                unique_id = save_export_entry(
                                    prompt_name=result['prompt_name'],
                                    system_prompt=suggestion,
                                    query=query_text,
                                    response=edited_response,
                                    mode="Individual",
                                    remark="Reverse prompt generated",
                                    status=result['status'],
                                    status_code=result.get('status_code', 'N/A'),
                                    edited=True,
                                    rating=rating
                                )
                                st.session_state.test_results.at[i, 'unique_id'] = unique_id
                                st.success("Prompt updated based on edited response!")
                                st.rerun()
                
                if st.button(f"🔮 Suggest Prompt for This Response", key=f"suggest_{i}") and st.session_state.get('gemini_api_key'):
                    with st.spinner("Generating prompt suggestion..."):
                        genai.configure(api_key=st.session_state.get('gemini_api_key'))
                        suggestion = suggest_func(edited_response, result['query'])
                        st.session_state[f"suggested_prompt_{i}"] = suggestion
                        st.session_state[f"suggested_prompt_name_{i}"] = f"Suggested Prompt {len(st.session_state.prompts) + 1}"
                        st.write("**Suggested System Prompt:**")
                        st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_{i}", disabled=True)
                
                if st.session_state.get(f"suggested_prompt_{i}"):
                    col_save, col_save_run, col_edit = st.columns(3)
                    with col_save:
                        prompt_name = st.text_input(
                            "Prompt Name:", 
                            value=st.session_state[f"suggested_prompt_name_{i}"], 
                            key=f"suggest_name_{i}"
                        )
                        if st.button("💾 Save as Prompt", key=f"save_suggest_{i}"):
                            if prompt_name.strip():
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_{i}"])
                                st.session_state.prompt_names.append(prompt_name.strip())
                                unique_id = save_export_entry(
                                    prompt_name=prompt_name.strip(),
                                    system_prompt=st.session_state[f"suggested_prompt_{i}"],
                                    query=query_text,
                                    response='Prompt saved but not executed',
                                    mode='Individual',
                                    remark='Save only',
                                    status='Not Executed',
                                    status_code='N/A',
                                    rating=0
                                )
                                new_result = pd.DataFrame([{
                                    'unique_id': unique_id,
                                    'prompt_name': prompt_name.strip(),
                                    'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                    'query': query_text,
                                    'response': 'Prompt saved but not executed',
                                    'status': 'Not Executed',
                                    'status_code': 'N/A',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': 0,
                                    'remark': 'Save only',
                                    'edited': False
                                }])
                                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                del st.session_state[f"suggested_prompt_{i}"]
                                del st.session_state[f"suggested_prompt_name_{i}"]
                                st.success(f"Saved as new prompt: {prompt_name.strip()}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")
                    with col_save_run:
                        run_prompt_name = st.text_input(
                            "Prompt Name:", 
                            value=st.session_state[f"suggested_prompt_name_{i}"], 
                            key=f"suggest_run_name_{i}"
                        )
                        if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_{i}"):
                            if run_prompt_name.strip():
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_{i}"])
                                st.session_state.prompt_names.append(run_prompt_name.strip())
                                with st.spinner("Running new prompt..."):
                                    result = call_api_func(st.session_state[f"suggested_prompt_{i}"], query_text, body_template, headers, response_path)
                                    unique_id = save_export_entry(
                                        prompt_name=run_prompt_name.strip(),
                                        system_prompt=st.session_state[f"suggested_prompt_{i}"],
                                        query=query_text,
                                        response=result['response'] if 'response' in result else None,
                                        mode='Individual',
                                        remark='Saved and ran',
                                        status=result['status'],
                                        status_code=result.get('status_code', 'N/A'),
                                        rating=0
                                    )
                                    new_result = pd.DataFrame([{
                                        'unique_id': unique_id,
                                        'prompt_name': run_prompt_name.strip(),
                                        'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                        'query': query_text,
                                        'response': result['response'] if 'response' in result else None,
                                        'status': result['status'],
                                        'status_code': str(result.get('status_code', 'N/A')),
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Saved and ran',
                                        'edited': False
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                del st.session_state[f"suggested_prompt_{i}"]
                                del st.session_state[f"suggested_prompt_name_{i}"]
                                st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")
                    with col_edit:
                        if st.button("✏️ Edit", key=f"edit_suggest_{i}"):
                            st.session_state[f"edit_suggest_{i}_active"] = True
                        
                        if st.session_state.get(f"edit_suggest_{i}_active", False):
                            edited_suggestion = st.text_area(
                                "Edit Suggested Prompt:", 
                                value=st.session_state[f"suggested_prompt_{i}"], 
                                height=100, 
                                key=f"edit_suggested_{i}"
                            )
                            edit_prompt_name = st.text_input(
                                "Prompt Name for Edited Prompt:", 
                                value=st.session_state[f"suggested_prompt_name_{i}"], 
                                key=f"edit_suggest_name_{i}"
                            )
                            if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_{i}"):
                                if edit_prompt_name.strip():
                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(edit_prompt_name.strip())
                                    unique_id = save_export_entry(
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        mode='Individual',
                                        remark='Save only',
                                        status='Not Executed',
                                        status_code='N/A',
                                        rating=0
                                    )
                                    new_result = pd.DataFrame([{
                                        'unique_id': unique_id,
                                        'prompt_name': edit_prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': query_text,
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Save only',
                                        'edited': False
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                    st.session_state[f"edit_suggest_{i}_active"] = False
                                    del st.session_state[f"suggested_prompt_{i}"]
                                    del st.session_state[f"suggested_prompt_name_{i}"]
                                    st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")
                
                st.write("**Details:**")
                st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']} | Rating: {rating}/10 ({rating*10}%)")