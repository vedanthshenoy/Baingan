import streamlit as st
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import uuid
import time
from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response
from app.utils import add_result_row
from app.export import save_export_entry

def render_prompt_chaining(api_url, query_text, body_template, headers, response_path, call_api_func, suggest_func, gemini_api_key=''):
    st.header("🔗 Prompt Chaining Testing")

    # Initialize export_data if missing
    if 'export_data' not in st.session_state or not isinstance(st.session_state.export_data, pd.DataFrame):
        st.session_state.export_data = pd.DataFrame(columns=[
            'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
            'status', 'status_code', 'timestamp', 'edited', 'step', 'input_query',
            'combination_strategy', 'combination_temperature', 'slider_weights',
            'rating', 'remark'
        ])

    # Session state cleanup
    if 'response_ratings' not in st.session_state:
        st.session_state.response_ratings = {}
    if 'test_results' in st.session_state and isinstance(st.session_state.test_results, pd.DataFrame):
        st.session_state.test_results['rating'] = st.session_state.test_results['rating'].fillna(0).astype(int)
        st.session_state.test_results = st.session_state.test_results[
            st.session_state.test_results['response'].notnull() & st.session_state.test_results['status'].notnull()
        ].reset_index(drop=True)

    # Helper to normalize uid
    def normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=None):
        if isinstance(maybe_uid, str) and maybe_uid:
            uid = maybe_uid
        else:
            uid = generated_uid or f"Chaining_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}"

        try:
            exists = uid in st.session_state.export_data.get('unique_id', pd.Series(dtype="object")).values
        except Exception:
            exists = False

        if not exists:
            row = export_row_dict.copy()
            row['unique_id'] = uid
            if 'timestamp' not in row:
                row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for col in st.session_state.export_data.columns:
                if col not in row:
                    row[col] = None
            st.session_state.export_data = pd.concat([st.session_state.export_data, pd.DataFrame([row])], ignore_index=True)

        if uid not in st.session_state.response_ratings:
            if generated_uid and generated_uid in st.session_state.response_ratings:
                st.session_state.response_ratings[uid] = st.session_state.response_ratings.pop(generated_uid)
            else:
                st.session_state.response_ratings[uid] = export_row_dict.get('rating', 0) or 0

        return uid

    # Debug: Log gemini_api_key and api_url
    st.write(f"Debug: Gemini API Key available: {bool(gemini_api_key)}")
    st.write(f"Debug: API URL: {api_url}")

    # Test All Prompts in Chain
    if st.button("🚀 Test Chained Prompts", type="primary", disabled=not (api_url and st.session_state.get('prompts') and query_text)):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.get('prompts'):
            st.error("Please add at least one system prompt")
        elif not query_text:
            st.error("Please enter a query")
        else:
            st.subheader("Chained Results")
            with st.spinner("Running chained tests..."):
                ensure_prompt_names()
                total_prompts = len(st.session_state.prompts)
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Debug: Log prompts and names
                st.write(f"Debug: Prompts={st.session_state.prompts}, Names={st.session_state.prompt_names}")
                st.write(f"Debug: Prompt count={len(st.session_state.prompts)}, Unique prompts={len(set(st.session_state.prompts))}")

                current_query = query_text  # Start with initial query
                for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                    status_text.text(f"Testing {prompt_name} (Step {i+1}/{total_prompts})...")
                    st.write(f"Debug: Testing Prompt '{prompt_name}' with query '{current_query}'")

                    try:
                        result = call_api_func(
                            system_prompt=system_prompt,
                            query=current_query,
                            body_template=body_template,
                            headers=headers,
                            response_path=response_path
                        )
                        response_text = result.get('response', None)
                        status = result.get('status', 'Failed')
                        status_code = str(result.get('status_code', 'N/A'))
                    except Exception as e:
                        st.error(f"Error in API call for {prompt_name}: {str(e)}")
                        response_text = f"Error: {str(e)}"
                        status = 'Failed'
                        status_code = 'N/A'

                    st.write(f"Debug: Result for '{prompt_name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                    export_row_dict = {
                        'test_type': 'Chaining',
                        'prompt_name': prompt_name,
                        'system_prompt': system_prompt,
                        'query': current_query,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': 0,
                        'remark': f'Chained step {i+1}',
                        'edited': False,
                        'step': i + 1,
                        'input_query': query_text,
                        'combination_strategy': None,
                        'combination_temperature': None,
                        'slider_weights': None
                    }

                    maybe_uid = save_export_entry(
                        prompt_name=prompt_name,
                        system_prompt=system_prompt,
                        query=current_query,
                        response=response_text,
                        mode="Chaining",
                        remark=f"Chained step {i+1}",
                        status=status,
                        status_code=status_code,
                        rating=0,
                        step=i + 1,
                        input_query=query_text
                    )

                    generated_uid = f"Chaining_{prompt_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                    unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                    add_result_row(
                        test_type='Chaining',
                        prompt_name=prompt_name,
                        system_prompt=system_prompt,
                        query=current_query,
                        response=response_text,
                        status=status,
                        status_code=status_code,
                        remark=f'Chained step {i+1}',
                        rating=0,
                        edited=False,
                        step=i + 1,
                        input_query=query_text,
                        combination_strategy=None,
                        combination_temperature=None
                    )

                    last_index = st.session_state.test_results.index[-1]
                    if st.session_state.test_results.at[last_index, 'unique_id'] != unique_id:
                        st.session_state.test_results.at[last_index, 'unique_id'] = unique_id

                    st.session_state.response_ratings[unique_id] = 0
                    current_query = response_text if response_text else current_query  # Use response as next query
                    progress_bar.progress((i + 1) / total_prompts)

                    # Add delay to avoid rate-limiting
                    time.sleep(1)

                status_text.text("Chained tests completed!")
                st.success(f"Tested {total_prompts} chained prompts!")
                st.rerun()

    # Display Results
    st.subheader("Saved Chained Results")
    if 'test_results' not in st.session_state or st.session_state.test_results.empty:
        st.info("No results to display yet. Run some chained tests first!")
    else:
        display_df = st.session_state.test_results[
            (st.session_state.test_results['test_type'] == 'Chaining') &
            st.session_state.test_results['response'].notna()
        ].copy()

        if display_df.empty:
            st.info("No chained test results to display.")
        else:
            success_count = len(display_df[display_df['status'] == 'Success'])
            st.metric("Successful Chained Tests", f"{success_count}/{len(display_df)}")

            for i, result in display_df.sort_values('step').iterrows():
                unique_id = result['unique_id']
                prompt_name = result['prompt_name']
                step = result['step']
                with st.expander(f"{'🟢' if result['status'] == 'Success' else '🔴'} Step {step}: {prompt_name} - {result['status']}"):
                    st.write("**System Prompt:**")
                    st.text(result['system_prompt'])
                    st.write("**Query:**")
                    st.text(result['query'])
                    st.write("**Response:**")

                    edited_response = st.text_area(
                        "Response (editable):",
                        value=result['response'] if pd.notnull(result['response']) else "",
                        height=150,
                        key=f"edit_response_{unique_id}"
                    )

                    # Rating slider
                    current_rating = st.session_state.response_ratings.get(unique_id, int(result.get('rating', 0) or 0))
                    new_rating = st.slider(
                        "Rate this response (0-10):",
                        min_value=0,
                        max_value=10,
                        value=int(current_rating),
                        key=f"rating_{unique_id}"
                    )

                    if new_rating != current_rating:
                        st.session_state.response_ratings[unique_id] = new_rating
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = new_rating
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                        if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                            st.session_state.export_data.loc[
                                st.session_state.export_data['unique_id'] == unique_id, 'rating'
                            ] = new_rating
                            st.session_state.export_data.loc[
                                st.session_state.export_data['unique_id'] == unique_id, 'edited'
                            ] = True
                        st.rerun()

                    if edited_response != (result['response'] or ""):
                        col_save, col_reverse = st.columns(2)
                        with col_save:
                            if st.button(f"💾 Save Edited Response", key=f"save_response_{unique_id}"):
                                export_row_dict = {
                                    'test_type': 'Chaining',
                                    'prompt_name': prompt_name,
                                    'system_prompt': result['system_prompt'],
                                    'query': result['query'],
                                    'response': edited_response,
                                    'status': result['status'],
                                    'status_code': result['status_code'],
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': new_rating,
                                    'remark': f'Edited chained step {step}',
                                    'edited': True,
                                    'step': step,
                                    'input_query': result['input_query'],
                                    'combination_strategy': None,
                                    'combination_temperature': None,
                                    'slider_weights': None
                                }

                                maybe_uid = save_export_entry(
                                    prompt_name=prompt_name,
                                    system_prompt=result['system_prompt'],
                                    query=result['query'],
                                    response=edited_response,
                                    mode="Chaining",
                                    remark=f"Edited chained step {step}",
                                    status=result['status'],
                                    status_code=result['status_code'],
                                    rating=new_rating,
                                    edited=True,
                                    step=step,
                                    input_query=result['input_query']
                                )

                                saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=unique_id)
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = edited_response
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = f'Edited chained step {step}'
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                st.session_state.response_ratings[saved_unique_id] = new_rating
                                if saved_unique_id != unique_id:
                                    st.session_state.response_ratings.pop(unique_id, None)
                                st.success("Response updated!")
                                st.rerun()

                        with col_reverse:
                            if st.button(f"🔄 Reverse Prompt", key=f"reverse_{unique_id}", disabled=not gemini_api_key):
                                with st.spinner("Generating updated prompt..."):
                                    try:
                                        genai.configure(api_key=gemini_api_key)
                                        suggestion = suggest_func(edited_response, result['query'])
                                        export_row_dict = {
                                            'test_type': 'Chaining',
                                            'prompt_name': prompt_name,
                                            'system_prompt': suggestion,
                                            'query': result['query'],
                                            'response': edited_response,
                                            'status': result['status'],
                                            'status_code': result['status_code'],
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': new_rating,
                                            'remark': f'Reverse prompt for chained step {step}',
                                            'edited': True,
                                            'step': step,
                                            'input_query': result['input_query'],
                                            'combination_strategy': None,
                                            'combination_temperature': None,
                                            'slider_weights': None
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=prompt_name,
                                            system_prompt=suggestion,
                                            query=result['query'],
                                            response=edited_response,
                                            mode="Chaining",
                                            remark=f"Reverse prompt for chained step {step}",
                                            status=result['status'],
                                            status_code=result['status_code'],
                                            rating=new_rating,
                                            edited=True,
                                            step=step,
                                            input_query=result['input_query']
                                        )

                                        saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=unique_id)
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'system_prompt'] = suggestion
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = f'Reverse prompt for chained step {step}'
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                        st.session_state.response_ratings[saved_unique_id] = new_rating
                                        if saved_unique_id != unique_id:
                                            st.session_state.response_ratings.pop(unique_id, None)
                                        st.success("Prompt updated based on edited response!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error generating reverse prompt: {str(e)}")

                    if st.button(f"🔮 Suggest Prompt for This Response", key=f"suggest_btn_{unique_id}", disabled=not gemini_api_key):
                        with st.spinner("Generating prompt suggestion..."):
                            try:
                                genai.configure(api_key=gemini_api_key)
                                suggestion = suggest_func(edited_response if edited_response else (result['response'] or ""), result['query'])
                                st.session_state[f"suggested_prompt_{unique_id}"] = suggestion
                                st.session_state[f"suggested_prompt_name_{unique_id}"] = f"Suggested Prompt Step {step}"
                                st.write("**Suggested System Prompt:**")
                                st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_{unique_id}", disabled=True)
                            except Exception as e:
                                st.error(f"Error generating suggestion: {str(e)}")

                    if st.session_state.get(f"suggested_prompt_{unique_id}"):
                        col_save, col_save_run, col_edit = st.columns(3)
                        with col_save:
                            prompt_name = st.text_input(
                                "Prompt Name:",
                                value=st.session_state[f"suggested_prompt_name_{unique_id}"],
                                key=f"suggest_name_{unique_id}"
                            )
                            if st.button("💾 Save as Prompt", key=f"save_suggest_{unique_id}"):
                                if prompt_name.strip():
                                    export_row_dict = {
                                        'test_type': 'Chaining',
                                        'prompt_name': prompt_name.strip(),
                                        'system_prompt': st.session_state[f"suggested_prompt_{unique_id}"],
                                        'query': result['query'],
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': f'Save only for step {step}',
                                        'edited': False,
                                        'step': step,
                                        'input_query': result['input_query'],
                                        'combination_strategy': None,
                                        'combination_temperature': None,
                                        'slider_weights': None
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=prompt_name.strip(),
                                        system_prompt=st.session_state[f"suggested_prompt_{unique_id}"],
                                        query=result['query'],
                                        response='Prompt saved but not executed',
                                        mode='Chaining',
                                        remark=f"Save only for step {step}",
                                        status='Not Executed',
                                        status_code='N/A',
                                        rating=0,
                                        step=step,
                                        input_query=result['input_query']
                                    )

                                    generated_uid = f"Chaining_{prompt_name.strip()}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                    add_result_row(
                                        test_type='Chaining',
                                        prompt_name=prompt_name.strip(),
                                        system_prompt=st.session_state[f"suggested_prompt_{unique_id}"],
                                        query=result['query'],
                                        response='Prompt saved but not executed',
                                        status='Not Executed',
                                        status_code='N/A',
                                        remark=f'Save only for step {step}',
                                        rating=0,
                                        edited=False,
                                        step=step,
                                        input_query=result['input_query'],
                                        combination_strategy=None,
                                        combination_temperature=None
                                    )

                                    last_index = st.session_state.test_results.index[-1]
                                    if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                        st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                    st.session_state.response_ratings[saved_unique_id] = 0
                                    st.session_state.prompts.append(st.session_state[f"suggested_prompt_{unique_id}"])
                                    st.session_state.prompt_names.append(prompt_name.strip())
                                    del st.session_state[f"suggested_prompt_{unique_id}"]
                                    del st.session_state[f"suggested_prompt_name_{unique_id}"]
                                    st.success(f"Saved as new prompt: {prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                        with col_save_run:
                            run_prompt_name = st.text_input(
                                "Prompt Name:",
                                value=st.session_state[f"suggested_prompt_name_{unique_id}"],
                                key=f"suggest_run_name_{unique_id}"
                            )
                            if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_{unique_id}"):
                                if run_prompt_name.strip():
                                    st.session_state.prompts.append(st.session_state[f"suggested_prompt_{unique_id}"])
                                    st.session_state.prompt_names.append(run_prompt_name.strip())

                                    with st.spinner("Running new prompt..."):
                                        try:
                                            run_result = call_api_func(
                                                system_prompt=st.session_state[f"suggested_prompt_{unique_id}"],
                                                query=result['query'],
                                                body_template=body_template,
                                                headers=headers,
                                                response_path=response_path
                                            )
                                            response_text = run_result.get('response', None)
                                            status = run_result.get('status', 'Failed')
                                            status_code = str(run_result.get('status_code', 'N/A'))
                                        except Exception as e:
                                            st.error(f"Error running suggested prompt: {str(e)}")
                                            response_text = f"Error: {str(e)}"
                                            status = 'Failed'
                                            status_code = 'N/A'

                                        export_row_dict = {
                                            'test_type': 'Chaining',
                                            'prompt_name': run_prompt_name.strip(),
                                            'system_prompt': st.session_state[f"suggested_prompt_{unique_id}"],
                                            'query': result['query'],
                                            'response': response_text,
                                            'status': status,
                                            'status_code': status_code,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': 0,
                                            'remark': f'Saved and ran for step {step}',
                                            'edited': False,
                                            'step': step,
                                            'input_query': result['input_query'],
                                            'combination_strategy': None,
                                            'combination_temperature': None,
                                            'slider_weights': None
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=run_prompt_name.strip(),
                                            system_prompt=st.session_state[f"suggested_prompt_{unique_id}"],
                                            query=result['query'],
                                            response=response_text,
                                            mode='Chaining',
                                            remark=f'Saved and ran for step {step}',
                                            status=status,
                                            status_code=status_code,
                                            rating=0,
                                            step=step,
                                            input_query=result['input_query']
                                        )

                                        generated_uid = f"Chaining_{run_prompt_name.strip()}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                        saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                        add_result_row(
                                            test_type='Chaining',
                                            prompt_name=run_prompt_name.strip(),
                                            system_prompt=st.session_state[f"suggested_prompt_{unique_id}"],
                                            query=result['query'],
                                            response=response_text,
                                            status=status,
                                            status_code=status_code,
                                            remark=f'Saved and ran for step {step}',
                                            rating=0,
                                            edited=False,
                                            step=step,
                                            input_query=result['input_query'],
                                            combination_strategy=None,
                                            combination_temperature=None
                                        )

                                        last_index = st.session_state.test_results.index[-1]
                                        if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                            st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                        st.session_state.response_ratings[saved_unique_id] = 0
                                        del st.session_state[f"suggested_prompt_{unique_id}"]
                                        del st.session_state[f"suggested_prompt_name_{unique_id}"]
                                        st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                        st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                        with col_edit:
                            if st.button("✏️ Edit", key=f"edit_suggest_{unique_id}"):
                                st.session_state[f"edit_suggest_{unique_id}_active"] = True

                            if st.session_state.get(f"edit_suggest_{unique_id}_active", False):
                                edited_suggestion = st.text_area(
                                    "Edit Suggested Prompt:",
                                    value=st.session_state[f"suggested_prompt_{unique_id}"],
                                    height=100,
                                    key=f"edit_suggested_{unique_id}"
                                )
                                edit_prompt_name = st.text_input(
                                    "Prompt Name for Edited Prompt:",
                                    value=st.session_state[f"suggested_prompt_name_{unique_id}"],
                                    key=f"edit_suggest_name_{unique_id}"
                                )
                                if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_{unique_id}"):
                                    if edit_prompt_name.strip():
                                        export_row_dict = {
                                            'test_type': 'Chaining',
                                            'prompt_name': edit_prompt_name.strip(),
                                            'system_prompt': edited_suggestion,
                                            'query': result['query'],
                                            'response': 'Prompt saved but not executed',
                                            'status': 'Not Executed',
                                            'status_code': 'N/A',
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': 0,
                                            'remark': f'Save only for step {step}',
                                            'edited': False,
                                            'step': step,
                                            'input_query': result['input_query'],
                                            'combination_strategy': None,
                                            'combination_temperature': None,
                                            'slider_weights': None
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=result['query'],
                                            response='Prompt saved but not executed',
                                            mode='Chaining',
                                            remark=f"Save only for step {step}",
                                            status='Not Executed',
                                            status_code='N/A',
                                            rating=0,
                                            step=step,
                                            input_query=result['input_query']
                                        )

                                        generated_uid = f"Chaining_{edit_prompt_name.strip()}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                        saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                        add_result_row(
                                            test_type='Chaining',
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=result['query'],
                                            response='Prompt saved but not executed',
                                            status='Not Executed',
                                            status_code='N/A',
                                            remark=f'Save only for step {step}',
                                            rating=0,
                                            edited=False,
                                            step=step,
                                            input_query=result['input_query'],
                                            combination_strategy=None,
                                            combination_temperature=None
                                        )

                                        last_index = st.session_state.test_results.index[-1]
                                        if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                            st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                        st.session_state.response_ratings[saved_unique_id] = 0
                                        st.session_state.prompts.append(edited_suggestion)
                                        st.session_state.prompt_names.append(edit_prompt_name.strip())
                                        st.session_state[f"edit_suggest_{unique_id}_active"] = False
                                        del st.session_state[f"suggested_prompt_{unique_id}"]
                                        del st.session_state[f"suggested_prompt_name_{unique_id}"]
                                        st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                        st.rerun()
                                    else:
                                        st.error("Please provide a prompt name")

                    st.write("**Details:**")
                    st.write(
                        f"Status Code: {result['status_code']} | "
                        f"Time: {result['timestamp']} | "
                        f"Step: {step} | "
                        f"Rating: {st.session_state.response_ratings.get(unique_id, result.get('rating', 0))}/10 "
                        f"({st.session_state.response_ratings.get(unique_id, result.get('rating', 0))*10}%)"
                    )