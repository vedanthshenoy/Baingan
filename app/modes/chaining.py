import streamlit as st
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import re
import uuid

from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response
from app.export import save_export_entry


# Utility to sanitize names
def sanitize_prompt_name(name, index):
    if not name or not name.strip() or not re.match(r'^[a-zA-Z0-9\s_-]+$', name):
        return f"Prompt_{index + 1}"
    return name.strip()


def render_prompt_chaining(api_url, query_text, body_template, headers, response_path,
                          call_api_func=call_api, suggest_func=suggest_prompt_from_response):
    st.header("🔗 Prompt Chaining")

    # Initialize test_results with all required columns
    required_columns = [
        'unique_id', 'prompt_name', 'system_prompt', 'query', 'response',
        'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited',
        'step', 'input_query'
    ]
    if 'test_results' not in st.session_state or not isinstance(st.session_state.test_results, pd.DataFrame):
        st.session_state.test_results = pd.DataFrame(columns=required_columns)
    else:
        for col in required_columns:
            if col not in st.session_state.test_results.columns:
                st.session_state.test_results[col] = None
        # keep rating column consistent
        # st.session_state.test_results['rating'] = st.session_state.test_results['rating'].fillna(0).astype(int)
        st.session_state.test_results['rating'] = st.session_state.test_results['rating'].infer_objects(copy=False).fillna(0).astype(int)
        st.session_state.test_results = st.session_state.test_results[
            st.session_state.test_results['response'].notnull() &
            st.session_state.test_results['status'].notnull()
        ].reset_index(drop=True)

    # Initialize export_data if missing
    if 'export_data' not in st.session_state or not isinstance(st.session_state.export_data, pd.DataFrame):
        st.session_state.export_data = pd.DataFrame(columns=[
            'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
            'status', 'status_code', 'timestamp', 'edited', 'step', 'input_query',
            'combination_strategy', 'combination_temperature', 'slider_weights',
            'rating', 'remark'
        ])

    if 'response_ratings' not in st.session_state:
        st.session_state.response_ratings = {}

    # Helper: ensure export_data has an entry for uid, return the uid to use.
    def _normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=None):
        """
        Return a stable uid to use for both test_results and export_data.
        - If maybe_uid (returned by save_export_entry) is provided, use it.
        - Otherwise use generated_uid (if provided) or create a new one.
        Ensure export_data contains a row for uid, and register response_ratings[uid].
        """
        if isinstance(maybe_uid, str) and maybe_uid:
            uid = maybe_uid
        else:
            uid = generated_uid or f"Chain_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}"

        # Ensure st.session_state.export_data contains a row with uid (append if missing)
        try:
            exists = ('export_data' in st.session_state and
                      uid in st.session_state.export_data.get('unique_id', pd.Series(dtype="object")).values)
        except Exception:
            exists = False

        if not exists:
            # create a row based on export_row_dict, but make sure column names match export_data
            row = export_row_dict.copy()
            row['unique_id'] = uid
            # ensure required columns exist in row
            if 'timestamp' not in row:
                row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # append missing columns with None if needed to match export_data
            for col in st.session_state.export_data.columns:
                if col not in row:
                    row[col] = None
            st.session_state.export_data = pd.concat([st.session_state.export_data, pd.DataFrame([row])], ignore_index=True)

        # Ensure a response rating entry exists (transfer from generated uid if present)
        if uid not in st.session_state.response_ratings:
            # transfer
            if generated_uid and generated_uid in st.session_state.response_ratings:
                st.session_state.response_ratings[uid] = st.session_state.response_ratings.pop(generated_uid)
            else:
                st.session_state.response_ratings[uid] = export_row_dict.get('rating', 0) or 0

        return uid

    # Normalize 'step' dtype helper to avoid pyarrow conversion warnings
    def _normalize_step_dtype():
        if 'step' in st.session_state.test_results.columns:
            st.session_state.test_results['step'] = pd.to_numeric(
                st.session_state.test_results['step'], errors='coerce'
            ).astype('Int64')
        if 'step' in st.session_state.export_data.columns:
            st.session_state.export_data['step'] = pd.to_numeric(
                st.session_state.export_data['step'], errors='coerce'
            ).astype('Int64')

    # Show chain setup
    if st.session_state.get('prompts', []):
        ensure_prompt_names()
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

    # Execute chain
    if st.button("⛓️ Execute Chain", type="primary",
                 disabled=not (api_url and st.session_state.get('prompts', []) and query_text)):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.get('prompts', []):
            st.error("Please add at least one system prompt")
        elif not query_text or query_text.isspace():
            st.error("Please enter a valid query")
        else:
            ensure_prompt_names()
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Remove earlier 'Saved and ran' results to avoid duplicates when re-running
            st.session_state.test_results = st.session_state.test_results[
                ~st.session_state.test_results['remark'].str.contains("Saved and ran", na=False)
            ].reset_index(drop=True)

            current_query = query_text.strip()
            total_steps = len(st.session_state.prompts)

            for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                status_text.text(f"Executing step {i+1}: {prompt_name}...")

                # New chaining logic: combine prompt + previous response
                if i == 0:
                    step_input_query = current_query
                else:
                    step_input_query = f"{system_prompt}\n\n{current_query}"

                result = call_api_func(system_prompt, step_input_query, body_template, headers, response_path)

                if i < total_steps - 1:
                    display_name = f"intermediate_response_after_{prompt_name}"
                    mode = "Chain_Intermediate"
                else:
                    display_name = "final_response"
                    mode = "Chain_Final"

                # generate a temporary uid for the row; may be replaced by save_export_entry's uid
                generated_uid = f"Chain_{display_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                # ensure a rating entry for this temporary uid so slider behaves immediately
                st.session_state.response_ratings[generated_uid] = 0

                new_result = pd.DataFrame([{
                    'unique_id': generated_uid,
                    'prompt_name': display_name,
                    'system_prompt': system_prompt,
                    'query': query_text,
                    'response': result['response'] if 'response' in result else None,
                    'status': result['status'],
                    'status_code': str(result.get('status_code', 'N/A')),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rating': 0,
                    'remark': 'Saved and ran',
                    'edited': False,
                    'step': i + 1,
                    'input_query': step_input_query
                }])
                # append to test_results
                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                # remember appended row index (last one)
                appended_index = st.session_state.test_results.index[-1]

                # Build export_row for saving
                export_row_dict = {
                    'prompt_name': display_name,
                    'system_prompt': system_prompt,
                    'query': query_text,
                    'response': result['response'] if 'response' in result else None,
                    'mode': mode,
                    'remark': f"{'Intermediate' if i < total_steps - 1 else 'Final'} chained result",
                    'status': result['status'],
                    'status_code': result.get('status_code', 'N/A'),
                    'step': i + 1,
                    'input_query': step_input_query,
                    'rating': 0,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # try to save and normalize uid
                try:
                    maybe_uid = save_export_entry(**export_row_dict)
                except Exception:
                    maybe_uid = None

                saved_uid = _normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                # If saved_uid differs from the generated uid, transfer rating mapping and update test_results
                if saved_uid != generated_uid:
                    # transfer rating (if any)
                    val = st.session_state.response_ratings.pop(generated_uid, 0)
                    st.session_state.response_ratings[saved_uid] = val
                    # update the test_results row to use saved_uid
                    st.session_state.test_results.at[appended_index, 'unique_id'] = saved_uid
                    st.session_state.test_results.at[appended_index, 'rating'] = val
                else:
                    # ensure export_data row rating is set (best-effort)
                    try:
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == saved_uid, 'rating'
                        ] = st.session_state.response_ratings.get(saved_uid, 0)
                    except Exception:
                        pass

                # normalise step dtype to avoid arrow warnings
                _normalize_step_dtype()

                if result['status'] == 'Success':
                    current_query = result['response']
                else:
                    st.warning(f"Step {i+1} failed: {result.get('response', 'No response')}. Continuing with previous query...")

                progress_bar.progress((i + 1) / total_steps)

            status_text.text("Chain execution completed!")
            st.success(f"Executed {total_steps} chain steps!")

    # Show chaining results
    if not st.session_state.test_results.empty:
        st.subheader("🔗 Chain Results")
        success_count = len(st.session_state.test_results[st.session_state.test_results['status'] == 'Success'])
        st.metric("Successful Chain Steps", f"{success_count}/{len(st.session_state.test_results)}")

        for i, row in st.session_state.test_results.iterrows():
            status_color = "🟢" if row['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} Step {row.get('step', 'N/A')}: {row['prompt_name']}"):
                st.write(f"- **Input Query**: {row.get('input_query', 'N/A')}")
                st.write(f"- **Status**: {row.get('status', 'N/A')}")
                st.write(f"- **Status Code**: {row.get('status_code', 'N/A')}")
                st.write(f"- **Timestamp**: {row.get('timestamp', 'N/A')}")
                st.write(f"- **Remark**: {row.get('remark', 'N/A')}")

                edited_response = st.text_area(
                    "Response (editable):",
                    value=row['response'] if pd.notnull(row['response']) else "",
                    height=150,
                    key=f"edit_response_chain_{i}"
                )

                unique_id = row['unique_id']
                rating_key = f"rating_{unique_id}"
                current_rating = st.session_state.response_ratings.get(unique_id, int(row.get('rating', 0) or 0))
                new_rating = st.slider(
                    f"Rate response for {row['prompt_name']} (Step {row.get('step', 'N/A')})",
                    min_value=0, max_value=10,
                    value=int(current_rating),
                    key=rating_key
                )

                # Write rating to both dataframes immediately
                if new_rating != (row.get('rating', 0) or 0):
                    st.session_state.response_ratings[unique_id] = new_rating
                    # update test_results safe by index
                    try:
                        st.session_state.test_results.at[i, 'rating'] = new_rating
                        st.session_state.test_results.at[i, 'edited'] = True
                    except Exception:
                        # fallback: locate row by unique_id
                        idxs = st.session_state.test_results.index[st.session_state.test_results['unique_id'] == unique_id].tolist()
                        if idxs:
                            st.session_state.test_results.at[idxs[0], 'rating'] = new_rating
                            st.session_state.test_results.at[idxs[0], 'edited'] = True

                    # update export_data by unique_id
                    if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == unique_id, 'rating'
                        ] = new_rating
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == unique_id, 'edited'
                        ] = True

                    st.rerun()

                # Save / Reverse buttons for edited response
                col_save, col_reverse = st.columns(2)
                with col_save:
                    if st.button("💾 Save Edited Response", key=f"save_chain_response_{i}"):
                        saved_uid = save_export_entry(
                            prompt_name=row['prompt_name'],
                            system_prompt=row['system_prompt'],
                            query=row['query'],
                            response=edited_response,
                            mode="Chain_Edit",
                            remark="Edited and saved",
                            status=row['status'],
                            status_code=row.get('status_code', 'N/A'),
                            edited=True,
                            rating=st.session_state.response_ratings.get(unique_id, 0)
                        )
                        final_uid = _normalize_saved_uid(saved_uid, {
                            'prompt_name': row['prompt_name'],
                            'system_prompt': row['system_prompt'],
                            'query': row['query'],
                            'response': edited_response,
                            'rating': st.session_state.response_ratings.get(unique_id, 0),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }, generated_uid=unique_id)
                        # ensure rating mapping uses final uid
                        st.session_state.response_ratings[final_uid] = st.session_state.response_ratings.pop(unique_id, 0)
                        st.session_state.test_results.at[i, 'unique_id'] = final_uid
                        st.session_state.test_results.at[i, 'response'] = edited_response
                        st.session_state.test_results.at[i, 'remark'] = 'Edited and saved'
                        st.success("Response updated!")
                        st.rerun()
                with col_reverse:
                    if st.button("🔄 Reverse Prompt", key=f"reverse_chain_{i}") and st.session_state.get('gemini_api_key'):
                        with st.spinner("Generating updated prompt..."):
                            genai.configure(api_key=st.session_state.get('gemini_api_key'))
                            suggestion = suggest_func(edited_response, row['query'])
                            st.session_state.prompts[i] = suggestion
                            st.session_state.test_results.at[i, 'system_prompt'] = suggestion
                            st.session_state.test_results.at[i, 'edited'] = True
                            st.session_state.test_results.at[i, 'remark'] = 'Reverse prompt generated'

                            saved_uid = save_export_entry(
                                prompt_name=row['prompt_name'],
                                system_prompt=suggestion,
                                query=row['query'],
                                response=edited_response,
                                mode="Chain_Edit",
                                remark="Reverse prompt generated",
                                status=row['status'],
                                status_code=row.get('status_code', 'N/A'),
                                edited=True,
                                rating=st.session_state.response_ratings.get(unique_id, 0)
                            )
                            final_uid = _normalize_saved_uid(saved_uid, {
                                'prompt_name': row['prompt_name'],
                                'system_prompt': suggestion,
                                'query': row['query'],
                                'response': edited_response,
                                'rating': st.session_state.response_ratings.get(unique_id, 0),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }, generated_uid=unique_id)
                            st.session_state.response_ratings[final_uid] = st.session_state.response_ratings.pop(unique_id, 0)
                            st.session_state.test_results.at[i, 'unique_id'] = final_uid
                            st.success("Prompt updated based on edited response!")
                            st.rerun()

                # --- Suggest Prompt button ---
                if st.button("🔮 Suggest Prompt for This Response", key=f"suggest_chain_btn_{i}") and st.session_state.get('gemini_api_key'):
                    with st.spinner("Generating prompt suggestion..."):
                        genai.configure(api_key=st.session_state.get('gemini_api_key'))
                        suggestion = suggest_func(edited_response if edited_response else (row['response'] or ""), row['query'])
                        st.session_state[f"suggested_prompt_{i}"] = suggestion
                        st.session_state[f"suggested_prompt_name_{i}"] = f"Suggested Prompt {len(st.session_state.get('prompts', [])) + 1}"
                        st.write("**Suggested System Prompt:**")
                        st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_chain_{i}", disabled=True)

                # --- Suggestion flows ---
                if st.session_state.get(f"suggested_prompt_{i}"):
                    col_save, col_save_run, col_edit = st.columns(3)
                    with col_save:
                        prompt_name = st.text_input(
                            "Prompt Name:",
                            value=st.session_state[f"suggested_prompt_name_{i}"],
                            key=f"suggest_chain_name_{i}"
                        )
                        if st.button("💾 Save as Prompt", key=f"save_suggest_chain_{i}"):
                            if prompt_name.strip():
                                maybe_uid = save_export_entry(
                                    prompt_name=prompt_name.strip(),
                                    system_prompt=st.session_state[f"suggested_prompt_{i}"],
                                    query=row['query'],
                                    response='Prompt saved but not executed',
                                    mode='Chain_Suggest_Save',
                                    remark='Save only',
                                    status='Not Executed',
                                    status_code='N/A',
                                    rating=0
                                )
                                export_row = {
                                    'prompt_name': prompt_name.strip(),
                                    'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                    'query': row['query'],
                                    'response': 'Prompt saved but not executed',
                                    'rating': 0,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                saved_uid = _normalize_saved_uid(maybe_uid, export_row, generated_uid=None)
                                st.session_state.response_ratings[saved_uid] = 0
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_{i}"])
                                st.session_state.prompt_names.append(prompt_name.strip())
                                new_result = pd.DataFrame([{
                                    'unique_id': saved_uid,
                                    'prompt_name': prompt_name.strip(),
                                    'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                    'query': row['query'],
                                    'response': 'Prompt saved but not executed',
                                    'status': 'Not Executed',
                                    'status_code': 'N/A',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': 0,
                                    'remark': 'Save only',
                                    'edited': False,
                                    'step': None,
                                    'input_query': row.get('input_query', row['query'])
                                }])
                                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                _normalize_step_dtype()
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
                            key=f"suggest_chain_run_name_{i}"
                        )
                        if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_chain_{i}"):
                            if run_prompt_name.strip():
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_{i}"])
                                st.session_state.prompt_names.append(run_prompt_name.strip())
                                with st.spinner("Running new prompt..."):
                                    run_result = call_api_func(st.session_state[f"suggested_prompt_{i}"], row.get('input_query', row['query']), body_template, headers, response_path)
                                    maybe_uid = save_export_entry(
                                        prompt_name=run_prompt_name.strip(),
                                        system_prompt=st.session_state[f"suggested_prompt_{i}"],
                                        query=row.get('input_query', row['query']),
                                        response=run_result['response'] if 'response' in run_result else None,
                                        mode='Chain_Suggest_Run',
                                        remark='Saved and ran',
                                        status=run_result['status'],
                                        status_code=run_result.get('status_code', 'N/A'),
                                        rating=0
                                    )
                                    export_row = {
                                        'prompt_name': run_prompt_name.strip(),
                                        'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                        'query': row.get('input_query', row['query']),
                                        'response': run_result['response'] if 'response' in run_result else None,
                                        'rating': 0,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    saved_uid = _normalize_saved_uid(maybe_uid, export_row, generated_uid=None)
                                    st.session_state.response_ratings[saved_uid] = 0
                                    new_result = pd.DataFrame([{
                                        'unique_id': saved_uid,
                                        'prompt_name': run_prompt_name.strip(),
                                        'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                        'query': row.get('input_query', row['query']),
                                        'response': run_result['response'] if 'response' in run_result else None,
                                        'status': run_result['status'],
                                        'status_code': str(run_result.get('status_code', 'N/A')),
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Saved and ran',
                                        'edited': False,
                                        'step': None,
                                        'input_query': row.get('input_query', row['query'])
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                    _normalize_step_dtype()
                                del st.session_state[f"suggested_prompt_{i}"]
                                del st.session_state[f"suggested_prompt_name_{i}"]
                                st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")
                    with col_edit:
                        if st.button("✏️ Edit", key=f"edit_suggest_chain_{i}"):
                            st.session_state[f"edit_suggest_chain_{i}_active"] = True
                        if st.session_state.get(f"edit_suggest_chain_{i}_active", False):
                            edited_suggestion = st.text_area(
                                "Edit Suggested Prompt:",
                                value=st.session_state[f"suggested_prompt_{i}"],
                                height=100,
                                key=f"edit_suggested_chain_{i}"
                            )
                            edit_prompt_name = st.text_input(
                                "Prompt Name for Edited Prompt:",
                                value=st.session_state[f"suggested_prompt_name_{i}"],
                                key=f"edit_suggest_chain_name_{i}"
                            )
                            if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_chain_{i}"):
                                if edit_prompt_name.strip():
                                    maybe_uid = save_export_entry(
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=row.get('input_query', row['query']),
                                        response='Prompt saved but not executed',
                                        mode='Chain_Suggest_Save',
                                        remark='Save only',
                                        status='Not Executed',
                                        status_code='N/A',
                                        rating=0
                                    )
                                    export_row = {
                                        'prompt_name': edit_prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': row.get('input_query', row['query']),
                                        'response': 'Prompt saved but not executed',
                                        'rating': 0,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    }
                                    saved_uid = _normalize_saved_uid(maybe_uid, export_row, generated_uid=None)
                                    st.session_state.response_ratings[saved_uid] = 0
                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(edit_prompt_name.strip())
                                    new_result = pd.DataFrame([{
                                        'unique_id': saved_uid,
                                        'prompt_name': edit_prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': row.get('input_query', row['query']),
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Save only',
                                        'edited': False,
                                        'step': None,
                                        'input_query': row.get('input_query', row['query'])
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                    _normalize_step_dtype()
                                    st.session_state[f"edit_suggest_chain_{i}_active"] = False
                                    del st.session_state[f"suggested_prompt_{i}"]
                                    del st.session_state[f"suggested_prompt_name_{i}"]
                                    st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                st.markdown("---")
