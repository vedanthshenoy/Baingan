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

    if 'response_ratings' not in st.session_state:
        st.session_state.response_ratings = {}

    if st.button("🚀 Test All Prompts", type="primary", disabled=not (api_url and st.session_state.get('prompts') and query_text)):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.get('prompts'):
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

                # save entry and expect a unique_id returned
                unique_id = save_export_entry(
                    prompt_name=prompt_name,
                    system_prompt=system_prompt,
                    query=query_text,
                    response=result['response'] if 'response' in result else None,
                    mode="Individual",
                    remark="Saved and ran",
                    status=result['status'],
                    status_code=result.get('status_code', 'N/A'),
                    rating=0
                )

                # register default rating for this unique_id
                st.session_state.response_ratings[unique_id] = 0

                new_result = pd.DataFrame([{
                    'unique_id': unique_id,
                    'prompt_name': prompt_name,
                    'system_prompt': system_prompt,
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
                progress_bar.progress((i + 1) / len(st.session_state.prompts))

            status_text.text("Testing completed!")
            st.success(f"Tested {len(st.session_state.prompts)} prompts!")

    if not st.session_state.test_results.empty:
        st.subheader("📊 Test Results")
        success_count = sum(1 for _, row in st.session_state.test_results.iterrows() if row['status'] == 'Success')
        st.metric("Successful Tests", f"{success_count}/{len(st.session_state.test_results)}")

        # iterate by index so we can update by location
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
                    value=result['response'] if pd.notnull(result['response']) else "",
                    height=150,
                    key=f"edit_response_{i}"
                )

                # 🔑 Rating slider linked to unique_id
                unique_id = result['unique_id']
                rating_key = f"rating_{unique_id}"
                current_rating = st.session_state.response_ratings.get(unique_id, int(result.get('rating', 0) or 0))
                new_rating = st.slider(
                    "Rate this response (0-10):",
                    min_value=0,
                    max_value=10,
                    value=int(current_rating),
                    key=rating_key
                )

                # Update ratings dynamically in DataFrames
                if new_rating != (result.get('rating', 0) or 0):
                    st.session_state.response_ratings[unique_id] = new_rating
                    st.session_state.test_results.at[i, 'rating'] = new_rating
                    st.session_state.test_results.at[i, 'edited'] = True
                    if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == unique_id, 'rating'
                        ] = new_rating
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == unique_id, 'edited'
                        ] = True
                    # refresh UI so slider reflects everywhere
                    st.rerun()

                if edited_response != (result['response'] or ""):
                    col_save, col_reverse = st.columns(2)
                    with col_save:
                        if st.button(f"💾 Save Edited Response", key=f"save_response_{i}"):
                            st.session_state.test_results.at[i, 'response'] = edited_response
                            st.session_state.test_results.at[i, 'edited'] = True

                            # save edited response to export and capture unique_id
                            saved_unique_id = save_export_entry(
                                prompt_name=result['prompt_name'],
                                system_prompt=result['system_prompt'],
                                query=result['query'],
                                response=edited_response,
                                mode="Individual",
                                remark="Edited and saved",
                                status=result['status'],
                                status_code=result.get('status_code', 'N/A'),
                                edited=True,
                                rating=new_rating
                            )

                            # register rating for the saved_unique_id
                            st.session_state.response_ratings[saved_unique_id] = new_rating

                            # update the test_results row with the returned unique_id
                            st.session_state.test_results.at[i, 'unique_id'] = saved_unique_id
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

                                saved_unique_id = save_export_entry(
                                    prompt_name=result['prompt_name'],
                                    system_prompt=suggestion,
                                    query=result['query'],
                                    response=edited_response,
                                    mode="Individual",
                                    remark="Reverse prompt generated",
                                    status=result['status'],
                                    status_code=result.get('status_code', 'N/A'),
                                    edited=True,
                                    rating=new_rating
                                )

                                # register rating for that unique id
                                st.session_state.response_ratings[saved_unique_id] = new_rating

                                # update the test_results unique id
                                st.session_state.test_results.at[i, 'unique_id'] = saved_unique_id
                                st.success("Prompt updated based on edited response!")
                                st.rerun()

                # Suggest prompt flow
                if st.button(f"🔮 Suggest Prompt for This Response", key=f"suggest_btn_{i}") and st.session_state.get('gemini_api_key'):
                    with st.spinner("Generating prompt suggestion..."):
                        genai.configure(api_key=st.session_state.get('gemini_api_key'))
                        suggestion = suggest_func(edited_response if edited_response else (result['response'] or ""), result['query'])
                        st.session_state[f"suggested_prompt_{i}"] = suggestion
                        st.session_state[f"suggested_prompt_name_{i}"] = f"Suggested Prompt {len(st.session_state.get('prompts', [])) + 1}"
                        st.write("**Suggested System Prompt:**")
                        st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_{i}", disabled=True)

                # If a suggestion exists, show save/run/edit options
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
                                # save suggestion as new prompt (not executed)
                                saved_unique_id = save_export_entry(
                                    prompt_name=prompt_name.strip(),
                                    system_prompt=st.session_state[f"suggested_prompt_{i}"],
                                    query=result['query'],
                                    response='Prompt saved but not executed',
                                    mode='Individual',
                                    remark='Save only',
                                    status='Not Executed',
                                    status_code='N/A',
                                    rating=0
                                )

                                # register rating default for this new row
                                st.session_state.response_ratings[saved_unique_id] = 0

                                # add to prompts list
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_{i}"])
                                st.session_state.prompt_names.append(prompt_name.strip())

                                # add to test_results
                                new_result = pd.DataFrame([{
                                    'unique_id': saved_unique_id,
                                    'prompt_name': prompt_name.strip(),
                                    'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                    'query': result['query'],
                                    'response': 'Prompt saved but not executed',
                                    'status': 'Not Executed',
                                    'status_code': 'N/A',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': 0,
                                    'remark': 'Save only',
                                    'edited': False
                                }])
                                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                                # cleanup
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
                                # add to prompts
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_{i}"])
                                st.session_state.prompt_names.append(run_prompt_name.strip())

                                with st.spinner("Running new prompt..."):
                                    run_result = call_api_func(st.session_state[f"suggested_prompt_{i}"], result['query'], body_template, headers, response_path)

                                    saved_unique_id = save_export_entry(
                                        prompt_name=run_prompt_name.strip(),
                                        system_prompt=st.session_state[f"suggested_prompt_{i}"],
                                        query=result['query'],
                                        response=run_result['response'] if 'response' in run_result else None,
                                        mode='Individual',
                                        remark='Saved and ran',
                                        status=run_result['status'],
                                        status_code=run_result.get('status_code', 'N/A'),
                                        rating=0
                                    )

                                    # register rating default for this new row
                                    st.session_state.response_ratings[saved_unique_id] = 0

                                    new_result = pd.DataFrame([{
                                        'unique_id': saved_unique_id,
                                        'prompt_name': run_prompt_name.strip(),
                                        'system_prompt': st.session_state[f"suggested_prompt_{i}"],
                                        'query': result['query'],
                                        'response': run_result['response'] if 'response' in run_result else None,
                                        'status': run_result['status'],
                                        'status_code': str(run_result.get('status_code', 'N/A')),
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Saved and ran',
                                        'edited': False
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                                # cleanup
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
                                    saved_unique_id = save_export_entry(
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=result['query'],
                                        response='Prompt saved but not executed',
                                        mode='Individual',
                                        remark='Save only',
                                        status='Not Executed',
                                        status_code='N/A',
                                        rating=0
                                    )

                                    # register rating default for this new row
                                    st.session_state.response_ratings[saved_unique_id] = 0

                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(edit_prompt_name.strip())

                                    new_result = pd.DataFrame([{
                                        'unique_id': saved_unique_id,
                                        'prompt_name': edit_prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': result['query'],
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

                # Show details
                st.write("**Details:**")
                st.write(
                    f"Status Code: {result['status_code']} | "
                    f"Time: {result['timestamp']} | "
                    f"Rating: {st.session_state.response_ratings.get(unique_id, result.get('rating', 0))}/10 "
                    f"({st.session_state.response_ratings.get(unique_id, result.get('rating', 0))*10}%)"
                )
