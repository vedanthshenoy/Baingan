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
    if 'test_results' not in st.session_state or not isinstance(st.session_state.test_results, pd.DataFrame):
        st.session_state.test_results = pd.DataFrame(columns=[
            'unique_id', 'prompt_name', 'system_prompt', 'query', 'response',
            'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited',
            'step', 'input_query'
        ])

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

    # Show chain setup
    if st.session_state.get('prompts', []):
        ensure_prompt_names()
        st.write("**Current Chain Order:**")
        for i, name in enumerate(st.session_state.prompt_names):
            st.write(f"**Step {i+1}:** {name}")

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

            # Clear old chain results before running new chain
            st.session_state.test_results = st.session_state.test_results[
                ~st.session_state.test_results['remark'].str.contains("Saved and ran", na=False)
            ].reset_index(drop=True)

            current_query = query_text.strip()
            total_steps = len(st.session_state.prompts)

            for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                status_text.text(f"Executing step {i+1}: {prompt_name}...")

                result = call_api_func(system_prompt, current_query, body_template, headers, response_path)

                # Use Intermediate name for all except final
                if i < total_steps - 1:
                    display_name = f"Intermediate_Prompt_{i+1}"
                else:
                    display_name = prompt_name

                # Unique id for row
                unique_id = f"Chain_{display_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"

                new_result = pd.DataFrame([{
                    'unique_id': unique_id,
                    'prompt_name': display_name,
                    'system_prompt': system_prompt,
                    'query': query_text,
                    'response': result['response'] if 'response' in result else None,
                    'status': result['status'],
                    'status_code': str(result.get('status_code', 'N/A')),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rating': st.session_state.response_ratings.get(f"chain_{i}", 0),
                    'remark': 'Saved and ran',
                    'edited': False,
                    'step': i + 1,
                    'input_query': current_query
                }])

                # Save all steps into test_results
                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                # Save ONLY final step into export_data
                if i == total_steps - 1:
                    save_export_entry(
                        prompt_name=f"Chain_Final_{display_name}",
                        system_prompt=system_prompt,
                        query=query_text,
                        response=result['response'] if 'response' in result else None,
                        mode="Chain_Final",
                        remark="Final chained result",
                        status=result['status'],
                        status_code=result.get('status_code', 'N/A'),
                        step=i + 1,
                        input_query=current_query
                    )
                    st.session_state.export_data = pd.concat([st.session_state.export_data, new_result], ignore_index=True)

                if result['status'] == 'Success':
                    current_query = result['response']
                else:
                    st.warning(f"Step {i+1} failed: {result['response']}. Continuing with previous query...")

                progress_bar.progress((i + 1) / total_steps)

            status_text.text("Chain execution completed!")
            st.success(f"Executed {total_steps} chain steps!")

    # Show results
    if not st.session_state.test_results.empty:
        st.subheader("🔗 Chain Results")
        success_count = len(st.session_state.test_results[st.session_state.test_results['status'] == 'Success'])
        st.metric("Successful Chain Steps", f"{success_count}/{len(st.session_state.test_results)}")

        # Filter final result, handling cases where 'step' might be NaN
        if 'step' in st.session_state.test_results.columns and st.session_state.test_results['step'].notna().any():
            final_result = st.session_state.test_results[
                st.session_state.test_results['step'] == st.session_state.test_results['step'].max()
            ]
        else:
            final_result = pd.DataFrame()  # Fallback if no valid step data

        if not final_result.empty and final_result.iloc[0]['status'] == 'Success':
            st.success("✅ Chain completed successfully!")
            st.subheader("🎯 Final Result")

            final_result = final_result.iloc[0]
            edited_final = st.text_area(
                "Final Output (editable):",
                value=final_result['response'],
                height=150,
                key="edit_final_chain"
            )

            rating = st.slider(
                "Rate this response (0-10):",
                min_value=0,
                max_value=10,
                value=int(st.session_state.response_ratings.get("chain_final", 5)),
                key="rating_chain_final"
            )
            st.session_state.response_ratings["chain_final"] = rating

            if st.button("💾 Save Final Response"):
                last_index = st.session_state.test_results.index[-1]
                st.session_state.test_results.at[last_index, 'response'] = edited_final
                st.session_state.test_results.at[last_index, 'edited'] = True

                unique_id = save_export_entry(
                    prompt_name=f"Chain_Final_{final_result['prompt_name']}",
                    system_prompt=final_result['system_prompt'],
                    query=query_text,
                    response=edited_final,
                    mode="Chain_Final",
                    remark="Edited and saved",
                    status=final_result['status'],
                    status_code=final_result['status_code'],
                    step=final_result['step'],
                    input_query=final_result['input_query'],
                    edited=True,
                    rating=rating
                )
                if unique_id:
                    st.session_state.test_results.at[last_index, 'unique_id'] = unique_id
                    st.session_state.test_results.at[last_index, 'remark'] = 'Edited and saved'
                    # Update export_data with edited final result
                    st.session_state.export_data = pd.concat([st.session_state.export_data,
                                                             st.session_state.test_results.iloc[[last_index]]],
                                                            ignore_index=True)
                    st.success("Final response updated!")

        st.subheader("📋 Step-by-Step Results")
        for i, result in st.session_state.test_results.iterrows():
            status_color = "🟢" if result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} Step {result.get('step', 'N/A')}: {result['prompt_name']} - {result['status']}"):
                st.write("**System Prompt:**")
                st.text(result['system_prompt'])
                st.write("**Input Query:**")
                st.text(result.get('input_query', 'N/A'))
                st.write("**Response:**")
                st.text(result['response'])
                st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']}")