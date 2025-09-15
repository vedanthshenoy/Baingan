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
        # Ensure all required columns exist in test_results
        for col in required_columns:
            if col not in st.session_state.test_results.columns:
                st.session_state.test_results[col] = None
        # Convert 'rating' column to integer, replacing None with 0
        st.session_state.test_results['rating'] = st.session_state.test_results['rating'].fillna(0).astype(int)
        # Remove incomplete rows (missing 'response' or 'status')
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
                    display_name = f"intermediate_response_after_{prompt_name}"
                    mode = "Chain_Intermediate"
                else:
                    display_name = "final_response"
                    mode = "Chain_Final"

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
                    'rating': int(st.session_state.response_ratings.get(f"chain_{i}", 0)),
                    'remark': 'Saved and ran',
                    'edited': False,
                    'step': i + 1,
                    'input_query': current_query
                }])

                # Save all steps into test_results
                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                # Save all steps into export_data via save_export_entry
                save_export_entry(
                    prompt_name=display_name,
                    system_prompt=system_prompt,
                    query=query_text,
                    response=result['response'] if 'response' in result else None,
                    mode=mode,
                    remark=f"{'Intermediate' if i < total_steps - 1 else 'Final'} chained result",
                    status=result['status'],
                    status_code=result.get('status_code', 'N/A'),
                    step=i + 1,
                    input_query=current_query,
                    rating=int(st.session_state.response_ratings.get(f"chain_{i}", 0))
                )

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

        # Filter and display results
        st.subheader("Filter Results")
        filter_status = st.selectbox("Filter by Status", ["All", "Success", "Failed"], index=0)
        filter_step = st.multiselect(
            "Filter by Step",
            options=sorted(st.session_state.test_results['prompt_name'].unique()),
            default=sorted(st.session_state.test_results['prompt_name'].unique())
        )

        filtered_results = st.session_state.test_results
        if filter_status != "All":
            filtered_results = filtered_results[filtered_results['status'] == filter_status]
        if filter_step:
            filtered_results = filtered_results[filtered_results['prompt_name'].isin(filter_step)]

        # Display results in a table
        if not filtered_results.empty:
            # Define columns to display, ensuring only available columns are used
            display_columns = [
                col for col in [
                    'prompt_name', 'step', 'input_query', 'response', 'status',
                    'status_code', 'timestamp', 'rating', 'remark'
                ] if col in filtered_results.columns
            ]
            try:
                st.dataframe(
                    filtered_results[display_columns].sort_values(by='step' if 'step' in display_columns else 'timestamp'),
                    use_container_width=True
                )

                # Allow rating updates only for complete rows
                for index, row in filtered_results.iterrows():
                    if pd.notnull(row['response']) and pd.notnull(row['status']):
                        rating_key = f"rating_{row['unique_id']}"
                        current_rating = st.session_state.response_ratings.get(rating_key, row.get('rating', 0))
                        # Ensure current_rating is an integer
                        current_rating = int(current_rating) if current_rating is not None else 0
                        new_rating = st.slider(
                            f"Rate response for {row['prompt_name']} (Step {row.get('step', 'N/A')})",
                            min_value=0, max_value=5, value=current_rating, key=rating_key
                        )
                        if new_rating != row.get('rating', 0):
                            st.session_state.response_ratings[rating_key] = new_rating
                            st.session_state.test_results.loc[index, 'rating'] = new_rating
                            st.session_state.test_results.loc[index, 'edited'] = True
            except KeyError as e:
                st.error(f"Error displaying results: {e}. Some columns are missing. Showing available columns.")
                st.dataframe(filtered_results, use_container_width=True)
        else:
            st.info("No results match the selected filters.")