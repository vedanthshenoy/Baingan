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

    # Show chain setup (reorder only, no prompts table)
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

                # Default rating 0
                st.session_state.response_ratings[unique_id] = 0

                new_result = pd.DataFrame([{
                    'unique_id': unique_id,
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
                    'input_query': current_query
                }])

                # Save all steps into test_results
                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                # Save step into export_data (without unique_id)
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
                    rating=0
                )

                # Inject unique_id manually into last row of export_data
                if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                    last_index = st.session_state.export_data.index[-1]
                    st.session_state.export_data.at[last_index, 'unique_id'] = unique_id

                if result['status'] == 'Success':
                    current_query = result['response']
                else:
                    st.warning(f"Step {i+1} failed: {result['response']}. Continuing with previous query...")

                progress_bar.progress((i + 1) / total_steps)

            status_text.text("Chain execution completed!")
            st.success(f"Executed {total_steps} chain steps!")

    # Show chaining results in textual format
    if not st.session_state.test_results.empty:
        st.subheader("🔗 Chain Results")
        success_count = len(st.session_state.test_results[st.session_state.test_results['status'] == 'Success'])
        st.metric("Successful Chain Steps", f"{success_count}/{len(st.session_state.test_results)}")

        for index, row in st.session_state.test_results.iterrows():
            st.markdown(f"**Step {row.get('step', 'N/A')}: {row['prompt_name']}**")
            st.write(f"- **Input Query**: {row.get('input_query', 'N/A')}")
            st.write(f"- **Response**: {row.get('response', 'N/A')}")
            st.write(f"- **Status**: {row.get('status', 'N/A')}")
            st.write(f"- **Status Code**: {row.get('status_code', 'N/A')}")
            st.write(f"- **Timestamp**: {row.get('timestamp', 'N/A')}")
            st.write(f"- **Rating**: {row.get('rating', 0)}")
            st.write(f"- **Remark**: {row.get('remark', 'N/A')}")
            st.markdown("---")

        # Allow rating updates for complete rows
        st.subheader("Rate Responses")
        for index, row in st.session_state.test_results.iterrows():
            if pd.notnull(row['response']) and pd.notnull(row['status']):
                unique_id = row['unique_id']
                rating_key = f"rating_{unique_id}"
                current_rating = st.session_state.response_ratings.get(unique_id, row.get('rating', 0))
                new_rating = st.slider(
                    f"Rate response for {row['prompt_name']} (Step {row.get('step', 'N/A')})",
                    min_value=0, max_value=5, value=int(current_rating), key=rating_key
                )
                if new_rating != row.get('rating', 0):
                    st.session_state.response_ratings[unique_id] = new_rating
                    # Update rating in test_results
                    st.session_state.test_results.loc[index, 'rating'] = new_rating
                    st.session_state.test_results.loc[index, 'edited'] = True
                    # Update rating in export_data for matching unique_id
                    if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == unique_id, 'rating'
                        ] = new_rating
                        st.session_state.export_data.loc[
                            st.session_state.export_data['unique_id'] == unique_id, 'edited'
                        ] = True
                    st.rerun()
