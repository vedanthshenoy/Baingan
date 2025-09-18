# individual.py (patched: uses save_export_entry for all saves, updates export_data live,
# no per-row preview; ratings update dynamically in export preview table)
import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
import time
import os

# Load .env to get GEMINI_API_KEY
from dotenv import load_dotenv
load_dotenv()

# Try to import google.generativeai (Gemini)
try:
    import google.generativeai as genai
except Exception:
    genai = None

from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api
from app.utils import add_result_row
from app.export import save_export_entry

# Load Gemini API key from .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize Gemini if available
gemini_available = False
gemini_model = None
if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        gemini_available = True
    except Exception:
        gemini_model = None
        gemini_available = False


# Default fallback API call function
def _default_call_api_func(system_prompt, query, body_template, headers, response_path):
    return {
        "response": f"call_api not configured. Tried to call with query: {query[:200]}",
        "status": "Failed",
        "status_code": "N/A"
    }


# Default suggest function using Gemini (if available)
def _gemini_suggest_func(response_text, original_query):
    if not gemini_available or not gemini_model:
        return "Gemini is not available. Cannot suggest prompt."

    prompt = f"""
You are given a user query and a response produced by a model.
Query: {original_query}
Response: {response_text}

Your task: Suggest an improved system prompt that would likely produce that kind of response.
Output only the improved system prompt (no extra commentary).
"""
    try:
        result = gemini_model.generate_content(prompt)
        return result.text.strip() if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"Error generating suggestion: {str(e)}"


def render_individual_testing(
    api_url,
    query_text,
    body_template,
    headers,
    response_path,
    call_api_func=None,
    suggest_func=None,
):
    """
    Individual testing UI.
    call_api_func: function(system_prompt, query, body_template, headers, response_path) -> dict
    suggest_func: function(response_text, original_query) -> str
    """

    st.header("🧪 Individual Testing")

    call_api_func = call_api_func or _default_call_api_func
    suggest_func = suggest_func or _gemini_suggest_func

    # Expected schema for both test_results and export_data
    expected_columns = [
        'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
        'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited'
    ]

    # Ensure test_results exists and has expected columns
    if 'test_results' not in st.session_state or not isinstance(st.session_state.test_results, pd.DataFrame):
        st.session_state.test_results = pd.DataFrame(columns=expected_columns)
    else:
        for c in expected_columns:
            if c not in st.session_state.test_results.columns:
                st.session_state.test_results[c] = None

    # Ensure export_data exists and has expected columns (export.py depends on this)
    if 'export_data' not in st.session_state or not isinstance(st.session_state.export_data, pd.DataFrame):
        st.session_state.export_data = pd.DataFrame(columns=expected_columns)
    else:
        for c in expected_columns:
            if c not in st.session_state.export_data.columns:
                st.session_state.export_data[c] = None

    if 'response_ratings' not in st.session_state or not isinstance(st.session_state.response_ratings, dict):
        st.session_state.response_ratings = {}

    if 'prompts' not in st.session_state or not isinstance(st.session_state.prompts, list):
        st.session_state.prompts = []

    if 'prompt_names' not in st.session_state or not isinstance(st.session_state.prompt_names, list):
        st.session_state.prompt_names = []

    # ---- Run all prompts button ----
    can_run_all = bool(api_url) and bool(st.session_state.get('prompts')) and bool(query_text)
    if st.button("🚀 Test All Prompts", type="primary", disabled=not can_run_all):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.get('prompts'):
            st.error("Please add at least one system prompt")
        elif not query_text:
            st.error("Please enter a query")
        else:
            st.subheader("Results")
            with st.spinner("Running tests..."):
                ensure_prompt_names()
                total_prompts = len(st.session_state.prompts)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                    status_text.text(f"Testing {prompt_name} ({i+1}/{total_prompts})...")
                    try:
                        result = call_api_func(
                            system_prompt=system_prompt,
                            query=query_text,
                            body_template=body_template,
                            headers=headers,
                            response_path=response_path
                        ) or {}
                        response_text = result.get('response')
                        status = result.get('status', 'Failed')
                        status_code = str(result.get('status_code', 'N/A'))
                    except Exception as e:
                        response_text = f"Error: {str(e)}"
                        status = 'Failed'
                        status_code = 'N/A'

                    # Save via save_export_entry to ensure export_data is canonical
                    unique_id = save_export_entry(
                        prompt_name=prompt_name,
                        system_prompt=system_prompt,
                        query=query_text,
                        response=response_text,
                        mode="Individual",
                        remark="Saved and ran",
                        status=status,
                        status_code=status_code,
                        rating=0,
                        edited=False
                    )

                    # register default rating for this unique_id
                    st.session_state.response_ratings[unique_id] = 0

                    new_result = pd.DataFrame([{
                        'unique_id': unique_id,
                        'test_type': 'Individual',
                        'prompt_name': prompt_name,
                        'system_prompt': system_prompt,
                        'query': query_text,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': 0,
                        'remark': 'Saved and ran',
                        'edited': False
                    }])
                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                    progress_bar.progress((i + 1) / max(1, total_prompts))
                    time.sleep(1)

                status_text.text("Tests completed!")
                st.success(f"Tested {total_prompts} prompts!")

    # ---- Display Results (expanders) ----
    st.subheader("Saved Results")

    if st.session_state.test_results.empty:
        st.info("No results to display yet. Run some tests first!")
        return

    # Filter for individual tests with responses, without resetting index to preserve original indices
    individual_results = st.session_state.test_results[
        (st.session_state.test_results['test_type'] == 'Individual') &
        st.session_state.test_results['response'].notna()
    ]

    if individual_results.empty:
        st.info("No individual test results to display.")
        return

    # show a simple metric
    try:
        success_count = int(len(individual_results[individual_results['status'] == 'Success']))
        st.metric("Successful Tests", f"{success_count}/{len(individual_results)}")
    except Exception:
        st.metric("Results", f"{len(individual_results)} tests")

    # iterate and show expanders using original indices
    for i, result in individual_results.iterrows():
        unique_id = str(result['unique_id'])
        prompt_name = result.get('prompt_name') or f"Prompt {i+1}"
        status_color = "🟢" if result['status'] == 'Success' else "🔴"
        with st.expander(f"{status_color} {prompt_name} — {result.get('status', 'Unknown')}"):
            st.write("**System Prompt:**")
            st.text(result.get('system_prompt', ''))
            st.write("**Query:**")
            st.text(result.get('query', ''))
            st.write("**Response:**")

            edited_response = st.text_area(
                "Response (editable):",
                value=result['response'] if pd.notnull(result['response']) else "",
                height=150,
                key=f"edit_response_{i}"
            )

            # Rating slider linked to unique_id
            current_rating = st.session_state.response_ratings.get(unique_id, int(result.get('rating', 0) or 0))
            new_rating = st.slider(
                "Rate this response (0-10):",
                min_value=0,
                max_value=10,
                value=int(current_rating),
                key=f"rating_{i}"
            )

            # Update ratings dynamically in DataFrames
            if new_rating != result.get('rating', 0):
                st.session_state.response_ratings[unique_id] = new_rating
                st.session_state.test_results.at[i, 'rating'] = new_rating
                st.session_state.test_results.at[i, 'edited'] = True
                if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                    export_mask = st.session_state.export_data['unique_id'] == unique_id
                    if export_mask.any():
                        st.session_state.export_data.loc[export_mask, 'rating'] = new_rating
                        st.session_state.export_data.loc[export_mask, 'edited'] = True
                # refresh UI so slider reflects everywhere
                st.rerun()

            if edited_response != (result['response'] or ""):
                if st.button("💾 Save Edited Response", key=f"save_edited_{i}"):
                    st.session_state.test_results.at[i, 'response'] = edited_response
                    st.session_state.test_results.at[i, 'edited'] = True

                    # save edited response to export and capture unique_id
                    saved_unique_id = save_export_entry(
                        prompt_name=prompt_name,
                        system_prompt=result.get('system_prompt', ''),
                        query=result.get('query', ''),
                        response=edited_response,
                        mode="Individual",
                        remark="Edited and saved",
                        status=result.get('status', 'Unknown'),
                        status_code=result.get('status_code', 'N/A'),
                        rating=new_rating,
                        edited=True
                    )

                    # register rating for the saved_unique_id
                    st.session_state.response_ratings[saved_unique_id] = new_rating

                    # update the test_results row with the returned unique_id
                    st.session_state.test_results.at[i, 'unique_id'] = saved_unique_id
                    st.session_state.test_results.at[i, 'remark'] = 'Edited and saved'
                    st.success("Edited response saved.")
                    st.rerun()

            # Suggest Prompt (Gemini-assisted)
            suggest_disabled = not gemini_available
            if st.button("🔮 Suggest Prompt for This Response", key=f"suggest_btn_{i}", disabled=suggest_disabled):
                with st.spinner("Generating prompt suggestion..."):
                    suggestion = suggest_func(edited_response if edited_response else (result['response'] or ""), result['query'])
                    st.session_state[f"suggested_prompt_{i}"] = suggestion
                    st.session_state[f"suggested_prompt_name_{i}"] = f"Suggested Prompt {len(st.session_state.get('prompts', [])) + 1}"

            # If suggestion exists, show save / save & run UI
            if st.session_state.get(f"suggested_prompt_{i}"):
                st.write("**Suggested System Prompt:**")
                st.text_area("Suggested Prompt:", value=st.session_state[f"suggested_prompt_{i}"], height=120, key=f"suggested_display_{i}", disabled=True)

                prompt_name_input = st.text_input(
                    "Name this suggested prompt:",
                    value=st.session_state.get(f"suggested_prompt_name_{i}", f"Suggested Prompt {i+1}"),
                    key=f"suggested_name_input_{i}"
                )

                c1, c2 = st.columns(2)

                # Save only
                with c1:
                    if st.button("💾 Save Prompt", key=f"save_suggest_{i}"):
                        suggested_prompt = st.session_state.get(f"suggested_prompt_{i}")
                        saved_name = prompt_name_input or f"Suggested Prompt {i+1}"
                        # Save to export_data via save_export_entry (Not executed)
                        saved_unique_id = save_export_entry(
                            prompt_name=saved_name,
                            system_prompt=suggested_prompt,
                            query=result.get('query', ''),
                            response='Prompt saved but not executed',
                            mode='Individual',
                            remark='Saved only',
                            status='Not Executed',
                            status_code='N/A',
                            rating=0,
                            edited=False
                        )

                        # register rating default for this new row
                        st.session_state.response_ratings[saved_unique_id] = 0

                        # add to prompts list
                        st.session_state.prompts.append(suggested_prompt)
                        st.session_state.prompt_names.append(saved_name)

                        # append to test_results
                        new_result = pd.DataFrame([{
                            'unique_id': saved_unique_id,
                            'test_type': 'Individual',
                            'prompt_name': saved_name,
                            'system_prompt': suggested_prompt,
                            'query': result.get('query', ''),
                            'response': 'Prompt saved but not executed',
                            'status': 'Not Executed',
                            'status_code': 'N/A',
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'rating': 0,
                            'remark': 'Saved only',
                            'edited': False
                        }])
                        st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                        # clear suggestion state
                        del st.session_state[f"suggested_prompt_{i}"]
                        del st.session_state[f"suggested_prompt_name_{i}"]
                        st.success(f"Saved suggested prompt: {saved_name}")
                        st.rerun()

                # Save & Run
                with c2:
                    if st.button("🏃 Save & Run Prompt", key=f"save_run_suggest_{i}"):
                        suggested_prompt = st.session_state.get(f"suggested_prompt_{i}")
                        saved_name = prompt_name_input or f"Suggested Prompt {i+1}"
                        with st.spinner("Running suggested prompt..."):
                            try:
                                run_result = call_api_func(
                                    system_prompt=suggested_prompt,
                                    query=result.get('query', ''),
                                    body_template=body_template,
                                    headers=headers,
                                    response_path=response_path
                                ) or {}
                                response_text = run_result.get('response')
                                status = run_result.get('status', 'Failed')
                                status_code = str(run_result.get('status_code', 'N/A'))
                            except Exception as e:
                                response_text = f"Error: {str(e)}"
                                status = 'Failed'
                                status_code = 'N/A'

                            # Save to export_data via save_export_entry
                            saved_unique_id = save_export_entry(
                                prompt_name=saved_name,
                                system_prompt=suggested_prompt,
                                query=result.get('query', ''),
                                response=response_text,
                                mode='Individual',
                                remark='Saved and ran',
                                status=status,
                                status_code=status_code,
                                rating=0,
                                edited=False
                            )

                            # register rating default for this new row
                            st.session_state.response_ratings[saved_unique_id] = 0

                            # add to prompts list
                            st.session_state.prompts.append(suggested_prompt)
                            st.session_state.prompt_names.append(saved_name)

                            # append to test_results
                            new_result = pd.DataFrame([{
                                'unique_id': saved_unique_id,
                                'test_type': 'Individual',
                                'prompt_name': saved_name,
                                'system_prompt': suggested_prompt,
                                'query': result.get('query', ''),
                                'response': response_text,
                                'status': status,
                                'status_code': status_code,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'rating': 0,
                                'remark': 'Saved and ran',
                                'edited': False
                            }])
                            st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)

                        st.success(f"Saved and ran suggested prompt: {saved_name}")

                        # show immediate response and rating slider
                        st.write("**Response from Suggested Prompt:**")
                        st.text_area("Response:", value=response_text or "", height=150, key=f"suggested_run_resp_{i}")
                        rating_val = st.slider(
                            "Rate this response (0-10):",
                            min_value=0,
                            max_value=10,
                            value=0,
                            key=f"rating_suggested_{i}"
                        )
                        if rating_val != 0:
                            st.session_state.response_ratings[saved_unique_id] = rating_val
                            # Update the new row, but since it's appended, we need to find its index
                            new_index = st.session_state.test_results.index[-1]
                            st.session_state.test_results.at[new_index, 'rating'] = rating_val
                            export_mask = st.session_state.export_data['unique_id'] == saved_unique_id
                            st.session_state.export_data.loc[export_mask, 'rating'] = rating_val
                            st.rerun()

                        # clear suggestion state
                        del st.session_state[f"suggested_prompt_{i}"]
                        del st.session_state[f"suggested_prompt_name_{i}"]
                        st.rerun()

            # details footer
            st.write("**Details:**")
            st.write(
                f"Status Code: {result.get('status_code', 'N/A')} | "
                f"Time: {result.get('timestamp', 'N/A')} | "
                f"Rating: {st.session_state.response_ratings.get(unique_id, result.get('rating', 0))}/10"
            )

    # Note: export preview table is managed by app.export.render_export_section()
    # We intentionally do NOT render another dataframe here to avoid duplication.