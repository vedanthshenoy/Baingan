# combination.py (patched)
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import pandas as pd
import uuid
import time
from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response
from app.utils import add_result_row
from app.export import save_export_entry
from app.export_with_db import save_export_entry #with db

def debug_log(message):
    """Log debug messages to session state instead of displaying on screen."""
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []
    st.session_state.debug_log.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}")

def render_prompt_combination(api_url, query_text, body_template, headers, response_path, call_api_func, suggest_func, gemini_api_key='', user_name="Unknown"):
    st.header("🤝 Prompt Combination")

    # Initialize export_data if missing
    if 'export_data' not in st.session_state or not isinstance(st.session_state.export_data, pd.DataFrame):
        st.session_state.export_data = pd.DataFrame(
            columns=[
                'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
                'status', 'status_code', 'timestamp', 'edited', 'step',
                'combination_strategy', 'combination_temperature', 'slider_weights',
                'rating', 'remark'
            ]
        ).astype({'step': 'str', 'rating': 'Int64', 'edited': 'bool', 'timestamp': 'str', 'status_code': 'str'})

    # Session state cleanup
    if 'response_ratings' not in st.session_state:
        st.session_state.response_ratings = {}
    if 'test_results' not in st.session_state or not isinstance(st.session_state.test_results, pd.DataFrame):
        st.session_state.test_results = pd.DataFrame(
            columns=[
                'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
                'status', 'status_code', 'timestamp', 'edited', 'step',
                'combination_strategy', 'combination_temperature', 'rating', 'remark'
            ]
        ).astype({'step': 'str', 'rating': 'int', 'edited': 'bool', 'timestamp': 'str', 'status_code': 'str'})
    elif isinstance(st.session_state.test_results, pd.DataFrame):
        # Ensure all expected columns exist
        expected_columns = [
            'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response',
            'status', 'status_code', 'timestamp', 'edited', 'step',
            'combination_strategy', 'combination_temperature', 'rating', 'remark'
        ]
        for col in expected_columns:
            if col not in st.session_state.test_results.columns:
                st.session_state.test_results[col] = None

        # Apply type conversions and fill missing values
        st.session_state.test_results['rating'] = st.session_state.test_results['rating'].astype('Int64')
        st.session_state.test_results['step'] = st.session_state.test_results['step'].astype(str).fillna('')
        st.session_state.test_results['status_code'] = st.session_state.test_results['status_code'].astype(str).fillna('N/A')
        st.session_state.test_results['edited'] = st.session_state.test_results['edited'].astype(bool).fillna(False)
        st.session_state.test_results['timestamp'] = st.session_state.test_results['timestamp'].astype(str).fillna('')
        st.session_state.test_results = st.session_state.test_results[
            st.session_state.test_results['response'].notnull() & st.session_state.test_results['status'].notnull()
        ].reset_index(drop=True)

    # Counter for unique widget keys
    if 'result_counter' not in st.session_state:
        st.session_state.result_counter = 0

    # Helper to normalize uid
    def normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=None):
        if isinstance(maybe_uid, str) and maybe_uid:
            uid = maybe_uid
        else:
            uid = generated_uid or f"Combination_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}"

        try:
            exists = uid in st.session_state.export_data.get('unique_id', pd.Series(dtype="object")).values
        except Exception:
            exists = False

        if not exists:
            row = export_row_dict.copy()
            row['unique_id'] = uid
            if 'timestamp' not in row:
                row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row['step'] = str(row.get('step', ''))  # Ensure step is string
            row['status_code'] = str(row.get('status_code', 'N/A'))
            row['rating'] = row.get('rating')
            row['edited'] = bool(row.get('edited', False))
            for col in st.session_state.export_data.columns:
                if col not in row:
                    row[col] = None
            st.session_state.export_data = pd.concat([st.session_state.export_data, pd.DataFrame([row])], ignore_index=True)

        if uid not in st.session_state.response_ratings:
            if generated_uid and generated_uid in st.session_state.response_ratings:
                st.session_state.response_ratings[uid] = st.session_state.response_ratings.pop(generated_uid)
            else:
                st.session_state.response_ratings[uid] = export_row_dict.get('rating')

        return uid

    if not gemini_api_key:
        st.warning("⚠️ Please configure Gemini API key to use prompt combination")

    temperature = st.slider(
        "🌡️ AI Temperature (Creativity)",
        min_value=0,
        max_value=100,
        value=0,
        help="Controls creativity of AI responses. Lower = more focused, Higher = more creative"
    )
    st.session_state.temperature = temperature

    selected_prompts = []

    if st.session_state.get('prompts'):
        ensure_prompt_names()
        selected_prompts = st.multiselect(
            "Choose prompts to combine:",
            options=list(range(len(st.session_state.prompts))),
            format_func=lambda x: f"{st.session_state.prompt_names[x]}: {st.session_state.prompts[x][:50]}...",
            default=list(range(min(2, len(st.session_state.prompts))))
        )

        if selected_prompts != st.session_state.get('last_selected_prompts', []):
            st.session_state.slider_weights = {}
            st.session_state.last_selected_prompts = selected_prompts

        if selected_prompts:
            st.subheader("Selected Prompts Preview")
            for idx in selected_prompts:
                with st.expander(f"{st.session_state.prompt_names[idx]}"):
                    st.text(st.session_state.prompts[idx])

            combination_strategy = st.selectbox(
                "Combination Strategy:",
                [
                    "Merge and optimize for clarity",
                    "Combine while preserving all instructions",
                    "Create a hierarchical prompt structure",
                    "Synthesize into a concise unified prompt",
                    "Slider - Custom influence weights"
                ]
            )

            if combination_strategy == "Slider - Custom influence weights":
                st.subheader("🎚️ Influence Weights")
                st.write("Set how much influence each prompt should have (0-100%, auto-adjusted to sum to 100%):")

                if not st.session_state.get('slider_weights') or len(st.session_state.slider_weights) != len(selected_prompts):
                    default_weight = 100 // max(1, len(selected_prompts))
                    st.session_state.slider_weights = {idx: default_weight for idx in selected_prompts}
                    if selected_prompts:
                        total = sum(st.session_state.slider_weights.values())
                        if total != 100:
                            st.session_state.slider_weights[selected_prompts[-1]] = 100 - sum(
                                st.session_state.slider_weights.get(i, 0) for i in selected_prompts[:-1]
                            )

                def update_weights(changed_idx):
                    new_value = st.session_state[f"weight_{changed_idx}"]
                    st.session_state.slider_weights[changed_idx] = new_value
                    remaining_weight = 100 - new_value
                    other_indices = [i for i in selected_prompts if i != changed_idx]
                    if other_indices:
                        if len(other_indices) == 1:
                            st.session_state.slider_weights[other_indices[0]] = remaining_weight
                        else:
                            total_other = sum(st.session_state.slider_weights.get(i, 0) for i in other_indices)
                            if total_other > 0:
                                for i in other_indices:
                                    current_weight = st.session_state.slider_weights.get(i, 0)
                                    st.session_state.slider_weights[i] = int((current_weight / total_other) * remaining_weight)
                                total_new = sum(st.session_state.slider_weights.get(i, 0) for i in selected_prompts)
                                if total_new != 100:
                                    st.session_state.slider_weights[other_indices[-1]] += 100 - total_new

                for idx in selected_prompts:
                    weight = st.session_state.slider_weights.get(idx, 100 // max(1, len(selected_prompts)))
                    st.session_state.slider_weights[idx] = weight

                    st.slider(
                        f"{st.session_state.prompt_names[idx]}:",
                        min_value=0,
                        max_value=100,
                        value=weight,
                        key=f"weight_{idx}",
                        on_change=update_weights,
                        args=(idx,)
                    )

                total_weight = sum(st.session_state.slider_weights.get(idx, 0) for idx in selected_prompts)
                st.write(f"**Total Weight:** {total_weight}%")
                if total_weight != 100:
                    st.warning("Weights adjusted to sum to 100%")
    else:
        st.info("Add system prompts first to combine them")

    # Placeholder for combined prompt
    combined_prompt_container = st.container()
    # Placeholder for individual results
    individual_results_container = st.container()
    # Placeholder for combined result
    combined_result_container = st.container()
    # Placeholder for suggested results
    suggested_results_container = st.container()

    # Combined button to combine and test
    if st.button("🧪 Combine and Test Prompts", type="primary", disabled=not (gemini_api_key and selected_prompts and api_url and query_text)):
        if not gemini_api_key:
            st.error("Please configure Gemini API key")
        elif not selected_prompts:
            st.error("Please select at least 2 prompts to combine")
        elif combination_strategy == "Slider - Custom influence weights" and sum(st.session_state.slider_weights.get(idx, 0) for idx in selected_prompts) == 0:
            st.error("Please set at least one prompt weight > 0%")
        elif not api_url or not query_text:
            st.error("Please configure API endpoint and enter a query")
        else:
            # Reset counter for new run
            st.session_state.result_counter = 0

            # Initialize progress bar
            progress_bar = st.progress(0)
            total_steps = len(selected_prompts) + 2  # Combine + individual prompts + combined prompt
            current_step = 0

            # Step 1: Combine Prompts
            with st.spinner("AI is combining prompts..."):
                try:
                    genai.configure(api_key=gemini_api_key)
                    gemini_temperature = (temperature / 100.0) * 2.0
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    selected_prompt_texts = [st.session_state.prompts[i] for i in selected_prompts]
                    selected_prompt_names = [st.session_state.prompt_names[i] for i in selected_prompts]

                    if combination_strategy == "Slider - Custom influence weights":
                        total_weight = sum(st.session_state.slider_weights.get(idx, 0) for idx in selected_prompts)
                        if total_weight == 0:
                            total_weight = 1
                        weighted_prompts = []
                        for idx in selected_prompts:
                            weight = st.session_state.slider_weights.get(idx, 0)
                            percentage = (weight / total_weight) * 100
                            weighted_prompts.append(
                                f"Prompt '{st.session_state.prompt_names[idx]}' (Influence: {percentage:.1f}%):\n{st.session_state.prompts[idx]}"
                            )
                        prompts_text = "\n\n".join(weighted_prompts)
                        combine_prompt = f"""
Combine these system prompts into a single coherent system prompt, using the specified influence weights as a guide for how much each should contribute to the final prompt.
Prioritize elements from higher-influence prompts while ensuring the result is logical and flows well.

Prompts to combine:
{prompts_text}

Output ONLY the combined system prompt, without any additional text or explanations.
"""
                    else:
                        prompts_text = "\n\n".join([f"Prompt '{name}':\n{text}" for name, text in zip(selected_prompt_names, selected_prompt_texts)])
                        combine_prompt = f"""
Combine these system prompts into a single coherent system prompt using this strategy: {combination_strategy}.

Prompts to combine:
{prompts_text}

Output ONLY the combined system prompt, without any additional text or explanations.
"""

                    response = model.generate_content(combine_prompt, generation_config={"temperature": gemini_temperature})
                    combined_prompt = response.text.strip()
                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                    # Store combination results
                    st.session_state.combination_results = {
                        'combined_prompt': combined_prompt,
                        'individual_results': [],
                        'combined_result': None,
                        'strategy': combination_strategy,
                        'temperature': temperature,
                        'slider_weights': st.session_state.slider_weights if combination_strategy == "Slider - Custom influence weights" else None,
                        'suggested_results': st.session_state.combination_results.get('suggested_results', []) if st.session_state.get('combination_results') else []
                    }

                    # Display Combined Prompt
                    with combined_prompt_container:
                        st.subheader("Combined System Prompt")
                        st.text_area("Combined Prompt:", value=combined_prompt, height=200)
                except Exception as e:
                    st.error(f"Error combining prompts: {str(e)}")
                    return

            # Step 2: Test Individual Prompts
            with st.spinner("Testing individual prompts..."):
                individual_results = []
                for idx in selected_prompts:
                    system_prompt = st.session_state.prompts[idx]
                    prompt_name = st.session_state.prompt_names[idx]
                    try:
                        result = call_api_func(
                            system_prompt=system_prompt,
                            query=query_text,
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

                    export_row_dict = {
                        'user_name': user_name,
                        'test_type': 'Combination_Individual',
                        'prompt_name': prompt_name,
                        'system_prompt': system_prompt,
                        'query': query_text,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': None,
                        'remark': 'Individual prompt in combination test',
                        'edited': False,
                        'step': '',
                        'combination_strategy': combination_strategy,
                        'combination_temperature': temperature,
                        'slider_weights': st.session_state.slider_weights if combination_strategy == "Slider - Custom influence weights" else None
                    }

                    maybe_uid = save_export_entry(
                        prompt_name=prompt_name,
                        system_prompt=system_prompt,
                        query=query_text,
                        response=response_text,
                        mode="Combination_Individual",
                        remark="Individual prompt in combination test",
                        status=status,
                        status_code=status_code,
                        rating=None,
                        step='',
                        combination_strategy=combination_strategy,
                        combination_temperature=temperature,
                        slider_weights=st.session_state.slider_weights if combination_strategy == "Slider - Custom influence weights" else None,
                        user_name=user_name
                    )

                    generated_uid = f"Combination_Individual_{prompt_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                    unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                    add_result_row(
                        test_type='Combination_Individual',
                        prompt_name=prompt_name,
                        system_prompt=system_prompt,
                        query=query_text,
                        response=response_text,
                        status=status,
                        status_code=status_code,
                        remark='Individual prompt in combination test',
                        rating=None,
                        edited=False,
                        step='',
                        combination_strategy=combination_strategy,
                        combination_temperature=temperature,
                        user_name=user_name
                    )

                    last_index = st.session_state.test_results.index[-1]
                    if st.session_state.test_results.at[last_index, 'unique_id'] != unique_id:
                        st.session_state.test_results.at[last_index, 'unique_id'] = unique_id

                    st.session_state.response_ratings[unique_id] = None

                    individual_results.append({
                        'prompt_name': prompt_name,
                        'system_prompt': system_prompt,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': 0,
                        'remark': 'Individual prompt in combination test',
                        'edited': False,
                        'unique_id': unique_id
                    })

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)
                    time.sleep(1)

                st.session_state.combination_results['individual_results'] = individual_results

            # Step 3: Test Combined Prompt
            with st.spinner("Testing combined prompt..."):
                # Define a unique ID for the initial result dict before the save logic corrects it
                generated_uid_combined = f"Combination_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                try:
                    result = call_api_func(
                        system_prompt=combined_prompt,
                        query=query_text,
                        body_template=body_template,
                        headers=headers,
                        response_path=response_path
                    )
                    response_text = result.get('response', None)
                    status = result.get('status', 'Failed')
                    status_code = str(result.get('status_code', 'N/A'))

                    # Initialize combined result with all necessary fields
                    st.session_state.combination_results['combined_result'] = {
                        'unique_id': generated_uid_combined,
                        'system_prompt': combined_prompt,
                        'query': query_text,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'combination_strategy': combination_strategy,
                        'combination_temperature': temperature,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                except Exception as e:
                    st.error(f"Error in API call for combined prompt: {str(e)}")
                    response_text = f"Error: {str(e)}"
                    status = 'Failed'
                    status_code = 'N/A'

                    # Initialize combined result even in error case
                    st.session_state.combination_results['combined_result'] = {
                        'unique_id': generated_uid_combined,
                        'system_prompt': combined_prompt,
                        'query': query_text,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'combination_strategy': combination_strategy,
                        'combination_temperature': temperature,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                export_row_dict = {
                    'user_name': user_name,
                    'test_type': 'Combination',
                    'prompt_name': 'Combined Prompt',
                    'system_prompt': combined_prompt,
                    'query': query_text,
                    'response': response_text,
                    'status': status,
                    'status_code': status_code,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rating': 0,
                    'remark': 'Combined prompt test',
                    'edited': False,
                    'step': '',
                    'combination_strategy': combination_strategy,
                    'combination_temperature': temperature,
                    'slider_weights': st.session_state.slider_weights if combination_strategy == "Slider - Custom influence weights" else None
                }

                maybe_uid = save_export_entry(
                    prompt_name='Combined Prompt',
                    system_prompt=combined_prompt,
                    query=query_text,
                    response=response_text,
                    mode="Combination",
                    remark="Combined prompt test",
                    status=status,
                    status_code=status_code,
                    rating=None,
                    step='',
                    combination_strategy=combination_strategy,
                    combination_temperature=temperature,
                    slider_weights=st.session_state.slider_weights if combination_strategy == "Slider - Custom influence weights" else None,
                    user_name=user_name
                )

                generated_uid = f"Combination_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                add_result_row(
                    test_type='Combination',
                    prompt_name='Combined Prompt',
                    system_prompt=combined_prompt,
                    query=query_text,
                    response=response_text,
                    status=status,
                    status_code=status_code,
                    remark='Combined prompt test',
                    rating=None,
                    edited=False,
                    step='',
                    combination_strategy=combination_strategy,
                    combination_temperature=temperature,
                    user_name=user_name
                )

                last_index = st.session_state.test_results.index[-1]
                if st.session_state.test_results.at[last_index, 'unique_id'] != unique_id:
                    st.session_state.test_results.at[last_index, 'unique_id'] = unique_id

                st.session_state.response_ratings[unique_id] = None

                st.session_state.combination_results['combined_result'] = {
                    'prompt_name': 'Combined Prompt',
                    'system_prompt': combined_prompt,
                    'query': query_text,
                    'response': response_text,
                    'status': status,
                    'status_code': status_code,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'rating': None,
                    'remark': 'Combined prompt test',
                    'edited': False,
                    'unique_id': unique_id,
                    'combination_strategy': combination_strategy,
                    'combination_temperature': temperature
                }

                current_step += 1
                progress_bar.progress(current_step / total_steps)

            st.success("Combination and testing completed!")
            st.rerun()

    # Display Individual Results if exist
    if st.session_state.get('combination_results') and st.session_state.combination_results.get('individual_results'):
        with individual_results_container:
            st.subheader("Individual Prompt Results")
            # Initialize enhancement requests storage if not exists
            if 'enhancement_requests' not in st.session_state:
                st.session_state.enhancement_requests = {}

            for i, individual_result in enumerate(st.session_state.combination_results['individual_results']):
                unique_id = individual_result['unique_id']
                status_color = "🟢" if individual_result['status'] == 'Success' else "🔴"
                with st.expander(f"{status_color} {individual_result['prompt_name']} - {individual_result['status']}"):
                    st.write("**System Prompt:**")
                    st.text(individual_result['system_prompt'])
                    st.write("**Query:**")
                    st.text(query_text)
                    st.write("**Response:**")

                    edited_response = st.text_area(
                        "Response (editable):",
                        value=individual_result['response'] if pd.notnull(individual_result['response']) else "",
                        height=150,
                        key=f"edit_individual_response_{i}_{unique_id}"
                    )

                    # Dynamic rating update
                    live_rating = st.session_state.response_ratings.get(unique_id, individual_result.get('rating'))
                    if pd.isna(live_rating) or live_rating is None:
                        live_rating = 0
                    rating_value = st.slider(
                        "Rate this response (0-10):",
                        min_value=0,
                        max_value=10,
                        value=int(live_rating),
                        key=f"rating_individual_{i}_{unique_id}"
                    )

                    if rating_value != live_rating:
                        st.session_state.response_ratings[unique_id] = rating_value
                        individual_result['rating'] = rating_value
                        st.session_state.combination_results['individual_results'][i] = individual_result
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating_value
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'rating'] = rating_value
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                        st.rerun()

                    if edited_response != (individual_result['response'] or ""):
                        if st.button("💾 Save Edited Response", key=f"save_individual_edited_{i}_{unique_id}"):
                            individual_result['response'] = edited_response
                            individual_result['edited'] = True
                            individual_result['remark'] = 'Edited and saved'
                            st.session_state.combination_results['individual_results'][i] = individual_result

                            saved_unique_id = save_export_entry(
                                prompt_name=individual_result.get('prompt_name', 'Unknown Individual Prompt'),
                                system_prompt=individual_result['system_prompt'],
                                query=query_text,
                                response=edited_response,
                                mode="Combination_Individual",
                                remark="Edited and saved",
                                status=individual_result['status'],
                                status_code=individual_result['status_code'],
                                rating=st.session_state.response_ratings.get(unique_id, 0),
                                step=str(individual_result.get('step')) if individual_result.get('step') else '',
                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                combination_temperature=temperature,
                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                edited=True,
                                user_name=user_name
                            )

                            individual_result['unique_id'] = saved_unique_id
                            st.session_state.combination_results['individual_results'][i] = individual_result
                            if saved_unique_id != unique_id:
                                st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.pop(unique_id)
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = edited_response
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = 'Edited and saved'
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'response'] = edited_response
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'remark'] = 'Edited and saved'
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                            st.success("Edited response saved.")
                            st.rerun()

                    # MODIFIED: Add enhancement request input and submit button for suggestion
                    if st.button("🔮 Suggest Prompt for This Response", key=f"suggest_individual_btn_{i}_{unique_id}", disabled=not gemini_api_key):
                        st.session_state[f"suggest_individual_active_{i}_{unique_id}"] = True

                    if st.session_state.get(f"suggest_individual_active_{i}_{unique_id}", False):
                        st.write("**Suggest Prompt Improvements**")
                        enhancement_request = st.text_area(
                            "Describe desired improvements for the response (optional):",
                            value="",  # Let Streamlit manage the value
                            height=100,
                            key=f"enhancement_request_individual_{i}_{unique_id}",
                            placeholder="e.g., Make the response more detailed, add examples, change tone to be more professional..."
                        )
                        if st.button("Submit Suggestion Request", key=f"submit_suggest_individual_{i}_{unique_id}"):
                            with st.spinner("Generating prompt suggestion..."):
                                try:
                                    genai.configure(api_key=gemini_api_key)
                                    suggestion = suggest_func(
                                        individual_result['system_prompt'],  # Existing prompt
                                        edited_response,                    # Target response
                                        query_text,                         # Query
                                        rating=st.session_state.response_ratings.get(unique_id),  # Rating (optional)
                                        enhancement_request=enhancement_request if enhancement_request.strip() else None  # Enhancement (optional)
                                    )
                                    st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"] = suggestion
                                    st.session_state[f"suggested_prompt_name_individual_{i}_{unique_id}"] = f"Suggested Prompt for {individual_result['prompt_name']}"
                                    st.session_state[f"suggest_individual_active_{i}_{unique_id}"] = False  # Reset to hide input
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating suggestion: {str(e)}")

                    # If suggestion exists, show save / save & run UI
                    if f"suggested_prompt_individual_{i}_{unique_id}" in st.session_state:
                        st.write("**Suggested System Prompt:**")
                        st.text_area(
                            "Suggested Prompt:",
                            value=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                            height=120,
                            key=f"suggested_display_individual_{i}_{unique_id}",
                            disabled=True
                        )

                        # Shared prompt name input (moved out of columns to match individual.py)
                        prompt_name_input = st.text_input(
                            "Name this suggested prompt:",
                            value=st.session_state[f"suggested_prompt_name_individual_{i}_{unique_id}"],
                            key=f"suggest_name_individual_{i}_{unique_id}"
                        )

                        col_save, col_save_run, col_edit = st.columns(3)

                        with col_save:
                            if st.button("💾 Save as Prompt", key=f"save_suggest_individual_{i}_{unique_id}"):
                                saved_name = prompt_name_input.strip() or f"Suggested Prompt for {individual_result['prompt_name']}"
                                if saved_name:
                                    export_row_dict = {
                                        'user_name': user_name,
                                        'test_type': 'Combination_Suggested',
                                        'prompt_name': saved_name,
                                        'system_prompt': st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                        'query': query_text,
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Saved only',
                                        'edited': False,
                                        'step': '',
                                        'combination_strategy': st.session_state.combination_results.get('strategy'),
                                        'combination_temperature': temperature,
                                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=saved_name,
                                        system_prompt=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        mode='Combination_Suggested',
                                        remark="Saved only",
                                        status='Not Executed',
                                        status_code='N/A',
                                        rating=None,
                                        step='',
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=temperature,
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        user_name=user_name
                                    )

                                    generated_uid = f"Combination_Suggested_{saved_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                    add_result_row(
                                        test_type='Combination_Suggested',
                                        prompt_name=saved_name,
                                        system_prompt=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        status='Not Executed',
                                        status_code='N/A',
                                        remark='Saved only',
                                        rating=None,
                                        edited=False,
                                        step='',
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=temperature,
                                        user_name=user_name
                                    )

                                    last_index = st.session_state.test_results.index[-1]
                                    if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                        st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                    st.session_state.response_ratings[saved_unique_id] = 0
                                    st.session_state.prompts.append(st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"])
                                    st.session_state.prompt_names.append(saved_name)
                                    del st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"]
                                    del st.session_state[f"suggested_prompt_name_individual_{i}_{unique_id}"]
                                    st.success(f"Saved as new prompt: {saved_name}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                        with col_save_run:
                            if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_individual_{i}_{unique_id}"):
                                saved_name = prompt_name_input.strip() or f"Suggested Prompt for {individual_result['prompt_name']}"
                                if saved_name:
                                    st.session_state.prompts.append(st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"])
                                    st.session_state.prompt_names.append(saved_name)

                                    with st.spinner("Running new prompt..."):
                                        try:
                                            run_result = call_api_func(
                                                system_prompt=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                                query=query_text,
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
                                            'user_name': user_name,
                                            'test_type': 'Combination_Suggested',
                                            'prompt_name': saved_name,
                                            'system_prompt': st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                            'query': query_text,
                                            'response': response_text,
                                            'status': status,
                                            'status_code': status_code,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': 0,
                                            'remark': 'Saved and ran suggested prompt',
                                            'edited': False,
                                            'step': '',
                                            'combination_strategy': st.session_state.combination_results.get('strategy'),
                                            'combination_temperature': temperature,
                                            'slider_weights': st.session_state.combination_results.get('slider_weights')
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=saved_name,
                                            system_prompt=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                            query=query_text,
                                            response=response_text,
                                            mode="Combination_Suggested",
                                            remark="Saved and ran suggested prompt",
                                            status=status,
                                            status_code=status_code,
                                            rating=None,
                                            step='',
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                                            user_name=user_name
                                        )

                                        generated_uid = f"Combination_Suggested_{saved_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                        saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                        add_result_row(
                                            test_type='Combination_Suggested',
                                            prompt_name=saved_name,
                                            system_prompt=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                            query=query_text,
                                            response=response_text,
                                            status=status,
                                            status_code=status_code,
                                            remark='Saved and ran suggested prompt',
                                            rating=None,
                                            edited=False,
                                            step='',
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            user_name=user_name
                                        )

                                        last_index = st.session_state.test_results.index[-1]
                                        if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                            st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                        st.session_state.response_ratings[saved_unique_id] = 0
                                        # Store suggested result in session state
                                        if 'suggested_results' not in st.session_state.combination_results:
                                            st.session_state.combination_results['suggested_results'] = []
                                        st.session_state.combination_results['suggested_results'].append({
                                            'prompt_name': saved_name,
                                            'system_prompt': st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                            'response': response_text,
                                            'status': status,
                                            'status_code': status_code,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': 0,
                                            'remark': 'Saved and ran suggested prompt',
                                            'edited': False,
                                            'unique_id': saved_unique_id
                                        })

                                        del st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"]
                                        del st.session_state[f"suggested_prompt_name_individual_{i}_{unique_id}"]
                                        st.success(f"Saved and ran new prompt: {saved_name}")

                                        # Add immediate response display and rating (to match individual.py)
                                        st.write("**Response from Suggested Prompt:**")
                                        st.text_area("Response:", value=response_text or "", height=150, key=f"suggested_run_resp_individual_{i}_{unique_id}")
                                        rating_val = st.slider(
                                            "Rate this response (0-10):",
                                            min_value=0,
                                            max_value=10,
                                            value=0,
                                            key=f"rating_suggested_individual_{i}_{unique_id}"
                                        )
                                        if rating_val != 0 or (rating_val == 0 and st.button("Set rating to 0", key=f"set_zero_individual_{i}_{unique_id}")):
                                            st.session_state.response_ratings[saved_unique_id] = rating_val
                                            # Update the new row in test_results (find by unique_id)
                                            mask = st.session_state.test_results['unique_id'] == saved_unique_id
                                            if mask.any():
                                                st.session_state.test_results.loc[mask, 'rating'] = rating_val
                                                st.session_state.test_results.loc[mask, 'edited'] = True
                                            export_mask = st.session_state.export_data['unique_id'] == saved_unique_id
                                            if export_mask.any():
                                                st.session_state.export_data.loc[export_mask, 'rating'] = rating_val
                                                st.session_state.export_data.loc[export_mask, 'edited'] = True
                                            st.rerun()

                                        st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                        with col_edit:
                            if st.button("✏️ Edit", key=f"edit_suggest_individual_{i}_{unique_id}"):
                                st.session_state[f"edit_suggest_individual_{i}_{unique_id}_active"] = True

                            if st.session_state.get(f"edit_suggest_individual_{i}_{unique_id}_active", False):
                                edited_suggestion = st.text_area(
                                    "Edit Suggested Prompt:",
                                    value=st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"],
                                    height=100,
                                    key=f"edit_suggested_individual_{i}_{unique_id}"
                                )
                                edit_prompt_name = st.text_input(
                                    "Prompt Name for Edited Prompt:",
                                    value=st.session_state[f"suggested_prompt_name_individual_{i}_{unique_id}"],
                                    key=f"edit_suggest_name_individual_{i}_{unique_id}"
                                )
                                if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_individual_{i}_{unique_id}"):
                                    if edit_prompt_name.strip():
                                        export_row_dict = {
                                            'user_name': user_name,
                                            'test_type': 'Combination_Suggested',
                                            'prompt_name': edit_prompt_name.strip(),
                                            'system_prompt': edited_suggestion,
                                            'query': query_text,
                                            'response': 'Prompt saved but not executed',
                                            'status': 'Not Executed',
                                            'status_code': 'N/A',
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': 0,
                                            'remark': 'Save only',
                                            'edited': False,
                                            'step': '',
                                            'combination_strategy': st.session_state.combination_results.get('strategy'),
                                            'combination_temperature': temperature,
                                            'slider_weights': st.session_state.combination_results.get('slider_weights')
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=query_text,
                                            response='Prompt saved but not executed',
                                            mode='Combination_Suggested',
                                            remark="Save only",
                                            status='Not Executed',
                                            status_code='N/A',
                                            rating=None,
                                            edited=False,
                                            step='',
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            user_name=user_name
                                        )

                                        last_index = st.session_state.test_results.index[-1]
                                        if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                            st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                        st.session_state.response_ratings[saved_unique_id] = None
                                        st.session_state.prompts.append(edited_suggestion)
                                        st.session_state.prompt_names.append(edit_prompt_name.strip())
                                        st.session_state[f"edit_suggest_individual_{i}_{unique_id}_active"] = False
                                        del st.session_state[f"suggested_prompt_individual_{i}_{unique_id}"]
                                        del st.session_state[f"suggested_prompt_name_individual_{i}_{unique_id}"]
                                        st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                        st.rerun()
                                    else:
                                        st.error("Please provide a prompt name")

                    st.write("**Details:**")
                    rating = st.session_state.response_ratings.get(unique_id, individual_result.get('rating'))
                    rating_display = f"{rating}/10" if rating is not None else "Not rated yet"
                    st.write(
                        f"Status Code: {individual_result.get('status_code', 'N/A')} | "
                        f"Time: {individual_result.get('timestamp', 'N/A')} | "
                        f"Rating: {rating_display}"
                    )

    # Display Combined Result if exists
    if st.session_state.get('combination_results') and st.session_state.combination_results.get('combined_result'):
        with combined_result_container:
            st.subheader("Combined Prompt Result")
            combined_result = st.session_state.combination_results['combined_result']
            unique_id = combined_result['unique_id']
            status_color = "🟢" if combined_result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} Combined Prompt - {combined_result['status']}"):
                st.write("**System Prompt:**")
                st.text(combined_result['system_prompt'])
                st.write("**Query:**")
                st.text(query_text)
                st.write("**Response:**")
                edited_response = st.text_area(
                    "Response (editable):",
                    value=combined_result['response'] if pd.notnull(combined_result['response']) else "",
                    height=150,
                    key=f"edit_combined_response_{unique_id}"
                )

                # Dynamic rating update
                live_rating = st.session_state.response_ratings.get(unique_id, combined_result.get('rating'))
                if pd.isna(live_rating) or live_rating is None:
                    live_rating = 0
                rating_value = st.slider(
                    "Rate this response (0-10):",
                    min_value=0,
                    max_value=10,
                    value=int(live_rating),
                    key=f"rating_combined_{unique_id}"
                )

                if rating_value != live_rating:
                    st.session_state.response_ratings[unique_id] = rating_value
                    combined_result['rating'] = rating_value
                    st.session_state.combination_results['combined_result'] = combined_result
                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating_value
                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                    st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'rating'] = rating_value
                    st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                    st.rerun()

                if edited_response != (combined_result['response'] or ""):
                    if st.button("💾 Save Edited Response", key=f"save_combined_edited_{unique_id}"):
                        combined_result['response'] = edited_response
                        combined_result['edited'] = True
                        combined_result['remark'] = 'Edited and saved'
                        st.session_state.combination_results['combined_result'] = combined_result

                        saved_unique_id = save_export_entry(
                            prompt_name='Combined Prompt',
                            system_prompt=combined_result['system_prompt'],
                            query=query_text,
                            response=edited_response,
                            mode="Combination",
                            remark="Edited and saved",
                            status=combined_result['status'],
                            status_code=combined_result['status_code'],
                            combination_strategy=combined_result.get('combination_strategy'),
                            combination_temperature=combined_result.get('combination_temperature'),
                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                            rating=st.session_state.response_ratings.get(unique_id, 0),
                            step=str(combined_result.get('step')) if combined_result.get('step') else '',
                            edited=True,
                            user_name=user_name
                        )

                        combined_result['unique_id'] = saved_unique_id
                        st.session_state.combination_results['combined_result'] = combined_result
                        if saved_unique_id != unique_id:
                            st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.pop(unique_id)
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = edited_response
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = 'Edited and saved'
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'response'] = edited_response
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'remark'] = 'Edited and saved'
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                        st.success("Edited response saved.")
                        st.rerun()

                # MODIFIED: Add enhancement request input and submit button for suggestion
                if st.button("🔮 Suggest Prompt for Combined Response", key=f"suggest_combined_btn_{unique_id}", disabled=not gemini_api_key):
                    st.session_state[f"suggest_combined_active_{unique_id}"] = True

                if st.session_state.get(f"suggest_combined_active_{unique_id}", False):
                    st.write("**Suggest Prompt Improvements**")
                    enhancement_request = st.text_area(
                        "Describe desired improvements for the response (optional):",
                        value="",  # Let Streamlit manage the value
                        height=100,
                        key=f"enhancement_request_combined_{unique_id}",
                        placeholder="e.g., Make the response more detailed, add examples, change tone to be more professional..."
                    )
                    if st.button("Submit Suggestion Request", key=f"submit_suggest_combined_{unique_id}"):
                        with st.spinner("Generating prompt suggestion..."):
                            try:
                                genai.configure(api_key=gemini_api_key)
                                suggestion = suggest_func(
                                    combined_result['system_prompt'],  # Existing prompt
                                    edited_response,                  # Target response
                                    query_text,                       # Query
                                    rating=st.session_state.response_ratings.get(unique_id),  # Rating (optional)
                                    enhancement_request=enhancement_request if enhancement_request.strip() else None  # Enhancement (optional)
                                )
                                st.session_state[f"suggested_prompt_combined_{unique_id}"] = suggestion
                                st.session_state[f"suggested_prompt_name_combined_{unique_id}"] = "Suggested Prompt for Combined Response"
                                st.session_state[f"suggest_combined_active_{unique_id}"] = False  # Reset to hide input
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating suggestion: {str(e)}")

                # If suggestion exists, show save / save & run UI
                if f"suggested_prompt_combined_{unique_id}" in st.session_state:
                    st.write("**Suggested System Prompt:**")
                    st.text_area(
                        "Suggested Prompt:",
                        value=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                        height=120,
                        key=f"suggested_display_combined_{unique_id}",
                        disabled=True
                    )

                    # Shared prompt name input (moved out of columns to match individual.py)
                    prompt_name_input = st.text_input(
                        "Name this suggested prompt:",
                        value=st.session_state[f"suggested_prompt_name_combined_{unique_id}"],
                        key=f"suggest_name_combined_{unique_id}"
                    )

                    col_save, col_save_run, col_edit = st.columns(3)

                    with col_save:
                        if st.button("💾 Save as Prompt", key=f"save_suggest_combined_{unique_id}"):
                            saved_name = prompt_name_input.strip() or "Suggested Prompt for Combined Response"
                            if saved_name:
                                export_row_dict = {
                                    'user_name': user_name,
                                    'test_type': 'Combination_Suggested',
                                    'prompt_name': saved_name,
                                    'system_prompt': st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                    'query': query_text,
                                    'response': 'Prompt saved but not executed',
                                    'status': 'Not Executed',
                                    'status_code': 'N/A',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': 0,
                                    'remark': 'Saved only',
                                    'edited': False,
                                    'step': '',
                                    'combination_strategy': combined_result.get('combination_strategy'),
                                    'combination_temperature': combined_result.get('combination_temperature'),
                                    'slider_weights': st.session_state.combination_results.get('slider_weights')
                                }

                                maybe_uid = save_export_entry(
                                    prompt_name=saved_name,
                                    system_prompt=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                    query=query_text,
                                    response='Prompt saved but not executed',
                                    mode='Combination_Suggested',
                                    remark="Saved only",
                                    status='Not Executed',
                                    status_code='N/A',
                                    rating=None,
                                    step='',
                                    combination_strategy=combined_result.get('combination_strategy'),
                                    combination_temperature=combined_result.get('combination_temperature'),
                                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                                    user_name=user_name
                                )

                                generated_uid = f"Combination_Suggested_{saved_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                add_result_row(
                                    test_type='Combination_Suggested',
                                    prompt_name=saved_name,
                                    system_prompt=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                    query=query_text,
                                    response='Prompt saved but not executed',
                                    status='Not Executed',
                                    status_code='N/A',
                                    remark='Saved only',
                                    rating=None,
                                    edited=False,
                                    step='',
                                    combination_strategy=combined_result.get('combination_strategy'),
                                    combination_temperature=combined_result.get('combination_temperature'),
                                    user_name=user_name
                                )

                                last_index = st.session_state.test_results.index[-1]
                                if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                    st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                st.session_state.response_ratings[saved_unique_id] = 0
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_combined_{unique_id}"])
                                st.session_state.prompt_names.append(saved_name)
                                del st.session_state[f"suggested_prompt_combined_{unique_id}"]
                                del st.session_state[f"suggested_prompt_name_combined_{unique_id}"]
                                st.success(f"Saved as new prompt: {saved_name}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")

                    with col_save_run:
                        if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_combined_{unique_id}"):
                            saved_name = prompt_name_input.strip() or "Suggested Prompt for Combined Response"
                            if saved_name:
                                st.session_state.prompts.append(st.session_state[f"suggested_prompt_combined_{unique_id}"])
                                st.session_state.prompt_names.append(saved_name)

                                with st.spinner("Running new prompt..."):
                                    try:
                                        run_result = call_api_func(
                                            system_prompt=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                            query=query_text,
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
                                        'user_name': user_name,
                                        'test_type': 'Combination_Suggested',
                                        'prompt_name': saved_name,
                                        'system_prompt': st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                        'query': query_text,
                                        'response': response_text,
                                        'status': status,
                                        'status_code': status_code,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Saved and ran suggested prompt',
                                        'edited': False,
                                        'step': '',
                                        'combination_strategy': combined_result.get('combination_strategy'),
                                        'combination_temperature': combined_result.get('combination_temperature'),
                                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=saved_name,
                                        system_prompt=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                        query=query_text,
                                        response=response_text,
                                        mode="Combination_Suggested",
                                        remark="Saved and ran suggested prompt",
                                        status=status,
                                        status_code=status_code,
                                        rating=None,
                                        step='',
                                        combination_strategy=combined_result.get('combination_strategy'),
                                        combination_temperature=combined_result.get('combination_temperature'),
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        user_name=user_name
                                    )

                                    generated_uid = f"Combination_Suggested_{saved_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                    add_result_row(
                                        test_type='Combination_Suggested',
                                        prompt_name=saved_name,
                                        system_prompt=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                        query=query_text,
                                        response=response_text,
                                        status=status,
                                        status_code=status_code,
                                        remark='Saved and ran suggested prompt',
                                        rating=None,
                                        edited=False,
                                        step='',
                                        combination_strategy=combined_result.get('combination_strategy'),
                                        combination_temperature=combined_result.get('combination_temperature'),
                                        user_name=user_name
                                    )

                                    last_index = st.session_state.test_results.index[-1]
                                    if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                        st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                    st.session_state.response_ratings[saved_unique_id] = 0
                                    # Store suggested result in session state
                                    if 'suggested_results' not in st.session_state.combination_results:
                                        st.session_state.combination_results['suggested_results'] = []
                                    st.session_state.combination_results['suggested_results'].append({
                                        'prompt_name': saved_name,
                                        'system_prompt': st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                        'response': response_text,
                                        'status': status,
                                        'status_code': status_code,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Saved and ran suggested prompt',
                                        'edited': False,
                                        'unique_id': saved_unique_id
                                    })

                                    del st.session_state[f"suggested_prompt_combined_{unique_id}"]
                                    del st.session_state[f"suggested_prompt_name_combined_{unique_id}"]
                                    st.success(f"Saved and ran new prompt: {saved_name}")

                                    # Add immediate response display and rating (to match individual.py)
                                    st.write("**Response from Suggested Prompt:**")
                                    st.text_area("Response:", value=response_text or "", height=150, key=f"suggested_run_resp_combined_{unique_id}")
                                    rating_val = st.slider(
                                        "Rate this response (0-10):",
                                        min_value=0,
                                        max_value=10,
                                        value=0,
                                        key=f"rating_suggested_combined_{unique_id}"
                                    )
                                    if rating_val != 0 or (rating_val == 0 and st.button("Set rating to 0", key=f"set_zero_combined_{unique_id}")):
                                        st.session_state.response_ratings[saved_unique_id] = rating_val
                                        # Update the new row in test_results (find by unique_id)
                                        mask = st.session_state.test_results['unique_id'] == saved_unique_id
                                        if mask.any():
                                            st.session_state.test_results.loc[mask, 'rating'] = rating_val
                                            st.session_state.test_results.loc[mask, 'edited'] = True
                                        export_mask = st.session_state.export_data['unique_id'] == saved_unique_id
                                        if export_mask.any():
                                            st.session_state.export_data.loc[export_mask, 'rating'] = rating_val
                                            st.session_state.export_data.loc[export_mask, 'edited'] = True
                                        st.rerun()

                                    st.rerun()
                            else:
                                st.error("Please provide a prompt name")

                    with col_edit:
                        if st.button("✏️ Edit", key=f"edit_suggest_combined_{unique_id}"):
                            st.session_state[f"edit_suggest_combined_{unique_id}_active"] = True

                        if st.session_state.get(f"edit_suggest_combined_{unique_id}_active", False):
                            edited_suggestion = st.text_area(
                                "Edit Suggested Prompt:",
                                value=st.session_state[f"suggested_prompt_combined_{unique_id}"],
                                height=100,
                                key=f"edit_suggested_combined_{unique_id}"
                            )
                            edit_prompt_name = st.text_input(
                                "Prompt Name for Edited Prompt:",
                                value=st.session_state[f"suggested_prompt_name_combined_{unique_id}"],
                                key=f"edit_suggest_name_combined_{unique_id}"
                            )
                            if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_combined_{unique_id}"):
                                if edit_prompt_name.strip():
                                    export_row_dict = {
                                        'user_name': user_name,
                                        'test_type': 'Combination_Suggested',
                                        'prompt_name': edit_prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': query_text,
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': 0,
                                        'remark': 'Save only',
                                        'edited': False,
                                        'step': '',
                                        'combination_strategy': combined_result.get('combination_strategy'),
                                        'combination_temperature': combined_result.get('combination_temperature'),
                                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        mode='Combination_Suggested',
                                        remark="Save only",
                                        status='Not Executed',
                                        status_code='N/A',
                                        rating=None,
                                        step='',
                                        combination_strategy=combined_result.get('combination_strategy'),
                                        combination_temperature=combined_result.get('combination_temperature'),
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        user_name=user_name
                                    )

                                    generated_uid = f"Combination_Suggested_{edit_prompt_name.strip()}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                    add_result_row(
                                        test_type='Combination_Suggested',
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        status='Not Executed',
                                        status_code='N/A',
                                        remark='Save only',
                                        rating=None,
                                        edited=False,
                                        step='',
                                        combination_strategy=combined_result.get('combination_strategy'),
                                        combination_temperature=combined_result.get('combination_temperature'),
                                        user_name=user_name
                                    )

                                    last_index = st.session_state.test_results.index[-1]
                                    if st.session_state.test_results.at[last_index, 'unique_id'] != saved_unique_id:
                                        st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id

                                    st.session_state.response_ratings[saved_unique_id] = 0
                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(edit_prompt_name.strip())
                                    st.session_state[f"edit_suggest_combined_{unique_id}_active"] = False
                                    del st.session_state[f"suggested_prompt_combined_{unique_id}"]
                                    del st.session_state[f"suggested_prompt_name_combined_{unique_id}"]
                                    st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                st.write("**Details:**")
                rating = st.session_state.response_ratings.get(unique_id, combined_result.get('rating'))
                rating_display = f"{rating}/10" if rating is not None else "Not rated yet"
                st.write(
                    f"Status Code: {combined_result.get('status_code', 'N/A')} | "
                    f"Time: {combined_result.get('timestamp', 'N/A')} | "
                    f"Rating: {rating_display}"
                )

    # Display Suggested Results if exist
    if st.session_state.get('combination_results') and st.session_state.combination_results.get('suggested_results'):
        with suggested_results_container:
            st.subheader("Suggested Prompt Results")

            for i, suggested_result in enumerate(st.session_state.combination_results['suggested_results']):
                unique_id = suggested_result['unique_id']
                status_color = "🟢" if suggested_result['status'] == 'Success' else "🔴"

                # Check if this result is already present in test_results
                if unique_id in st.session_state.test_results['unique_id'].values:
                    latest_row = st.session_state.test_results[st.session_state.test_results['unique_id'] == unique_id].iloc[-1]
                    suggested_result.update({
                        'response': latest_row['response'],
                        'status': latest_row['status'],
                        'status_code': latest_row['status_code'],
                        'edited': latest_row['edited'],
                        'rating': latest_row['rating']
                    })

                with st.expander(f"{status_color} {suggested_result['prompt_name']} - {suggested_result['status']}"):
                    st.write("**System Prompt:**")
                    st.text(suggested_result['system_prompt'])
                    st.write("**Query:**")
                    st.text(query_text)
                    st.write("**Response:**")

                    edited_response = st.text_area(
                        "Response (editable):",
                        value=suggested_result['response'] if pd.notnull(suggested_result['response']) else "",
                        height=150,
                        key=f"edit_suggested_response_{i}_{unique_id}"
                    )

                    # Dynamic rating update
                    live_rating = st.session_state.response_ratings.get(unique_id, suggested_result.get('rating'))
                    if pd.isna(live_rating) or live_rating is None:
                        live_rating = 0
                    rating_value = st.slider(
                        "Rate this response (0-10):",
                        min_value=0,
                        max_value=10,
                        value=int(live_rating),
                        key=f"rating_suggested_{i}_{unique_id}"
                    )

                    if rating_value != live_rating:
                        st.session_state.response_ratings[unique_id] = rating_value
                        suggested_result['rating'] = rating_value
                        st.session_state.combination_results['suggested_results'][i] = suggested_result
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating_value
                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'rating'] = rating_value
                        st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                        st.rerun()

                    if edited_response != (suggested_result['response'] or ""):
                        if st.button("💾 Save Edited Response", key=f"save_suggested_edited_{i}_{unique_id}"):
                            suggested_result['response'] = edited_response
                            suggested_result['edited'] = True
                            suggested_result['remark'] = 'Edited and saved'
                            st.session_state.combination_results['suggested_results'][i] = suggested_result

                            saved_unique_id = save_export_entry(
                                prompt_name=suggested_result.get('prompt_name', 'Unknown Suggested Prompt'),
                                system_prompt=suggested_result['system_prompt'],
                                query=query_text,
                                response=edited_response,
                                mode="Combination_Suggested",
                                remark="Edited and saved",
                                status=suggested_result['status'],
                                status_code=suggested_result['status_code'],
                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                combination_temperature=temperature,
                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                rating=st.session_state.response_ratings.get(unique_id, 0),
                                step=str(suggested_result.get('step')) if suggested_result.get('step') else '',
                                edited=True,
                                user_name=user_name
                            )

                            suggested_result['unique_id'] = saved_unique_id
                            st.session_state.combination_results['suggested_results'][i] = suggested_result
                            if saved_unique_id != unique_id:
                                st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.pop(unique_id)
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = edited_response
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = 'Edited and saved'
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'response'] = edited_response
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'remark'] = 'Edited and saved'
                            st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                            st.success("Edited response saved.")
                            st.rerun()

                    st.write("**Details:**")
                    rating = st.session_state.response_ratings.get(unique_id, suggested_result.get('rating', 0))
                    rating_display = f"{rating}/10" if rating is not None else "Not rated yet"
                    st.write(
                        f"Status Code: {suggested_result.get('status_code', 'N/A')} | "
                        f"Time: {suggested_result.get('timestamp', 'N/A')} | "
                        f"Rating: {rating_display}"
                    )

    # Debug Log Download (Hidden unless needed)
    if st.session_state.get('debug_log'):
        st.download_button(
            label="Download Debug Log",
            data="\n".join(st.session_state.debug_log),
            file_name="debug_log.txt",
            mime="text/plain"
        )