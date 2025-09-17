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

def debug_log(message):
    """Log debug messages to session state instead of displaying on screen."""
    if 'debug_log' not in st.session_state:
        st.session_state.debug_log = []
    st.session_state.debug_log.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}")

def render_prompt_combination(api_url, query_text, body_template, headers, response_path, call_api_func, suggest_func, gemini_api_key=''):
    st.header("🤝 Prompt Combination")

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
                            st.session_state.slider_weights[selected_prompts[-1]] = 100 - sum(st.session_state.slider_weights.get(i, 0) for i in selected_prompts[:-1])

                for idx in selected_prompts:
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
                        normalized_weights = {k: (v/total_weight)*100 for k, v in st.session_state.slider_weights.items() if k in selected_prompts}
                        weight_info = "\n".join([f"{st.session_state.prompt_names[idx]} ({normalized_weights.get(idx, 0):.1f}% influence): {st.session_state.prompts[idx]}" for idx in selected_prompts])
                        combination_prompt = f"""
Please combine the following system prompts into one optimized prompt, using the specified influence weights to determine how much each prompt should contribute to the final result.

Weighted Prompts:
{weight_info}

Requirements:
1. Use the influence percentages to determine how much each prompt contributes
2. Higher weight prompts should have more prominent influence on the final structure and content
3. Lower weight prompts should contribute supporting elements or nuances
4. Create a coherent, unified prompt that balances all influences appropriately
5. Preserve the most important aspects from higher-weighted prompts
6. Eliminate redundancy and conflicts intelligently

Return only the combined system prompt without additional explanation.
"""
                    else:
                        prompt_info = "\n".join([f'{name}: {prompt}' for name, prompt in zip(selected_prompt_names, selected_prompt_texts)])
                        combination_prompt = f"""
Please combine the following system prompts into one optimized, coherent system prompt.

Strategy: {combination_strategy}

Individual Prompts:
{prompt_info}

Requirements:
1. Preserve the core intent and functionality of each individual prompt
2. Eliminate redundancy and conflicts
3. Create a well-structured, clear, and actionable combined prompt
4. Maintain the essential instructions and constraints from all prompts
5. Ensure the combined prompt is more effective than using the prompts separately

Return only the combined system prompt without additional explanation.
"""

                    generation_config = genai.types.GenerationConfig(temperature=gemini_temperature)
                    response = model.generate_content(combination_prompt, generation_config=generation_config)
                    combined_prompt = response.text

                    # Initialize combination_results
                    st.session_state.combination_results = {
                        'individual_prompts': selected_prompt_texts,
                        'individual_names': selected_prompt_names,
                        'selected_indices': selected_prompts,
                        'combined_prompt': combined_prompt,
                        'strategy': combination_strategy,
                        'temperature': temperature,
                        'slider_weights': st.session_state.slider_weights if combination_strategy == "Slider - Custom influence weights" else None,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'individual_results': [],
                        'combined_result': None
                    }
                    st.session_state.suggested_prompt = None
                    st.session_state.suggested_prompt_name = None

                    # Display combined prompt immediately
                    with combined_prompt_container:
                        st.subheader("Combined Prompt (Editable)")
                        edited_combined_prompt = st.text_area(
                            "Edit Combined Prompt:",
                            value=combined_prompt,
                            key=f"edit_combined_prompt_{st.session_state.combination_results['timestamp']}",
                            height=150
                        )
                        col_save, col_save_run = st.columns(2)
                        with col_save:
                            if st.button("💾 Save Combined Prompt"):
                                if edited_combined_prompt.strip():
                                    st.session_state.combination_results['combined_prompt'] = edited_combined_prompt
                                    st.session_state.prompts.append(edited_combined_prompt)
                                    st.session_state.prompt_names.append(f"Combined_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                                    st.success("Combined prompt saved!")
                                    st.rerun()
                                else:
                                    st.error("Please provide a non-empty prompt")
                        with col_save_run:
                            if st.button("🏃 Save and Rerun Combined Prompt"):
                                if edited_combined_prompt.strip():
                                    st.session_state.combination_results['combined_prompt'] = edited_combined_prompt
                                    st.session_state.prompts.append(edited_combined_prompt)
                                    st.session_state.prompt_names.append(f"Combined_{datetime.now().strftime('%Y%m%d%H%M%S')}")
                                    # Trigger rerun of combined prompt test
                                    st.session_state.rerun_combined_prompt = True
                                    st.session_state.new_combined_prompt = edited_combined_prompt
                                    st.rerun()
                                else:
                                    st.error("Please provide a non-empty prompt")

                        st.success("✅ Prompts combined successfully!")

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                except Exception as e:
                    st.error(f"Error combining prompts: {str(e)}")
                    progress_bar.empty()
                    return

            # Step 2: Test Individual Prompts
            with individual_results_container:
                st.subheader("Individual Results")
                individual_results = []
                for i, (prompt, name) in enumerate(zip(st.session_state.combination_results['individual_prompts'], st.session_state.combination_results['individual_names'])):
                    with st.spinner(f"Testing prompt '{name}'..."):
                        debug_log(f"Testing Prompt '{name}' with query '{query_text}'")
                        try:
                            result = call_api_func(
                                system_prompt=prompt,
                                query=query_text,
                                body_template=body_template,
                                headers=headers,
                                response_path=response_path
                            )
                            response_text = result.get('response', None)
                            status = result.get('status', 'Failed')
                            status_code = str(result.get('status_code', 'N/A'))
                            # Conditional delay based on status code
                            delay = 0.3 if status_code != '429' else 1.0
                        except Exception as e:
                            st.error(f"Error in API call for {name}: {str(e)}")
                            response_text = f"Error: {str(e)}"
                            status = 'Failed'
                            status_code = 'N/A'
                            delay = 0.3

                        debug_log(f"Result for '{name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                        export_row_dict = {
                            'test_type': 'Combination_Individual',
                            'prompt_name': name,
                            'system_prompt': prompt,
                            'query': query_text,
                            'response': response_text,
                            'status': status,
                            'status_code': status_code,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'rating': 0,
                            'remark': 'Saved and ran',
                            'edited': False,
                            'step': str(i + 1),
                            'input_query': query_text,
                            'combination_strategy': st.session_state.combination_results.get('strategy'),
                            'combination_temperature': temperature,
                            'slider_weights': st.session_state.combination_results.get('slider_weights')
                        }

                        maybe_uid = save_export_entry(
                            prompt_name=name,
                            system_prompt=prompt,
                            query=query_text,
                            response=response_text,
                            mode="Combination_Individual",
                            remark="Saved and ran",
                            status=status,
                            status_code=status_code,
                            combination_strategy=st.session_state.combination_results.get('strategy'),
                            combination_temperature=temperature,
                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                            rating=0,
                            step=str(i + 1),
                            input_query=query_text
                        )

                        generated_uid = f"Combination_{name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                        unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                        individual_result = {
                            'prompt_name': name,
                            'system_prompt': prompt,
                            'response': response_text,
                            'status': status,
                            'status_code': status_code,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'rating': 0,
                            'remark': 'Saved and ran',
                            'edited': False,
                            'unique_id': unique_id
                        }
                        individual_results.append(individual_result)
                        add_result_row(
                            test_type='Combination_Individual',
                            prompt_name=name,
                            system_prompt=prompt,
                            query=query_text,
                            response=response_text,
                            status=status,
                            status_code=status_code,
                            remark='Saved and ran',
                            rating=0,
                            edited=False,
                            step=str(i + 1),
                            input_query=query_text,
                            combination_strategy=st.session_state.combination_results.get('strategy'),
                            combination_temperature=temperature
                        )
                        last_index = st.session_state.test_results.index[-1]
                        st.session_state.test_results.at[last_index, 'unique_id'] = unique_id
                        st.session_state.response_ratings[unique_id] = 0
                        time.sleep(delay)

                    current_step += 1
                    progress_bar.progress(current_step / total_steps)

                st.session_state.combination_results['individual_results'] = individual_results

            # Step 3: Test Combined Prompt
            with combined_result_container:
                st.subheader("Combined Result")
                with st.spinner("Testing combined prompt..."):
                    try:
                        result = call_api_func(
                            system_prompt=st.session_state.combination_results['combined_prompt'],
                            query=query_text,
                            body_template=body_template,
                            headers=headers,
                            response_path=response_path
                        )
                        response_text = result.get('response', None)
                        status = result.get('status', 'Failed')
                        status_code = str(result.get('status_code', 'N/A'))
                    except Exception as e:
                        st.error(f"Error in API call for combined prompt: {str(e)}")
                        response_text = f"Error: {str(e)}"
                        status = 'Failed'
                        status_code = 'N/A'

                    debug_log(f"Result for combined prompt: status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                    export_row_dict = {
                        'test_type': 'Combination_Combined',
                        'prompt_name': 'AI_Combined',
                        'system_prompt': st.session_state.combination_results['combined_prompt'],
                        'query': query_text,
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': 0,
                        'remark': 'Saved and ran',
                        'edited': False,
                        'step': '',
                        'input_query': query_text,
                        'combination_strategy': st.session_state.combination_results.get('strategy'),
                        'combination_temperature': temperature,
                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                    }

                    maybe_uid = save_export_entry(
                        prompt_name='AI_Combined',
                        system_prompt=st.session_state.combination_results['combined_prompt'],
                        query=query_text,
                        response=response_text,
                        mode="Combination_Combined",
                        remark="Saved and ran",
                        status=status,
                        status_code=status_code,
                        combination_strategy=st.session_state.combination_results.get('strategy'),
                        combination_temperature=temperature,
                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                        rating=0,
                        step='',
                        input_query=query_text
                    )

                    generated_uid = f"Combination_AI_Combined_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                    unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                    combined_result = {
                        'prompt_name': 'AI_Combined',
                        'system_prompt': st.session_state.combination_results['combined_prompt'],
                        'response': response_text,
                        'status': status,
                        'status_code': status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': 0,
                        'remark': 'Saved and ran',
                        'edited': False,
                        'unique_id': unique_id
                    }
                    st.session_state.combination_results['combined_result'] = combined_result
                    add_result_row(
                        test_type='Combination_Combined',
                        prompt_name='AI_Combined',
                        system_prompt=st.session_state.combination_results['combined_prompt'],
                        query=query_text,
                        response=response_text,
                        status=status,
                        status_code=status_code,
                        remark='Saved and ran',
                        rating=0,
                        edited=False,
                        step='',
                        input_query=query_text,
                        combination_strategy=st.session_state.combination_results.get('strategy'),
                        combination_temperature=temperature
                    )
                    last_index = st.session_state.test_results.index[-1]
                    st.session_state.test_results.at[last_index, 'unique_id'] = unique_id
                    st.session_state.response_ratings[unique_id] = 0

                current_step += 1
                progress_bar.progress(current_step / total_steps)

            progress_bar.empty()
            st.success("Combination test completed!")

    # Display Individual Results if exist
    if st.session_state.get('combination_results') and st.session_state.combination_results.get('individual_results'):
        with individual_results_container:
            st.subheader("Individual Results")
            for i, individual_result in enumerate(st.session_state.combination_results['individual_results']):
                unique_id = individual_result['unique_id']
                status_color = "🟢" if individual_result['status'] == 'Success' else "🔴"
                prompt_name = individual_result.get('prompt_name', 'Unknown Prompt')
                with st.expander(f"{status_color} {prompt_name} - {individual_result['status']}"):
                    st.write("**System Prompt:**")
                    st.text(individual_result['system_prompt'])
                    st.write("**Query:**")
                    st.text(query_text)
                    st.write("**Response:**")

                    edited_response = st.text_area(
                        "Response (editable):",
                        value=individual_result['response'] if pd.notnull(individual_result['response']) else "",
                        height=150,
                        key=f"edit_individual_response_{i}"
                    )

                    # Dynamic rating update
                    live_rating = st.session_state.response_ratings.get(unique_id, individual_result.get('rating', 0))
                    rating_value = st.slider(
                        "Rate this response (0-10):",
                        min_value=0,
                        max_value=10,
                        value=int(live_rating),
                        key=f"rating_individual_{i}"
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
                        if st.button("💾 Save Edited Response", key=f"save_individual_edited_{i}"):
                            individual_result['response'] = edited_response
                            individual_result['edited'] = True
                            individual_result['remark'] = 'Edited and saved'
                            st.session_state.combination_results['individual_results'][i] = individual_result

                            saved_unique_id = save_export_entry(
                                prompt_name=individual_result.get('prompt_name', 'Unknown Prompt'),
                                system_prompt=individual_result['system_prompt'],
                                query=query_text,
                                response=edited_response,
                                mode="Combination_Individual",
                                remark="Edited and saved",
                                status=individual_result['status'],
                                status_code=individual_result['status_code'],
                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                combination_temperature=temperature,
                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                rating=st.session_state.response_ratings.get(unique_id, 0),
                                step=str(individual_result.get('step')) if individual_result.get('step') else '',
                                input_query=query_text,
                                edited=True
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

                    # Suggest prompt for individual
                    if st.button("🔮 Suggest Prompt for This Response", key=f"suggest_individual_{i}"):
                        with st.spinner("Generating prompt suggestion..."):
                            suggestion = suggest_func(edited_response if edited_response else individual_result['response'], query_text)
                            st.session_state[f"suggested_prompt_individual_{i}"] = suggestion
                            st.session_state[f"suggested_prompt_name_individual_{i}"] = f"Suggested_{individual_result.get('prompt_name', 'Unknown')}"

                    if st.session_state.get(f"suggested_prompt_individual_{i}"):
                        st.write("**Suggested System Prompt:**")
                        st.text_area("Suggested Prompt:", value=st.session_state[f"suggested_prompt_individual_{i}"], height=120, disabled=True)

                        prompt_name_input = st.text_input(
                            "Name this suggested prompt:",
                            value=st.session_state[f"suggested_prompt_name_individual_{i}"],
                            key=f"suggested_name_individual_{i}"
                        )

                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("💾 Save Prompt", key=f"save_suggest_individual_{i}"):
                                suggested_prompt = st.session_state[f"suggested_prompt_individual_{i}"]
                                saved_name = prompt_name_input or f"Suggested_{individual_result.get('prompt_name', 'Unknown')}"
                                st.session_state.prompts.append(suggested_prompt)
                                st.session_state.prompt_names.append(saved_name)
                                st.success(f"Saved suggested prompt: {saved_name}")
                                del st.session_state[f"suggested_prompt_individual_{i}"]
                                del st.session_state[f"suggested_prompt_name_individual_{i}"]
                                st.rerun()
                        with c2:
                            if st.button("🏃 Save & Run Prompt", key=f"save_run_suggest_individual_{i}"):
                                suggested_prompt = st.session_state[f"suggested_prompt_individual_{i}"]
                                saved_name = prompt_name_input or f"Suggested_{individual_result.get('prompt_name', 'Unknown')}"
                                st.session_state.prompts.append(suggested_prompt)
                                st.session_state.prompt_names.append(saved_name)

                                # Run the new prompt
                                with st.spinner("Running suggested prompt..."):
                                    try:
                                        run_result = call_api_func(suggested_prompt, query_text, body_template, headers, response_path)
                                        response_text = run_result.get('response', None)
                                        status = run_result.get('status', 'Failed')
                                        status_code = str(run_result.get('status_code', 'N/A'))
                                    except Exception as e:
                                        st.error(f"Error in API call for suggested prompt: {str(e)}")
                                        response_text = f"Error: {str(e)}"
                                        status = 'Failed'
                                        status_code = 'N/A'

                                debug_log(f"Result for suggested prompt '{saved_name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                                # Save to DataFrames
                                export_row_dict = {
                                    'test_type': 'Combination_Suggested',
                                    'prompt_name': saved_name,
                                    'system_prompt': suggested_prompt,
                                    'query': query_text,
                                    'response': response_text,
                                    'status': status,
                                    'status_code': status_code,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': 0,
                                    'remark': 'Saved and ran suggested prompt',
                                    'edited': False,
                                    'step': str(i + 1),
                                    'input_query': query_text,
                                    'combination_strategy': st.session_state.combination_results.get('strategy'),
                                    'combination_temperature': temperature,
                                    'slider_weights': st.session_state.combination_results.get('slider_weights')
                                }

                                maybe_uid = save_export_entry(
                                    prompt_name=saved_name,
                                    system_prompt=suggested_prompt,
                                    query=query_text,
                                    response=response_text,
                                    mode="Combination_Suggested",
                                    remark="Saved and ran suggested prompt",
                                    status=status,
                                    status_code=status_code,
                                    combination_strategy=st.session_state.combination_results.get('strategy'),
                                    combination_temperature=temperature,
                                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                                    rating=0,
                                    step=str(i + 1),
                                    input_query=query_text
                                )

                                generated_uid = f"Combination_Suggested_{saved_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                                unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                                suggested_result = {
                                    'prompt_name': saved_name,
                                    'system_prompt': suggested_prompt,
                                    'response': response_text,
                                    'status': status,
                                    'status_code': status_code,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': 0,
                                    'remark': 'Saved and ran suggested prompt',
                                    'edited': False,
                                    'unique_id': unique_id
                                }

                                # Store suggested result in session state
                                if 'suggested_results' not in st.session_state.combination_results:
                                    st.session_state.combination_results['suggested_results'] = []
                                st.session_state.combination_results['suggested_results'].append(suggested_result)

                                add_result_row(
                                    test_type='Combination_Suggested',
                                    prompt_name=saved_name,
                                    system_prompt=suggested_prompt,
                                    query=query_text,
                                    response=response_text,
                                    status=status,
                                    status_code=status_code,
                                    remark='Saved and ran suggested prompt',
                                    rating=0,
                                    edited=False,
                                    step=str(i + 1),
                                    input_query=query_text,
                                    combination_strategy=st.session_state.combination_results.get('strategy'),
                                    combination_temperature=temperature
                                )
                                last_index = st.session_state.test_results.index[-1]
                                st.session_state.test_results.at[last_index, 'unique_id'] = unique_id
                                st.session_state.response_ratings[unique_id] = 0

                                # Display response with rating
                                st.write("**Response from Suggested Prompt:**")
                                response_key = f"suggested_response_individual_{i}_{unique_id}"
                                st.text_area("Response:", value=response_text, height=150, key=response_key)
                                rating_value = st.slider(
                                    "Rate this response (0-10):",
                                    min_value=0,
                                    max_value=10,
                                    value=0,
                                    key=f"rating_suggested_individual_{i}_{unique_id}"
                                )

                                if rating_value != suggested_result['rating']:
                                    st.session_state.response_ratings[unique_id] = rating_value
                                    suggested_result['rating'] = rating_value
                                    st.session_state.combination_results['suggested_results'][-1] = suggested_result
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating_value
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                    st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'rating'] = rating_value
                                    st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                                    st.rerun()

                                st.success(f"Saved and ran suggested prompt: {saved_name}")
                                del st.session_state[f"suggested_prompt_individual_{i}"]
                                del st.session_state[f"suggested_prompt_name_individual_{i}"]
                                st.rerun()

                    st.write("**Details:**")
                    st.write(
                        f"Status Code: {individual_result.get('status_code', 'N/A')} | "
                        f"Time: {individual_result.get('timestamp', 'N/A')} | "
                        f"Rating: {st.session_state.response_ratings.get(unique_id, individual_result.get('rating', 0))}/10"
                    )

    # Display Combined Result if exists
    if st.session_state.get('combination_results') and st.session_state.combination_results.get('combined_result'):
        with combined_result_container:
            st.subheader("Combined Result")
            combined_result = st.session_state.combination_results['combined_result']
            unique_id = combined_result['unique_id']
            status_color = "🟢" if combined_result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} {combined_result.get('prompt_name', 'AI Combined')} - {combined_result['status']}"):
                st.write("**System Prompt:**")
                st.text(combined_result['system_prompt'])
                st.write("**Query:**")
                st.text(query_text)
                st.write("**Response:**")

                edited_response = st.text_area(
                    "Response (editable):",
                    value=combined_result['response'] if pd.notnull(combined_result['response']) else "",
                    height=150,
                    key="edit_combined_response"
                )

                # Dynamic rating update for combined
                live_rating = st.session_state.response_ratings.get(unique_id, combined_result.get('rating', 0))
                rating_value = st.slider(
                    "Rate this response (0-10):",
                    min_value=0,
                    max_value=10,
                    value=int(live_rating),
                    key="rating_combined"
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
                    if st.button("💾 Save Edited Response", key="save_combined_edited"):
                        combined_result['response'] = edited_response
                        combined_result['edited'] = True
                        combined_result['remark'] = 'Edited and saved'
                        st.session_state.combination_results['combined_result'] = combined_result

                        saved_unique_id = save_export_entry(
                            prompt_name=combined_result.get('prompt_name', 'AI Combined'),
                            system_prompt=combined_result['system_prompt'],
                            query=query_text,
                            response=edited_response,
                            mode="Combination_Combined",
                            remark="Edited and saved",
                            status=combined_result['status'],
                            status_code=combined_result['status_code'],
                            combination_strategy=st.session_state.combination_results.get('strategy'),
                            combination_temperature=temperature,
                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                            rating=st.session_state.response_ratings.get(unique_id, 0),
                            step=str(combined_result.get('step')) if combined_result.get('step') else '',
                            input_query=query_text,
                            edited=True
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

                # Suggest prompt for combined
                if st.button("🔮 Suggest Prompt for This Response", key="suggest_combined"):
                    with st.spinner("Generating prompt suggestion..."):
                        suggestion = suggest_func(edited_response if edited_response else combined_result['response'], query_text)
                        st.session_state["suggested_prompt_combined"] = suggestion
                        st.session_state["suggested_prompt_name_combined"] = "Suggested_AI_Combined"

                if st.session_state.get("suggested_prompt_combined"):
                    st.write("**Suggested System Prompt:**")
                    st.text_area("Suggested Prompt:", value=st.session_state["suggested_prompt_combined"], height=120, disabled=True)

                    prompt_name_input = st.text_input(
                        "Name this suggested prompt:",
                        value=st.session_state["suggested_prompt_name_combined"],
                        key="suggested_name_combined"
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("💾 Save Prompt", key="save_suggest_combined"):
                            suggested_prompt = st.session_state["suggested_prompt_combined"]
                            saved_name = prompt_name_input or "Suggested_AI_Combined"
                            st.session_state.prompts.append(suggested_prompt)
                            st.session_state.prompt_names.append(saved_name)
                            st.success(f"Saved suggested prompt: {saved_name}")
                            del st.session_state["suggested_prompt_combined"]
                            del st.session_state["suggested_prompt_name_combined"]
                            st.rerun()
                    with c2:
                        if st.button("🏃 Save & Run Prompt", key="save_run_suggest_combined"):
                            suggested_prompt = st.session_state["suggested_prompt_combined"]
                            saved_name = prompt_name_input or "Suggested_AI_Combined"
                            st.session_state.prompts.append(suggested_prompt)
                            st.session_state.prompt_names.append(saved_name)

                            # Run the new prompt
                            with st.spinner("Running suggested prompt..."):
                                try:
                                    run_result = call_api_func(suggested_prompt, query_text, body_template, headers, response_path)
                                    response_text = run_result.get('response', None)
                                    status = run_result.get('status', 'Failed')
                                    status_code = str(run_result.get('status_code', 'N/A'))
                                except Exception as e:
                                    st.error(f"Error in API call for suggested prompt: {str(e)}")
                                    response_text = f"Error: {str(e)}"
                                    status = 'Failed'
                                    status_code = 'N/A'

                            debug_log(f"Result for suggested combined prompt '{saved_name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                            # Save to DataFrames
                            export_row_dict = {
                                'test_type': 'Combination_Suggested',
                                'prompt_name': saved_name,
                                'system_prompt': suggested_prompt,
                                'query': query_text,
                                'response': response_text,
                                'status': status,
                                'status_code': status_code,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'rating': 0,
                                'remark': 'Saved and ran suggested prompt',
                                'edited': False,
                                'step': '',
                                'input_query': query_text,
                                'combination_strategy': st.session_state.combination_results.get('strategy'),
                                'combination_temperature': temperature,
                                'slider_weights': st.session_state.combination_results.get('slider_weights')
                            }

                            maybe_uid = save_export_entry(
                                prompt_name=saved_name,
                                system_prompt=suggested_prompt,
                                query=query_text,
                                response=response_text,
                                mode="Combination_Suggested",
                                remark="Saved and ran suggested prompt",
                                status=status,
                                status_code=status_code,
                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                combination_temperature=temperature,
                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                rating=0,
                                step='',
                                input_query=query_text
                            )

                            generated_uid = f"Combination_Suggested_{saved_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                            unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                            suggested_result = {
                                'prompt_name': saved_name,
                                'system_prompt': suggested_prompt,
                                'response': response_text,
                                'status': status,
                                'status_code': status_code,
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'rating': 0,
                                'remark': 'Saved and ran suggested prompt',
                                'edited': False,
                                'unique_id': unique_id
                            }

                            # Store suggested result in session state
                            if 'suggested_results' not in st.session_state.combination_results:
                                st.session_state.combination_results['suggested_results'] = []
                            st.session_state.combination_results['suggested_results'].append(suggested_result)

                            add_result_row(
                                test_type='Combination_Suggested',
                                prompt_name=saved_name,
                                system_prompt=suggested_prompt,
                                query=query_text,
                                response=response_text,
                                status=status,
                                status_code=status_code,
                                remark='Saved and ran suggested prompt',
                                rating=0,
                                edited=False,
                                step='',
                                input_query=query_text,
                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                combination_temperature=temperature
                            )
                            last_index = st.session_state.test_results.index[-1]
                            st.session_state.test_results.at[last_index, 'unique_id'] = unique_id
                            st.session_state.response_ratings[unique_id] = 0

                            # Display response with rating
                            st.write("**Response from Suggested Prompt:**")
                            response_key = f"suggested_response_combined_{unique_id}"
                            st.text_area("Response:", value=response_text, height=150, key=response_key)
                            rating_value = st.slider(
                                "Rate this response (0-10):",
                                min_value=0,
                                max_value=10,
                                value=0,
                                key=f"rating_suggested_combined_{unique_id}"
                            )

                            if rating_value != suggested_result['rating']:
                                st.session_state.response_ratings[unique_id] = rating_value
                                suggested_result['rating'] = rating_value
                                st.session_state.combination_results['suggested_results'][-1] = suggested_result
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating_value
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'rating'] = rating_value
                                st.session_state.export_data.loc[st.session_state.export_data['unique_id'] == unique_id, 'edited'] = True
                                st.rerun()

                            st.success(f"Saved and ran suggested prompt: {saved_name}")
                            del st.session_state["suggested_prompt_combined"]
                            del st.session_state["suggested_prompt_name_combined"]
                            st.rerun()

                st.write("**Details:**")
                st.write(
                    f"Status Code: {combined_result.get('status_code', 'N/A')} | "
                    f"Time: {combined_result.get('timestamp', 'N/A')} | "
                    f"Rating: {st.session_state.response_ratings.get(unique_id, combined_result.get('rating', 0))}/10"
                )

    # Display Suggested Results if exist
    if st.session_state.get('combination_results') and st.session_state.combination_results.get('suggested_results'):
        with individual_results_container:
            st.subheader("Suggested Prompt Results")
            for i, suggested_result in enumerate(st.session_state.combination_results['suggested_results']):
                unique_id = suggested_result['unique_id']
                status_color = "🟢" if suggested_result['status'] == 'Success' else "🔴"
                prompt_name = suggested_result.get('prompt_name', 'Unknown Suggested Prompt')
                with st.expander(f"{status_color} {prompt_name} - {suggested_result['status']}"):
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
                    live_rating = st.session_state.response_ratings.get(unique_id, suggested_result.get('rating', 0))
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
                                input_query=query_text,
                                edited=True
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
                    st.write(
                        f"Status Code: {suggested_result.get('status_code', 'N/A')} | "
                        f"Time: {suggested_result.get('timestamp', 'N/A')} | "
                        f"Rating: {st.session_state.response_ratings.get(unique_id, suggested_result.get('rating', 0))}/10"
                    )

    # Debug Log Download (Hidden unless needed)
    if st.session_state.get('debug_log'):
        st.download_button(
            label="Download Debug Log",
            data="\n".join(st.session_state.debug_log),
            file_name="debug_log.txt",
            mime="text/plain",
            key="download_debug_log"
        )