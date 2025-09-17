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

    # Debug: Log gemini_api_key and api_url
    st.write(f"Debug: Gemini API Key available: {bool(gemini_api_key)}")
    st.write(f"Debug: API URL: {api_url}")

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
                        st.subheader("Combined Prompt (Preview)")
                        st.markdown(f"**Generated Combined Prompt:**\n\n> {combined_prompt}")
                        st.success("✅ Prompts combined successfully!")

                except Exception as e:
                    st.error(f"Error combining prompts: {str(e)}")
                    return

            # Step 2: Test Individual Prompts
            with individual_results_container:
                st.subheader("Individual Results")
                individual_results = []
                for i, (prompt, name) in enumerate(zip(st.session_state.combination_results['individual_prompts'], st.session_state.combination_results['individual_names'])):
                    with st.spinner(f"Testing prompt '{name}'..."):
                        st.write(f"Debug: Testing Prompt '{name}' with query '{query_text}'")
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
                        except Exception as e:
                            st.error(f"Error in API call for {name}: {str(e)}")
                            response_text = f"Error: {str(e)}"
                            status = 'Failed'
                            status_code = 'N/A'

                        st.write(f"Debug: Result for '{name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

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
                            'step': i + 1,
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
                            step=i + 1,
                            input_query=query_text
                        )

                        generated_uid = f"Combination_{name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                        unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

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
                            step=i + 1,
                            input_query=query_text,
                            combination_strategy=st.session_state.combination_results.get('strategy'),
                            combination_temperature=temperature,
                            slider_weights=st.session_state.combination_results.get('slider_weights')
                        )

                        last_index = st.session_state.test_results.index[-1]
                        st.session_state.test_results.at[last_index, 'unique_id'] = unique_id
                        st.session_state.response_ratings[unique_id] = 0
                        result['unique_id'] = unique_id
                        individual_results.append(result)

                        # Display individual result immediately
                        with st.expander(f"**Individual: {name}**"):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**Query:**\n\n> {query_text}")
                                st.markdown(f"**System Prompt:**\n\n> {prompt}")
                                st.text_area(
                                    "Response (editable):",
                                    value=response_text or "",
                                    key=f"edit_individual_response_{unique_id}",
                                    height=150
                                )
                            with col2:
                                st.slider(
                                    "Rating",
                                    min_value=0,
                                    max_value=10,
                                    value=0,
                                    key=f"rating_individual_{unique_id}_{i}"
                                )
                                st.write(f"**Status Code:** {status_code}")
                                st.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                                st.write(f"**Step:** {i + 1}")

                        # Conditional delay to avoid rate-limiting
                        if i < len(st.session_state.combination_results['individual_prompts']) - 1:
                            time.sleep(0.5)  # Reduced delay, adjust based on API behavior

                st.session_state.combination_results['individual_results'] = individual_results

            # Step 3: Test Combined Prompt
            with combined_result_container:
                st.subheader("Combined Result")
                with st.spinner("Testing combined prompt..."):
                    st.write(f"Debug: Testing Combined Prompt with query '{query_text}'")
                    try:
                        combined_result = call_api_func(
                            system_prompt=st.session_state.combination_results['combined_prompt'],
                            query=query_text,
                            body_template=body_template,
                            headers=headers,
                            response_path=response_path
                        )
                        combined_response_text = combined_result.get('response', None)
                        combined_status = combined_result.get('status', 'Failed')
                        combined_status_code = str(combined_result.get('status_code', 'N/A'))
                    except Exception as e:
                        st.error(f"Error in API call for combined prompt: {str(e)}")
                        combined_response_text = f"Error: {str(e)}"
                        combined_status = 'Failed'
                        combined_status_code = 'N/A'

                    st.write(f"Debug: Result for Combined Prompt: status={combined_status}, status_code={combined_status_code}, response={combined_response_text[:50] if combined_response_text else 'None'}...")

                    export_row_dict = {
                        'test_type': 'Combination_Combined',
                        'prompt_name': 'AI_Combined',
                        'system_prompt': st.session_state.combination_results['combined_prompt'],
                        'query': query_text,
                        'response': combined_response_text,
                        'status': combined_status,
                        'status_code': combined_status_code,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'rating': 0,
                        'remark': 'Saved and ran',
                        'edited': False,
                        'step': None,
                        'input_query': query_text,
                        'combination_strategy': st.session_state.combination_results.get('strategy'),
                        'combination_temperature': temperature,
                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                    }

                    maybe_uid = save_export_entry(
                        prompt_name='AI_Combined',
                        system_prompt=st.session_state.combination_results['combined_prompt'],
                        query=query_text,
                        response=combined_response_text,
                        mode="Combination_Combined",
                        remark="Saved and ran",
                        status=combined_status,
                        status_code=combined_status_code,
                        combination_strategy=st.session_state.combination_results.get('strategy'),
                        combination_temperature=temperature,
                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                        rating=0,
                        step=None,
                        input_query=query_text
                    )

                    generated_uid = f"Combination_Combined_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_{uuid.uuid4()}"
                    combined_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=generated_uid)

                    add_result_row(
                        test_type='Combination_Combined',
                        prompt_name='AI_Combined',
                        system_prompt=st.session_state.combination_results['combined_prompt'],
                        query=query_text,
                        response=combined_response_text,
                        status=combined_status,
                        status_code=combined_status_code,
                        remark='Saved and ran',
                        rating=0,
                        edited=False,
                        step=None,
                        input_query=query_text,
                        combination_strategy=st.session_state.combination_results.get('strategy'),
                        combination_temperature=temperature,
                        slider_weights=st.session_state.combination_results.get('slider_weights')
                    )

                    last_index = st.session_state.test_results.index[-1]
                    st.session_state.test_results.at[last_index, 'unique_id'] = combined_unique_id
                    st.session_state.response_ratings[combined_unique_id] = 0
                    combined_result['unique_id'] = combined_unique_id
                    st.session_state.combination_results['combined_result'] = combined_result

                    # Display combined result
                    with st.expander("**Combined Prompt**"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Query:**\n\n> {query_text}")
                            st.markdown(f"**System Prompt:**\n\n> {st.session_state.combination_results['combined_prompt']}")
                            st.text_area(
                                "Response (editable):",
                                value=combined_response_text or "",
                                key=f"edit_combined_response_{combined_unique_id}",
                                height=150
                            )
                        with col2:
                            st.slider(
                                "Rating",
                                min_value=0,
                                max_value=10,
                                value=0,
                                key=f"rating_combined_{combined_unique_id}"
                            )
                            st.write(f"**Status Code:** {combined_status_code}")
                            st.write(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                    st.success("✅ Tests completed!")

    # Display Existing Results
    if st.session_state.get('combination_results'):
        with individual_results_container:
            st.subheader("Individual Results")
            if st.session_state.combination_results.get('individual_results'):
                for i, individual_result in enumerate(st.session_state.combination_results['individual_results']):
                    prompt_name = st.session_state.combination_results['individual_names'][i]
                    matching_rows = st.session_state.test_results[
                        (st.session_state.test_results['test_type'] == 'Combination_Individual') &
                        (st.session_state.test_results['prompt_name'] == prompt_name) &
                        (st.session_state.test_results['timestamp'] == st.session_state.combination_results['timestamp'])
                    ]
                    unique_id = matching_rows['unique_id'].iloc[0] if not matching_rows.empty else f"fallback_{i}_{uuid.uuid4()}"
                    st.write(f"Debug: Displaying result for '{prompt_name}', unique_id={unique_id}")

                    with st.expander(f"**Individual: {prompt_name}**"):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Query:**\n\n> {query_text}")
                            st.markdown(f"**System Prompt:**\n\n> {st.session_state.combination_results['individual_prompts'][i]}")
                            edited_individual_response = st.text_area(
                                "Response (editable):",
                                value=individual_result.get('response', ""),
                                key=f"edit_individual_response_{unique_id}",
                                height=150
                            )
                            if edited_individual_response != (individual_result.get('response', "") or ""):
                                if st.button("💾 Save Edited Response", key=f"save_individual_response_{unique_id}"):
                                    export_row_dict = {
                                        'test_type': 'Combination_Individual',
                                        'prompt_name': prompt_name,
                                        'system_prompt': st.session_state.combination_results['individual_prompts'][i],
                                        'query': query_text,
                                        'response': edited_individual_response,
                                        'status': individual_result.get('status', 'Failed'),
                                        'status_code': str(individual_result.get('status_code', 'N/A')),
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': st.session_state.response_ratings.get(unique_id, 0),
                                        'remark': 'Edited response',
                                        'edited': True,
                                        'step': i + 1,
                                        'input_query': query_text,
                                        'combination_strategy': st.session_state.combination_results.get('strategy'),
                                        'combination_temperature': temperature,
                                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=prompt_name,
                                        system_prompt=st.session_state.combination_results['individual_prompts'][i],
                                        query=query_text,
                                        response=edited_individual_response,
                                        mode="Combination_Individual",
                                        remark="Edited response",
                                        status=individual_result.get('status', 'Failed'),
                                        status_code=str(individual_result.get('status_code', 'N/A')),
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=temperature,
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        rating=st.session_state.response_ratings.get(unique_id, 0),
                                        step=i + 1,
                                        input_query=query_text
                                    )

                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=unique_id)
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = edited_individual_response
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = 'Edited response'
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                    st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.pop(unique_id, 0)
                                    individual_result.update({
                                        'response': edited_individual_response,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'unique_id': saved_unique_id
                                    })
                                    st.session_state.combination_results['individual_results'][i] = individual_result
                                    st.success("Response updated!")
                                    st.rerun()

                        with col2:
                            current_rating = st.session_state.response_ratings.get(unique_id, individual_result.get('rating', 0))
                            rating = st.slider(
                                "Rating",
                                min_value=0,
                                max_value=10,
                                value=int(current_rating),
                                key=f"rating_individual_{unique_id}_{i}"
                            )
                            if rating != current_rating:
                                st.session_state.response_ratings[unique_id] = rating
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                                    st.session_state.export_data.loc[
                                        st.session_state.export_data['unique_id'] == unique_id, 'rating'
                                    ] = rating
                                    st.session_state.export_data.loc[
                                        st.session_state.export_data['unique_id'] == unique_id, 'edited'
                                    ] = True
                                st.rerun()

                            if st.button("Rerun", key=f"rerun_individual_{unique_id}_{i}"):
                                with st.spinner(f"Rerunning test for {prompt_name}..."):
                                    try:
                                        result = call_api_func(
                                            system_prompt=st.session_state.combination_results['individual_prompts'][i],
                                            query=query_text,
                                            body_template=body_template,
                                            headers=headers,
                                            response_path=response_path
                                        )
                                        response_text = result.get('response', None)
                                        status = result.get('status', 'Failed')
                                        status_code = str(result.get('status_code', 'N/A'))
                                    except Exception as e:
                                        st.error(f"Error rerunning {prompt_name}: {str(e)}")
                                        response_text = f"Error: {str(e)}"
                                        status = 'Failed'
                                        status_code = 'N/A'

                                    st.write(f"Debug: Rerun result for '{prompt_name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                                    export_row_dict = {
                                        'test_type': 'Combination_Individual',
                                        'prompt_name': prompt_name,
                                        'system_prompt': st.session_state.combination_results['individual_prompts'][i],
                                        'query': query_text,
                                        'response': response_text,
                                        'status': status,
                                        'status_code': status_code,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'rating': rating,
                                        'remark': 'Rerun',
                                        'edited': True,
                                        'step': i + 1,
                                        'input_query': query_text,
                                        'combination_strategy': st.session_state.combination_results.get('strategy'),
                                        'combination_temperature': temperature,
                                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=prompt_name,
                                        system_prompt=st.session_state.combination_results['individual_prompts'][i],
                                        query=query_text,
                                        response=response_text,
                                        mode="Combination_Individual",
                                        remark="Rerun",
                                        status=status,
                                        status_code=status_code,
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=temperature,
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        rating=rating,
                                        step=i + 1,
                                        input_query=query_text
                                    )

                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=unique_id)
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = response_text
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'status'] = status
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'status_code'] = status_code
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                    st.session_state.response_ratings[saved_unique_id] = rating
                                    if saved_unique_id != unique_id:
                                        st.session_state.response_ratings.pop(unique_id, None)
                                    individual_result.update({
                                        'response': response_text,
                                        'status': status,
                                        'status_code': status_code,
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'unique_id': saved_unique_id
                                    })
                                    st.session_state.combination_results['individual_results'][i] = individual_result
                                    st.success(f"Test reran successfully for {prompt_name}!")
                                    st.rerun()

                            if st.button("✨ Suggest a better prompt", key=f"suggest_individual_{unique_id}_{i}", disabled=not gemini_api_key):
                                with st.spinner("Generating prompt suggestion..."):
                                    try:
                                        genai.configure(api_key=gemini_api_key)
                                        suggestion = suggest_func(
                                            edited_individual_response if edited_individual_response else individual_result.get('response', ""),
                                            query_text
                                        )
                                        suggested_prompt_name = f"Suggested_{prompt_name}_{i+1}"
                                        st.session_state[f"suggested_prompt_individual_{unique_id}"] = suggestion
                                        st.session_state[f"suggested_prompt_name_individual_{unique_id}"] = suggested_prompt_name
                                        st.session_state[f"edit_suggest_individual_{unique_id}_active"] = True
                                    except Exception as e:
                                        st.error(f"Error generating suggestion for {prompt_name}: {str(e)}")

                        if st.session_state.get(f"edit_suggest_individual_{unique_id}_active"):
                            st.markdown("---")
                            st.subheader("💡 Prompt Suggestion")
                            suggested_prompt = st.session_state.get(f"suggested_prompt_individual_{unique_id}", "")
                            suggested_prompt_name = st.session_state.get(f"suggested_prompt_name_individual_{unique_id}", "")

                            edited_suggestion = st.text_area(
                                "Edit the suggestion if needed:",
                                value=suggested_prompt,
                                key=f"edit_suggested_individual_{unique_id}",
                                height=100
                            )
                            edit_prompt_name = st.text_input(
                                "Name for the new prompt:",
                                value=suggested_prompt_name,
                                key=f"edit_suggested_name_individual_{unique_id}"
                            )

                            col_save, col_save_run, col_cancel = st.columns(3)
                            with col_save:
                                if st.button("💾 Save as New Prompt", key=f"save_suggested_individual_{unique_id}"):
                                    if edit_prompt_name.strip():
                                        export_row_dict = {
                                            'test_type': 'Combination_Individual',
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
                                            'step': i + 1,
                                            'input_query': query_text,
                                            'combination_strategy': st.session_state.combination_results.get('strategy'),
                                            'combination_temperature': temperature,
                                            'slider_weights': st.session_state.combination_results.get('slider_weights')
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=query_text,
                                            response='Prompt saved but not executed',
                                            mode='Combination_Individual',
                                            remark='Save only',
                                            status='Not Executed',
                                            status_code='N/A',
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                                            rating=0,
                                            step=i + 1,
                                            input_query=query_text
                                        )

                                        saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict)
                                        add_result_row(
                                            test_type='Combination_Individual',
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=query_text,
                                            response='Prompt saved but not executed',
                                            status='Not Executed',
                                            status_code='N/A',
                                            remark='Save only',
                                            rating=0,
                                            edited=False,
                                            step=i + 1,
                                            input_query=query_text,
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            slider_weights=st.session_state.combination_results.get('slider_weights')
                                        )

                                        last_index = st.session_state.test_results.index[-1]
                                        st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id
                                        st.session_state.response_ratings[saved_unique_id] = 0
                                        st.session_state.prompts.append(edited_suggestion)
                                        st.session_state.prompt_names.append(edit_prompt_name.strip())
                                        st.session_state[f"edit_suggest_individual_{unique_id}_active"] = False
                                        del st.session_state[f"suggested_prompt_individual_{unique_id}"]
                                        del st.session_state[f"suggested_prompt_name_individual_{unique_id}"]
                                        st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                        st.rerun()
                                    else:
                                        st.error("Please provide a prompt name")

                            with col_save_run:
                                if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggested_individual_{unique_id}"):
                                    if edit_prompt_name.strip():
                                        st.session_state.prompts.append(edited_suggestion)
                                        st.session_state.prompt_names.append(edit_prompt_name.strip())
                                        with st.spinner("Running new prompt..."):
                                            try:
                                                result = call_api_func(
                                                    system_prompt=edited_suggestion,
                                                    query=query_text,
                                                    body_template=body_template,
                                                    headers=headers,
                                                    response_path=response_path
                                                )
                                                response_text = result.get('response', None)
                                                status = result.get('status', 'Failed')
                                                status_code = str(result.get('status_code', 'N/A'))
                                            except Exception as e:
                                                st.error(f"Error running suggested prompt: {str(e)}")
                                                response_text = f"Error: {str(e)}"
                                                status = 'Failed'
                                                status_code = 'N/A'

                                            st.write(f"Debug: Run suggested prompt '{edit_prompt_name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                                            export_row_dict = {
                                                'test_type': 'Combination_Individual',
                                                'prompt_name': edit_prompt_name.strip(),
                                                'system_prompt': edited_suggestion,
                                                'query': query_text,
                                                'response': response_text,
                                                'status': status,
                                                'status_code': status_code,
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'rating': 0,
                                                'remark': 'Saved and ran',
                                                'edited': False,
                                                'step': i + 1,
                                                'input_query': query_text,
                                                'combination_strategy': st.session_state.combination_results.get('strategy'),
                                                'combination_temperature': temperature,
                                                'slider_weights': st.session_state.combination_results.get('slider_weights')
                                            }

                                            maybe_uid = save_export_entry(
                                                prompt_name=edit_prompt_name.strip(),
                                                system_prompt=edited_suggestion,
                                                query=query_text,
                                                response=response_text,
                                                mode='Combination_Individual',
                                                remark='Saved and ran',
                                                status=status,
                                                status_code=status_code,
                                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                                combination_temperature=temperature,
                                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                                rating=0,
                                                step=i + 1,
                                                input_query=query_text
                                            )

                                            saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict)
                                            add_result_row(
                                                test_type='Combination_Individual',
                                                prompt_name=edit_prompt_name.strip(),
                                                system_prompt=edited_suggestion,
                                                query=query_text,
                                                response=response_text,
                                                status=status,
                                                status_code=status_code,
                                                remark='Saved and ran',
                                                rating=0,
                                                edited=False,
                                                step=i + 1,
                                                input_query=query_text,
                                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                                combination_temperature=temperature,
                                                slider_weights=st.session_state.combination_results.get('slider_weights')
                                            )

                                            last_index = st.session_state.test_results.index[-1]
                                            st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id
                                            st.session_state.response_ratings[saved_unique_id] = 0
                                            st.session_state[f"edit_suggest_individual_{unique_id}_active"] = False
                                            del st.session_state[f"suggested_prompt_individual_{unique_id}"]
                                            del st.session_state[f"suggested_prompt_name_individual_{unique_id}"]
                                            st.success(f"Saved and ran new prompt: {edit_prompt_name.strip()}")
                                            st.rerun()
                                    else:
                                        st.error("Please provide a prompt name")

                            with col_cancel:
                                if st.button("Cancel", key=f"cancel_suggested_individual_{unique_id}"):
                                    st.session_state[f"edit_suggest_individual_{unique_id}_active"] = False
                                    del st.session_state[f"suggested_prompt_individual_{unique_id}"]
                                    del st.session_state[f"suggested_prompt_name_individual_{unique_id}"]
                                    st.rerun()

                        st.write("**Details:**")
                        st.write(
                            f"Status Code: {individual_result.get('status_code', 'N/A')} | "
                            f"Time: {individual_result.get('timestamp', 'N/A')} | "
                            f"Step: {i + 1} | "
                            f"Rating: {st.session_state.response_ratings.get(unique_id, individual_result.get('rating', 0))}/10 "
                            f"({st.session_state.response_ratings.get(unique_id, individual_result.get('rating', 0))*10}%)"
                        )

        with combined_result_container:
            st.subheader("Combined Result")
            if st.session_state.combination_results.get('combined_result'):
                combined_result = st.session_state.combination_results['combined_result']
                unique_id = combined_result.get('unique_id', f"combined_{uuid.uuid4()}")
                st.write(f"Debug: Displaying combined result, unique_id={unique_id}")

                with st.expander("**Combined Prompt**"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Query:**\n\n> {query_text}")
                        st.markdown(f"**System Prompt:**\n\n> {st.session_state.combination_results['combined_prompt']}")
                        edited_combined_response = st.text_area(
                            "Response (editable):",
                            value=combined_result.get('response', ""),
                            key=f"edit_combined_response_{unique_id}",
                            height=150
                        )
                        if edited_combined_response != (combined_result.get('response', "") or ""):
                            if st.button("💾 Save Edited Response", key=f"save_combined_response_{unique_id}"):
                                export_row_dict = {
                                    'test_type': 'Combination_Combined',
                                    'prompt_name': 'AI_Combined',
                                    'system_prompt': st.session_state.combination_results['combined_prompt'],
                                    'query': query_text,
                                    'response': edited_combined_response,
                                    'status': combined_result.get('status', 'Failed'),
                                    'status_code': str(combined_result.get('status_code', 'N/A')),
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': st.session_state.response_ratings.get(unique_id, 0),
                                    'remark': 'Edited response',
                                    'edited': True,
                                    'step': None,
                                    'input_query': query_text,
                                    'combination_strategy': st.session_state.combination_results.get('strategy'),
                                    'combination_temperature': temperature,
                                    'slider_weights': st.session_state.combination_results.get('slider_weights')
                                }

                                maybe_uid = save_export_entry(
                                    prompt_name='AI_Combined',
                                    system_prompt=st.session_state.combination_results['combined_prompt'],
                                    query=query_text,
                                    response=edited_combined_response,
                                    mode="Combination_Combined",
                                    remark="Edited response",
                                    status=combined_result.get('status', 'Failed'),
                                    status_code=str(combined_result.get('status_code', 'N/A')),
                                    combination_strategy=st.session_state.combination_results.get('strategy'),
                                    combination_temperature=temperature,
                                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                                    rating=st.session_state.response_ratings.get(unique_id, 0),
                                    step=None,
                                    input_query=query_text
                                )

                                saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=unique_id)
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = edited_combined_response
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'remark'] = 'Edited response'
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.pop(unique_id, 0)
                                combined_result['unique_id'] = saved_unique_id
                                st.session_state.combination_results['combined_result'] = combined_result
                                st.success("Response updated!")
                                st.rerun()

                    with col2:
                        current_rating = st.session_state.response_ratings.get(unique_id, combined_result.get('rating', 0))
                        rating = st.slider(
                            "Rating",
                            min_value=0,
                            max_value=10,
                            value=int(current_rating),
                            key=f"rating_combined_{unique_id}"
                        )
                        if rating != current_rating:
                            st.session_state.response_ratings[unique_id] = rating
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'rating'] = rating
                            st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'edited'] = True
                            if 'export_data' in st.session_state and not st.session_state.export_data.empty:
                                st.session_state.export_data.loc[
                                    st.session_state.export_data['unique_id'] == unique_id, 'rating'
                                ] = rating
                                st.session_state.export_data.loc[
                                    st.session_state.export_data['unique_id'] == unique_id, 'edited'
                                ] = True
                            st.rerun()

                        if st.button("Rerun", key=f"rerun_combined_{unique_id}"):
                            with st.spinner("Rerunning test..."):
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
                                    st.error(f"Error rerunning combined prompt: {str(e)}")
                                    response_text = f"Error: {str(e)}"
                                    status = 'Failed'
                                    status_code = 'N/A'

                                st.write(f"Debug: Rerun result for combined prompt: status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                                export_row_dict = {
                                    'test_type': 'Combination_Combined',
                                    'prompt_name': 'AI_Combined',
                                    'system_prompt': st.session_state.combination_results['combined_prompt'],
                                    'query': query_text,
                                    'response': response_text,
                                    'status': status,
                                    'status_code': status_code,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'rating': rating,
                                    'remark': 'Rerun',
                                    'edited': True,
                                    'step': None,
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
                                    remark="Rerun",
                                    status=status,
                                    status_code=status_code,
                                    combination_strategy=st.session_state.combination_results.get('strategy'),
                                    combination_temperature=temperature,
                                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                                    rating=rating,
                                    step=None,
                                    input_query=query_text
                                )

                                saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict, generated_uid=unique_id)
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'response'] = response_text
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'status'] = status
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'status_code'] = status_code
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                st.session_state.response_ratings[saved_unique_id] = rating
                                if saved_unique_id != unique_id:
                                    st.session_state.response_ratings.pop(unique_id, None)
                                combined_result.update({
                                    'response': response_text,
                                    'status': status,
                                    'status_code': status_code,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'unique_id': saved_unique_id
                                })
                                st.session_state.combination_results['combined_result'] = combined_result
                                st.success("Test reran successfully!")
                                st.rerun()

                        if st.button("✨ Suggest a better prompt", key=f"suggest_combined_{unique_id}", disabled=not gemini_api_key):
                            with st.spinner("Generating prompt suggestion..."):
                                try:
                                    genai.configure(api_key=gemini_api_key)
                                    suggestion = suggest_func(
                                        edited_combined_response if edited_combined_response else combined_result.get('response', ""),
                                        query_text
                                    )
                                    suggested_prompt_name = f"Suggested_AI_Combined"
                                    st.session_state[f"suggested_prompt_combined_{unique_id}"] = suggestion
                                    st.session_state[f"suggested_prompt_name_combined_{unique_id}"] = suggested_prompt_name
                                    st.session_state[f"edit_suggest_combined_{unique_id}_active"] = True
                                except Exception as e:
                                    st.error(f"Error generating suggestion for combined prompt: {str(e)}")

                    if st.session_state.get(f"edit_suggest_combined_{unique_id}_active"):
                        st.markdown("---")
                        st.subheader("💡 Prompt Suggestion")
                        suggested_prompt = st.session_state.get(f"suggested_prompt_combined_{unique_id}", "")
                        suggested_prompt_name = st.session_state.get(f"suggested_prompt_name_combined_{unique_id}", "")

                        edited_suggestion = st.text_area(
                            "Edit the suggestion if needed:",
                            value=suggested_prompt,
                            key=f"edit_suggested_combined_{unique_id}",
                            height=100
                        )
                        edit_prompt_name = st.text_input(
                            "Name for the new prompt:",
                            value=suggested_prompt_name,
                            key=f"edit_suggested_name_combined_{unique_id}"
                        )

                        col_save, col_save_run, col_cancel = st.columns(3)
                        with col_save:
                            if st.button("💾 Save as New Prompt", key=f"save_suggested_combined_{unique_id}"):
                                if edit_prompt_name.strip():
                                    export_row_dict = {
                                        'test_type': 'Combination_Combined',
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
                                        'step': None,
                                        'input_query': query_text,
                                        'combination_strategy': st.session_state.combination_results.get('strategy'),
                                        'combination_temperature': temperature,
                                        'slider_weights': st.session_state.combination_results.get('slider_weights')
                                    }

                                    maybe_uid = save_export_entry(
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        mode='Combination_Combined',
                                        remark='Save only',
                                        status='Not Executed',
                                        status_code='N/A',
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=temperature,
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        rating=0,
                                        step=None,
                                        input_query=query_text
                                    )

                                    saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict)
                                    add_result_row(
                                        test_type='Combination_Combined',
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        status='Not Executed',
                                        status_code='N/A',
                                        remark='Save only',
                                        rating=0,
                                        edited=False,
                                        step=None,
                                        input_query=query_text,
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=temperature,
                                        slider_weights=st.session_state.combination_results.get('slider_weights')
                                    )

                                    last_index = st.session_state.test_results.index[-1]
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

                        with col_save_run:
                            if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggested_combined_{unique_id}"):
                                if edit_prompt_name.strip():
                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(edit_prompt_name.strip())
                                    with st.spinner("Running new prompt..."):
                                        try:
                                            result = call_api_func(
                                                system_prompt=edited_suggestion,
                                                query=query_text,
                                                body_template=body_template,
                                                headers=headers,
                                                response_path=response_path
                                            )
                                            response_text = result.get('response', None)
                                            status = result.get('status', 'Failed')
                                            status_code = str(result.get('status_code', 'N/A'))
                                        except Exception as e:
                                            st.error(f"Error running suggested prompt: {str(e)}")
                                            response_text = f"Error: {str(e)}"
                                            status = 'Failed'
                                            status_code = 'N/A'

                                        st.write(f"Debug: Run suggested prompt '{edit_prompt_name}': status={status}, status_code={status_code}, response={response_text[:50] if response_text else 'None'}...")

                                        export_row_dict = {
                                            'test_type': 'Combination_Combined',
                                            'prompt_name': edit_prompt_name.strip(),
                                            'system_prompt': edited_suggestion,
                                            'query': query_text,
                                            'response': response_text,
                                            'status': status,
                                            'status_code': status_code,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'rating': 0,
                                            'remark': 'Saved and ran',
                                            'edited': False,
                                            'step': None,
                                            'input_query': query_text,
                                            'combination_strategy': st.session_state.combination_results.get('strategy'),
                                            'combination_temperature': temperature,
                                            'slider_weights': st.session_state.combination_results.get('slider_weights')
                                        }

                                        maybe_uid = save_export_entry(
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=query_text,
                                            response=response_text,
                                            mode='Combination_Combined',
                                            remark='Saved and ran',
                                            status=status,
                                            status_code=status_code,
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                                            rating=0,
                                            step=None,
                                            input_query=query_text
                                        )

                                        saved_unique_id = normalize_saved_uid(maybe_uid, export_row_dict)
                                        add_result_row(
                                            test_type='Combination_Combined',
                                            prompt_name=edit_prompt_name.strip(),
                                            system_prompt=edited_suggestion,
                                            query=query_text,
                                            response=response_text,
                                            status=status,
                                            status_code=status_code,
                                            remark='Saved and ran',
                                            rating=0,
                                            edited=False,
                                            step=None,
                                            input_query=query_text,
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=temperature,
                                            slider_weights=st.session_state.combination_results.get('slider_weights')
                                        )

                                        last_index = st.session_state.test_results.index[-1]
                                        st.session_state.test_results.at[last_index, 'unique_id'] = saved_unique_id
                                        st.session_state.response_ratings[saved_unique_id] = 0
                                        st.session_state[f"edit_suggest_combined_{unique_id}_active"] = False
                                        del st.session_state[f"suggested_prompt_combined_{unique_id}"]
                                        del st.session_state[f"suggested_prompt_name_combined_{unique_id}"]
                                        st.success(f"Saved and ran new prompt: {edit_prompt_name.strip()}")
                                        st.rerun()
                                else:
                                    st.error("Please provide a prompt name")

                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_suggested_combined_{unique_id}"):
                                st.session_state[f"edit_suggest_combined_{unique_id}_active"] = False
                                del st.session_state[f"suggested_prompt_combined_{unique_id}"]
                                del st.session_state[f"suggested_prompt_name_combined_{unique_id}"]
                                st.rerun()

                    st.write("**Details:**")
                    st.write(
                        f"Status Code: {combined_result.get('status_code', 'N/A')} | "
                        f"Time: {combined_result.get('timestamp', 'N/A')} | "
                        f"Rating: {st.session_state.response_ratings.get(unique_id, combined_result.get('rating', 0))}/10 "
                        f"({st.session_state.response_ratings.get(unique_id, combined_result.get('rating', 0))*10}%)"
                    )
            else:
                st.info("No combined results to display yet.")