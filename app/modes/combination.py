import streamlit as st
import google.generativeai as genai
from datetime import datetime
from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response

def render_prompt_combination(api_url, query_text, body_template, headers, response_path, call_api_func, suggest_func, gemini_api_key):
    st.header("🤝 Prompt Combination")
    
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
    
    if st.session_state.prompts:
        ensure_prompt_names()
        selected_prompts = st.multiselect(
            "Choose prompts to combine:",
            options=list(range(len(st.session_state.prompts))),
            format_func=lambda x: f"{st.session_state.prompt_names[x]}: {st.session_state.prompts[x][:50]}...",
            default=list(range(min(2, len(st.session_state.prompts))))
        )
        
        if selected_prompts != st.session_state.last_selected_prompts:
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
                
                if not st.session_state.slider_weights or len(st.session_state.slider_weights) != len(selected_prompts):
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
    
    if st.button("🤖 Combine Prompts with AI", type="primary", disabled=not (gemini_api_key and st.session_state.prompts and selected_prompts)):
        if not gemini_api_key:
            st.error("Please configure Gemini API key")
        elif not selected_prompts or len(selected_prompts) < 2:
            st.error("Please select at least 2 prompts to combine")
        elif combination_strategy == "Slider - Custom influence weights" and sum(st.session_state.slider_weights.get(idx, 0) for idx in selected_prompts) == 0:
            st.error("Please set at least one prompt weight > 0%")
        else:
            try:
                gemini_temperature = (temperature / 100.0) * 2.0
                
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
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
                
                generation_config = genai.types.GenerationConfig(
                    temperature=gemini_temperature
                )
                
                with st.spinner("AI is combining prompts..."):
                    response = model.generate_content(combination_prompt, generation_config=generation_config)
                    combined_prompt = response.text
                    
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
                    
                    st.success("✅ Prompts combined successfully!")
                    
            except Exception as e:
                st.error(f"Error combining prompts: {str(e)}")
    
    if st.session_state.combination_results:
        if st.button("🧪 Test Combined vs Individual Prompts", type="primary", disabled=not (api_url and query_text)):
            if not api_url or not query_text:
                st.error("Please configure API endpoint and enter a query")
            else:
                with st.spinner("Testing individual prompts..."):
                    individual_results = []
                    
                    for i, (prompt, name) in enumerate(zip(st.session_state.combination_results['individual_prompts'], st.session_state.combination_results['individual_names'])):
                        result = call_api_func(prompt, query_text, body_template, headers, response_path)
                        result.update({
                            'prompt_index': i + 1,
                            'prompt_name': name,
                            'system_prompt': prompt,
                            'query': query_text,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'edited': False,
                            'remark': 'Saved and ran'
                        })
                        individual_results.append(result)
                
                with st.spinner("Testing combined prompt..."):
                    combined_result = call_api_func(st.session_state.combination_results['combined_prompt'], query_text, body_template, headers, response_path)
                    combined_result.update({
                        'system_prompt': st.session_state.combination_results['combined_prompt'],
                        'query': query_text,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'edited': False,
                        'remark': 'Saved and ran'
                    })
                
                st.session_state.combination_results['individual_results'] = individual_results
                st.session_state.combination_results['combined_result'] = combined_result
                
                st.success("✅ Testing completed!")

    if st.session_state.combination_results:
        st.subheader("🎯 Combination Results")
        
        st.subheader("🤖 AI-Generated Combined Prompt")
        combined_prompt_text = st.text_area(
            "Combined Prompt (editable):",
            value=st.session_state.combination_results['combined_prompt'],
            height=200,
            key="edit_combined_prompt"
        )
        
        if combined_prompt_text != st.session_state.combination_results['combined_prompt']:
            if st.button("💾 Save Combined Prompt"):
                st.session_state.combination_results['combined_prompt'] = combined_prompt_text
                if st.session_state.combination_results.get('combined_result'):
                    st.session_state.combination_results['combined_result']['system_prompt'] = combined_prompt_text
                    st.session_state.combination_results['combined_result']['edited'] = True
                st.success("Combined prompt updated!")
                st.rerun()
        
        st.write(f"**Strategy:** {st.session_state.combination_results.get('strategy')}")
        st.write(f"**Temperature:** {st.session_state.combination_results.get('temperature', 50)}%")
        
        if st.session_state.combination_results.get('slider_weights'):
            st.write("**Influence Weights Used:**")
            for idx, weight in st.session_state.combination_results['slider_weights'].items():
                if idx in st.session_state.combination_results['selected_indices']:
                    name = st.session_state.combination_results['individual_names'][st.session_state.combination_results['selected_indices'].index(idx)]
                    st.write(f"- {name}: {weight}%")
        
        if st.session_state.combination_results.get('individual_results') and st.session_state.combination_results.get('combined_result'):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📄 Individual Prompt Results")
                for j, result in enumerate(st.session_state.combination_results['individual_results']):
                    status_color = "🟢" if result['status'] == 'Success' else "🔴"
                    with st.expander(f"{status_color} {result['prompt_name']}"):
                        edited_individual_response = st.text_area(
                            "Response (editable):", 
                            value=result['response'], 
                            height=150, 
                            key=f"edit_individual_{j}"
                        )
                        
                        rating = st.slider(
                            "Rate this response (0-10):",
                            min_value=0,
                            max_value=10,
                            value=st.session_state.response_ratings.get(f"combination_individual_{j}", 5),
                            key=f"rating_individual_{j}"
                        )
                        st.session_state.response_ratings[f"combination_individual_{j}"] = rating
                        
                        if edited_individual_response != result['response']:
                            col_save, col_reverse = st.columns(2)
                            with col_save:
                                if st.button(f"💾 Save Response", key=f"save_individual_{j}"):
                                    st.session_state.combination_results['individual_results'][j]['response'] = edited_individual_response
                                    st.session_state.combination_results['individual_results'][j]['edited'] = True
                                    st.success("Response updated!")
                                    st.rerun()
                            with col_reverse:
                                if st.button(f"🔄 Reverse Prompt", key=f"reverse_individual_{j}"):
                                    with st.spinner("Generating updated prompt..."):
                                        suggestion = suggest_func(edited_individual_response, query_text)
                                        source_idx = st.session_state.combination_results['selected_indices'][j]
                                        st.session_state.prompts[source_idx] = suggestion
                                        st.session_state.combination_results['individual_prompts'][j] = suggestion
                                        st.session_state.combination_results['individual_results'][j]['system_prompt'] = suggestion
                                        st.session_state.combination_results['individual_results'][j]['edited'] = True
                                        st.session_state.combination_results['individual_results'][j]['remark'] = 'Saved and ran'
                                        st.success("Prompt updated based on edited response!")
                                        st.rerun()
                        
                        if st.button(f"🔮 Suggest Prompt", key=f"suggest_individual_{j}"):
                            with st.spinner("Generating prompt suggestion..."):
                                suggestion = suggest_func(edited_individual_response, query_text)
                                st.write("**Suggested System Prompt:**")
                                suggested_prompt = st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_individual_{j}", disabled=True)
                                
                                col_save, col_save_run, col_edit = st.columns(3)
                                with col_save:
                                    prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key=f"suggest_individual_name_{j}")
                                    if st.button("💾 Save as Prompt", key=f"save_suggest_individual_{j}"):
                                        if prompt_name.strip():
                                            st.session_state.prompts.append(suggestion)
                                            st.session_state.prompt_names.append(prompt_name.strip())
                                            st.session_state.test_results.append({
                                                'prompt_name': prompt_name.strip(),
                                                'system_prompt': suggestion,
                                                'query': query_text,
                                                'response': 'Prompt saved but not executed',
                                                'status': 'Not Executed',
                                                'status_code': 'N/A',
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'edited': False,
                                                'remark': 'Save only'
                                            })
                                            st.success(f"Saved as new prompt: {prompt_name.strip()}")
                                            st.rerun()
                                        else:
                                            st.error("Please provide a prompt name")
                                with col_save_run:
                                    run_prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key=f"suggest_individual_run_name_{j}")
                                    if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_individual_{j}"):
                                        if run_prompt_name.strip():
                                            st.session_state.prompts.append(suggestion)
                                            st.session_state.prompt_names.append(run_prompt_name.strip())
                                            with st.spinner("Running new prompt..."):
                                                result = call_api_func(suggestion, query_text, body_template, headers, response_path)
                                                result.update({
                                                    'prompt_name': run_prompt_name.strip(),
                                                    'system_prompt': suggestion,
                                                    'query': query_text,
                                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    'edited': False,
                                                    'remark': 'Saved and ran'
                                                })
                                                st.session_state.test_results.append(result)
                                            st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                            st.rerun()
                                        else:
                                            st.error("Please provide a prompt name")
                                with col_edit:
                                    if st.button("✏️ Edit", key=f"edit_suggest_individual_{j}"):
                                        st.session_state[f"edit_suggest_individual_{j}_active"] = True
                                
                                if st.session_state.get(f"edit_suggest_individual_{j}_active", False):
                                    edited_suggestion = st.text_area("Edit Suggested Prompt:", value=suggestion, height=100, key=f"edit_suggested_individual_{j}")
                                    if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_individual_{j}"):
                                        prompt_name = st.text_input("Prompt Name for Edited Prompt:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key=f"edit_suggest_individual_name_{j}")
                                        if prompt_name.strip():
                                            st.session_state.prompts.append(edited_suggestion)
                                            st.session_state.prompt_names.append(prompt_name.strip())
                                            st.session_state.test_results.append({
                                                'prompt_name': prompt_name.strip(),
                                                'system_prompt': edited_suggestion,
                                                'query': query_text,
                                                'response': 'Prompt saved but not executed',
                                                'status': 'Not Executed',
                                                'status_code': 'N/A',
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'edited': False,
                                                'remark': 'Save only'
                                            })
                                            st.session_state[f"edit_suggest_individual_{j}_active"] = False
                                            st.success(f"Saved edited prompt as: {prompt_name.strip()}")
                                            st.rerun()
                                        else:
                                            st.error("Please provide a prompt name")
            
            with col2:
                st.subheader("🤝 Combined Prompt Result")
                combined_result = st.session_state.combination_results['combined_result']
                status_color = "🟢" if combined_result['status'] == 'Success' else "🔴"
                
                st.markdown(f"**Status:** {status_color} {combined_result['status']}")
                
                edited_combined_response = st.text_area(
                    "Combined Response (editable):", 
                    value=combined_result['response'], 
                    height=300, 
                    key="edit_combined_response"
                )
                
                rating = st.slider(
                    "Rate this response (0-10):",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.response_ratings.get("combination_combined", 5),
                    key="rating_combined"
                )
                st.session_state.response_ratings["combination_combined"] = rating
                
                if edited_combined_response != combined_result['response']:
                    col_save, col_reverse = st.columns(2)
                    with col_save:
                        if st.button("💾 Save Combined Response"):
                            st.session_state.combination_results['combined_result']['response'] = edited_combined_response
                            st.session_state.combination_results['combined_result']['edited'] = True
                            st.success("Combined response updated!")
                            st.rerun()
                    with col_reverse:
                        if st.button("🔄 Reverse Prompt for Combined"):
                            with st.spinner("Generating updated prompt..."):
                                suggestion = suggest_func(edited_combined_response, query_text)
                                st.session_state.combination_results['combined_prompt'] = suggestion
                                st.session_state.combination_results['combined_result']['system_prompt'] = suggestion
                                st.session_state.combination_results['combined_result']['edited'] = True
                                st.session_state.combination_results['combined_result']['remark'] = 'Saved and ran'
                                st.success("Combined prompt updated based on edited response!")
                                st.rerun()
                
                if st.button("🔮 Suggest Prompt for Combined Response"):
                    with st.spinner("Generating prompt suggestion..."):
                        suggestion = suggest_func(edited_combined_response, query_text)
                        st.write("**Suggested System Prompt:**")
                        suggested_prompt = st.text_area("Suggested Prompt:", value=suggestion, height=100, key="suggested_combined", disabled=True)
                        
                        col_save, col_save_run, col_edit = st.columns(3)
                        with col_save:
                            prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key="suggest_combined_name")
                            if st.button("💾 Save as Prompt", key="save_suggest_combined"):
                                if prompt_name.strip():
                                    st.session_state.prompts.append(suggestion)
                                    st.session_state.prompt_names.append(prompt_name.strip())
                                    st.session_state.test_results.append({
                                        'prompt_name': prompt_name.strip(),
                                        'system_prompt': suggestion,
                                        'query': query_text,
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'edited': False,
                                        'remark': 'Save only'
                                    })
                                    st.success(f"Saved as new prompt: {prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")
                        with col_save_run:
                            run_prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key="suggest_combined_run_name")
                            if st.button("🏃 Save as Prompt and Run", key="save_run_suggest_combined"):
                                if run_prompt_name.strip():
                                    st.session_state.prompts.append(suggestion)
                                    st.session_state.prompt_names.append(run_prompt_name.strip())
                                    with st.spinner("Running new prompt..."):
                                        result = call_api_func(suggestion, query_text, body_template, headers, response_path)
                                        result.update({
                                            'prompt_name': run_prompt_name.strip(),
                                            'system_prompt': suggestion,
                                            'query': query_text,
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'edited': False,
                                            'remark': 'Saved and ran'
                                        })
                                        st.session_state.test_results.append(result)
                                    st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")
                        with col_edit:
                            if st.button("✏️ Edit", key="edit_suggest_combined"):
                                st.session_state["edit_suggest_combined_active"] = True
                        
                        if st.session_state.get("edit_suggest_combined_active", False):
                            edited_suggestion = st.text_area("Edit Suggested Prompt:", value=suggestion, height=100, key="edit_suggested_combined")
                            if st.button("💾 Save Edited Prompt", key="save_edited_suggest_combined"):
                                prompt_name = st.text_input("Prompt Name for Edited Prompt:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key="edit_suggest_combined_name")
                                if prompt_name.strip():
                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(prompt_name.strip())
                                    st.session_state.test_results.append({
                                        'prompt_name': prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': query_text,
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'edited': False,
                                        'remark': 'Save only'
                                    })
                                    st.session_state["edit_suggest_combined_active"] = False
                                    st.success(f"Saved edited prompt as: {prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")