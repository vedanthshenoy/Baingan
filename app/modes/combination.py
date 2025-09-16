import streamlit as st
import google.generativeai as genai
from datetime import datetime
import pandas as pd
from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response
from app.export import save_export_entry

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
    
    selected_prompts = []
    
    if st.session_state.prompts:
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
    
    # Combined button to combine and then test
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
            try:
                genai.configure(api_key=gemini_api_key)
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
                    st.session_state.suggested_prompt = None
                    st.session_state.suggested_prompt_name = None
                    
                    st.success("✅ Prompts combined successfully!")
                    
            except Exception as e:
                st.error(f"Error combining prompts: {str(e)}")
                return # Exit if combine fails
            
            # Step 2: Run Tests
            if 'test_results' not in st.session_state:
                st.session_state.test_results = pd.DataFrame(columns=[
                    'unique_id', 'prompt_name', 'system_prompt', 'query', 'response', 
                    'status', 'status_code', 'timestamp', 'edited', 'remark', 'rating'
                ])

            if 'response_ratings' not in st.session_state:
                st.session_state.response_ratings = {}

            with st.spinner("Testing individual prompts..."):
                individual_results = []
                for i, (prompt, name) in enumerate(zip(st.session_state.combination_results['individual_prompts'], st.session_state.combination_results['individual_names'])):
                    result = call_api_func(prompt, query_text, body_template, headers, response_path)
                    unique_id = save_export_entry(
                        prompt_name=name,
                        system_prompt=prompt,
                        query=query_text,
                        response=result['response'] if 'response' in result else None,
                        mode="Combination_Individual",
                        remark="Saved and ran",
                        status=result['status'],
                        status_code=result.get('status_code', 'N/A'),
                        combination_strategy=st.session_state.combination_results.get('strategy'),
                        combination_temperature=st.session_state.combination_results.get('temperature'),
                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                        rating=0
                    )
                    st.session_state.response_ratings[unique_id] = 0
                    new_result = pd.DataFrame([{
                        'unique_id': unique_id,
                        'prompt_name': name,
                        'system_prompt': prompt,
                        'query': query_text,
                        'response': result['response'] if 'response' in result else None,
                        'status': result['status'],
                        'status_code': str(result.get('status_code', 'N/A')),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'edited': False,
                        'remark': 'Saved and ran',
                        'rating': 0
                    }])
                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                    individual_results.append(new_result.to_dict('records')[0])
            
            with st.spinner("Testing combined prompt..."):
                combined_result = call_api_func(st.session_state.combination_results['combined_prompt'], query_text, body_template, headers, response_path)
                unique_id = save_export_entry(
                    prompt_name="AI_Combined",
                    system_prompt=st.session_state.combination_results['combined_prompt'],
                    query=query_text,
                    response=combined_result['response'] if 'response' in combined_result else None,
                    mode="Combination_Combined",
                    remark="Saved and ran",
                    status=combined_result['status'],
                    status_code=combined_result.get('status_code', 'N/A'),
                    combination_strategy=st.session_state.combination_results.get('strategy'),
                    combination_temperature=st.session_state.combination_results.get('temperature'),
                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                    rating=0
                )
                st.session_state.response_ratings[unique_id] = 0
                new_combined_result = pd.DataFrame([{
                    'unique_id': unique_id,
                    'prompt_name': "AI_Combined",
                    'system_prompt': st.session_state.combination_results['combined_prompt'],
                    'query': query_text,
                    'response': combined_result['response'] if 'response' in combined_result else None,
                    'status': combined_result['status'],
                    'status_code': str(combined_result.get('status_code', 'N/A')),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'edited': False,
                    'remark': 'Saved and ran',
                    'rating': 0
                }])
                st.session_state.test_results = pd.concat([st.session_state.test_results, new_combined_result], ignore_index=True)
            
            st.session_state.combination_results['individual_results'] = individual_results
            st.session_state.combination_results['combined_result'] = new_combined_result.to_dict('records')[0]
            
            st.success("✅ Testing completed!")

    if st.session_state.get('combination_results'):
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
                    unique_id = save_export_entry(
                        prompt_name="AI_Combined",
                        system_prompt=combined_prompt_text,
                        query=query_text,
                        response=st.session_state.combination_results['combined_result']['response'] if 'response' in st.session_state.combination_results['combined_result'] else None,
                        mode="Combination_Combined",
                        remark="Edited and saved",
                        status=st.session_state.combination_results['combined_result']['status'],
                        status_code=st.session_state.combination_results['combined_result'].get('status_code', 'N/A'),
                        combination_strategy=st.session_state.combination_results.get('strategy'),
                        combination_temperature=st.session_state.combination_results.get('temperature'),
                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                        edited=True
                    )
                    new_result = pd.DataFrame([{
                        'unique_id': unique_id,
                        'prompt_name': "AI_Combined",
                        'system_prompt': combined_prompt_text,
                        'query': query_text,
                        'response': st.session_state.combination_results['combined_result']['response'] if 'response' in st.session_state.combination_results['combined_result'] else None,
                        'status': st.session_state.combination_results['combined_result']['status'],
                        'status_code': str(st.session_state.combination_results['combined_result'].get('status_code', 'N/A')),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'edited': True,
                        'remark': 'Edited and saved'
                    }])
                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
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

                        if new_rating != current_rating:
                            st.session_state.response_ratings[unique_id] = new_rating
                            
                            saved_unique_id = save_export_entry(
                                prompt_name=result['prompt_name'],
                                system_prompt=result['system_prompt'],
                                query=query_text,
                                response=result['response'],
                                mode="Combination_Individual",
                                remark="Rating updated",
                                status=result['status'],
                                status_code=result.get('status_code', 'N/A'),
                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                combination_temperature=st.session_state.combination_results.get('temperature'),
                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                edited=True,
                                rating=new_rating
                            )
                            st.session_state.combination_results['individual_results'][j]['unique_id'] = saved_unique_id
                            st.session_state.combination_results['individual_results'][j]['rating'] = new_rating
                            st.session_state.combination_results['individual_results'][j]['edited'] = True
                            
                            new_df_row = pd.DataFrame([{
                                'unique_id': saved_unique_id,
                                'prompt_name': result['prompt_name'],
                                'system_prompt': result['system_prompt'],
                                'query': query_text,
                                'response': result['response'],
                                'status': result['status'],
                                'status_code': str(result.get('status_code', 'N/A')),
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'edited': True,
                                'remark': 'Rating updated',
                                'rating': new_rating
                            }])
                            st.session_state.test_results = pd.concat([st.session_state.test_results, new_df_row], ignore_index=True)
                            st.rerun()

                        if edited_individual_response != result['response']:
                            col_save, col_reverse = st.columns(2)
                            with col_save:
                                if st.button(f"💾 Save Response", key=f"save_individual_{j}"):
                                    st.session_state.combination_results['individual_results'][j]['response'] = edited_individual_response
                                    st.session_state.combination_results['individual_results'][j]['edited'] = True
                                    saved_unique_id = save_export_entry(
                                        prompt_name=result['prompt_name'],
                                        system_prompt=result['system_prompt'],
                                        query=query_text,
                                        response=edited_individual_response,
                                        mode="Combination_Individual",
                                        remark="Edited and saved",
                                        status=result['status'],
                                        status_code=result.get('status_code', 'N/A'),
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=st.session_state.combination_results.get('temperature'),
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        edited=True,
                                        rating=st.session_state.response_ratings.get(unique_id, 0)
                                    )
                                    st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.get(unique_id, 0)
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'response'] = edited_individual_response
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'edited'] = True
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'remark'] = 'Edited and saved'
                                    st.success("Response updated!")
                                    st.rerun()
                            with col_reverse:
                                if st.button(f"🔄 Reverse Prompt", key=f"reverse_individual_{j}"):
                                    with st.spinner("Generating updated prompt..."):
                                        genai.configure(api_key=gemini_api_key)
                                        suggestion = suggest_func(edited_individual_response, query_text)
                                        source_idx = st.session_state.combination_results['selected_indices'][j]
                                        st.session_state.prompts[source_idx] = suggestion
                                        st.session_state.combination_results['individual_prompts'][j] = suggestion
                                        st.session_state.combination_results['individual_results'][j]['system_prompt'] = suggestion
                                        st.session_state.combination_results['individual_results'][j]['edited'] = True
                                        st.session_state.combination_results['individual_results'][j]['remark'] = 'Reverse prompt generated'
                                        saved_unique_id = save_export_entry(
                                            prompt_name=result['prompt_name'],
                                            system_prompt=suggestion,
                                            query=query_text,
                                            response=edited_individual_response,
                                            mode="Combination_Individual",
                                            remark="Reverse prompt generated",
                                            status=result['status'],
                                            status_code=result.get('status_code', 'N/A'),
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=st.session_state.combination_results.get('temperature'),
                                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                                            edited=True,
                                            rating=st.session_state.response_ratings.get(unique_id, 0)
                                        )
                                        st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.get(unique_id, 0)
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'system_prompt'] = suggestion
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'edited'] = True
                                        st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'remark'] = 'Reverse prompt generated'
                                        st.success("Prompt updated based on edited response!")
                                        st.rerun()

                        st.write("**Details:**")
                        st.write(
                            f"Status: {result['status']} | "
                            f"Time: {result['timestamp']} | "
                            f"Rating: {st.session_state.response_ratings.get(unique_id, result.get('rating', 0))}/10"
                        )
                        
                        if st.button(f"🔮 Suggest Prompt", key=f"suggest_individual_{j}"):
                            with st.spinner("Generating prompt suggestion..."):
                                genai.configure(api_key=gemini_api_key)
                                suggestion = suggest_func(edited_individual_response, query_text)
                                st.session_state[f"suggested_prompt_individual_{j}"] = suggestion
                                st.session_state[f"suggested_prompt_name_individual_{j}"] = f"Suggested Prompt {len(st.session_state.prompts) + 1}"
                                st.write("**Suggested System Prompt:**")
                                st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_individual_{j}", disabled=True)
                        
                        if st.session_state.get(f"suggested_prompt_individual_{j}"):
                            col_save, col_save_run, col_edit = st.columns(3)
                            with col_save:
                                prompt_name = st.text_input("Prompt Name:", value=st.session_state[f"suggested_prompt_name_individual_{j}"], key=f"suggest_individual_name_{j}")
                                if st.button("💾 Save as Prompt", key=f"save_suggest_individual_{j}"):
                                    if prompt_name.strip():
                                        st.session_state.prompts.append(st.session_state[f"suggested_prompt_individual_{j}"])
                                        st.session_state.prompt_names.append(prompt_name.strip())
                                        unique_id = save_export_entry(
                                            prompt_name=prompt_name.strip(),
                                            system_prompt=st.session_state[f"suggested_prompt_individual_{j}"],
                                            query=query_text,
                                            response='Prompt saved but not executed',
                                            mode='Combination_Individual',
                                            remark='Save only',
                                            status='Not Executed',
                                            status_code='N/A',
                                            combination_strategy=st.session_state.combination_results.get('strategy'),
                                            combination_temperature=st.session_state.combination_results.get('temperature'),
                                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                                            rating=0
                                        )
                                        st.session_state.response_ratings[unique_id] = 0
                                        new_result = pd.DataFrame([{
                                            'unique_id': unique_id,
                                            'prompt_name': prompt_name.strip(),
                                            'system_prompt': st.session_state[f"suggested_prompt_individual_{j}"],
                                            'query': query_text,
                                            'response': 'Prompt saved but not executed',
                                            'status': 'Not Executed',
                                            'status_code': 'N/A',
                                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            'edited': False,
                                            'remark': 'Save only',
                                            'rating': 0
                                        }])
                                        st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                        st.session_state[f"suggested_prompt_individual_{j}"] = None
                                        st.session_state[f"suggested_prompt_name_individual_{j}"] = None
                                        st.success(f"Saved as new prompt: {prompt_name.strip()}")
                                        st.rerun()
                                    else:
                                        st.error("Please provide a prompt name")
                            with col_save_run:
                                run_prompt_name = st.text_input("Prompt Name:", value=st.session_state[f"suggested_prompt_name_individual_{j}"], key=f"suggest_individual_run_name_{j}")
                                if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_individual_{j}"):
                                    if run_prompt_name.strip():
                                        st.session_state.prompts.append(st.session_state[f"suggested_prompt_individual_{j}"])
                                        st.session_state.prompt_names.append(run_prompt_name.strip())
                                        with st.spinner("Running new prompt..."):
                                            result = call_api_func(st.session_state[f"suggested_prompt_individual_{j}"], query_text, body_template, headers, response_path)
                                            unique_id = save_export_entry(
                                                prompt_name=run_prompt_name.strip(),
                                                system_prompt=st.session_state[f"suggested_prompt_individual_{j}"],
                                                query=query_text,
                                                response=result['response'] if 'response' in result else None,
                                                mode='Combination_Individual',
                                                remark='Saved and ran',
                                                status=result['status'],
                                                status_code=result.get('status_code', 'N/A'),
                                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                                combination_temperature=st.session_state.combination_results.get('temperature'),
                                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                                rating=0
                                            )
                                            st.session_state.response_ratings[unique_id] = 0
                                            new_result = pd.DataFrame([{
                                                'unique_id': unique_id,
                                                'prompt_name': run_prompt_name.strip(),
                                                'system_prompt': st.session_state[f"suggested_prompt_individual_{j}"],
                                                'query': query_text,
                                                'response': result['response'] if 'response' in result else None,
                                                'status': result['status'],
                                                'status_code': str(result.get('status_code', 'N/A')),
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'edited': False,
                                                'remark': 'Saved and ran',
                                                'rating': 0
                                            }])
                                            st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                        st.session_state[f"suggested_prompt_individual_{j}"] = None
                                        st.session_state[f"suggested_prompt_name_individual_{j}"] = None
                                        st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                        st.rerun()
                                    else:
                                        st.error("Please provide a prompt name")
                            with col_edit:
                                if st.button("✏️ Edit", key=f"edit_suggest_individual_{j}"):
                                    st.session_state[f"edit_suggest_individual_{j}_active"] = True
                                
                                if st.session_state.get(f"edit_suggest_individual_{j}_active", False):
                                    edited_suggestion = st.text_area("Edit Suggested Prompt:", value=st.session_state[f"suggested_prompt_individual_{j}"], height=100, key=f"edit_suggested_individual_{j}")
                                    edit_prompt_name = st.text_input("Prompt Name for Edited Prompt:", value=st.session_state[f"suggested_prompt_name_individual_{j}"], key=f"edit_suggest_individual_name_{j}")
                                    if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_individual_{j}"):
                                        if edit_prompt_name.strip():
                                            st.session_state.prompts.append(edited_suggestion)
                                            st.session_state.prompt_names.append(edit_prompt_name.strip())
                                            unique_id = save_export_entry(
                                                prompt_name=edit_prompt_name.strip(),
                                                system_prompt=edited_suggestion,
                                                query=query_text,
                                                response='Prompt saved but not executed',
                                                mode='Combination_Individual',
                                                remark='Save only',
                                                status='Not Executed',
                                                status_code='N/A',
                                                combination_strategy=st.session_state.combination_results.get('strategy'),
                                                combination_temperature=st.session_state.combination_results.get('temperature'),
                                                slider_weights=st.session_state.combination_results.get('slider_weights'),
                                                rating=0
                                            )
                                            st.session_state.response_ratings[unique_id] = 0
                                            new_result = pd.DataFrame([{
                                                'unique_id': unique_id,
                                                'prompt_name': edit_prompt_name.strip(),
                                                'system_prompt': edited_suggestion,
                                                'query': query_text,
                                                'response': 'Prompt saved but not executed',
                                                'status': 'Not Executed',
                                                'status_code': 'N/A',
                                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'edited': False,
                                                'remark': 'Save only',
                                                'rating': 0
                                            }])
                                            st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                            st.session_state[f"edit_suggest_individual_{j}_active"] = False
                                            st.session_state[f"suggested_prompt_individual_{j}"] = None
                                            st.session_state[f"suggested_prompt_name_individual_{j}"] = None
                                            st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                            st.rerun()
                                        else:
                                            st.error("Please provide a prompt name")
            
            with col2:
                st.subheader("🤝 Combined Prompt Result")
                combined_result = st.session_state.combination_results['combined_result']
                
                if combined_result and combined_result.get('response'):
                    status_color = "🟢" if combined_result['status'] == 'Success' else "🔴"
                    
                    st.markdown(f"**Status:** {status_color} {combined_result['status']}")
                    
                    edited_combined_response = st.text_area(
                        "Combined Response (editable):", 
                        value=combined_result['response'], 
                        height=300, 
                        key="edit_combined_response"
                    )

                    unique_id = combined_result['unique_id']
                    rating_key = f"rating_{unique_id}"
                    current_rating = st.session_state.response_ratings.get(unique_id, int(combined_result.get('rating', 0) or 0))
                    new_rating = st.slider(
                        "Rate this response (0-10):",
                        min_value=0,
                        max_value=10,
                        value=int(current_rating),
                        key=rating_key
                    )

                    if new_rating != current_rating:
                        st.session_state.response_ratings[unique_id] = new_rating
                        
                        saved_unique_id = save_export_entry(
                            prompt_name="AI_Combined",
                            system_prompt=st.session_state.combination_results['combined_prompt'],
                            query=query_text,
                            response=combined_result['response'],
                            mode="Combination_Combined",
                            remark="Rating updated",
                            status=combined_result['status'],
                            status_code=combined_result.get('status_code', 'N/A'),
                            combination_strategy=st.session_state.combination_results.get('strategy'),
                            combination_temperature=st.session_state.combination_results.get('temperature'),
                            slider_weights=st.session_state.combination_results.get('slider_weights'),
                            edited=True,
                            rating=new_rating
                        )
                        st.session_state.combination_results['combined_result']['unique_id'] = saved_unique_id
                        st.session_state.combination_results['combined_result']['rating'] = new_rating
                        st.session_state.combination_results['combined_result']['edited'] = True
                        
                        new_df_row = pd.DataFrame([{
                            'unique_id': saved_unique_id,
                            'prompt_name': "AI_Combined",
                            'system_prompt': st.session_state.combination_results['combined_prompt'],
                            'query': query_text,
                            'response': combined_result['response'],
                            'status': combined_result['status'],
                            'status_code': str(combined_result.get('status_code', 'N/A')),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'edited': True,
                            'remark': 'Rating updated',
                            'rating': new_rating
                        }])
                        st.session_state.test_results = pd.concat([st.session_state.test_results, new_df_row], ignore_index=True)
                        st.rerun()

                    if edited_combined_response != combined_result['response']:
                        col_save, col_reverse = st.columns(2)
                        with col_save:
                            if st.button("💾 Save Combined Response"):
                                st.session_state.combination_results['combined_result']['response'] = edited_combined_response
                                st.session_state.combination_results['combined_result']['edited'] = True
                                saved_unique_id = save_export_entry(
                                    prompt_name="AI_Combined",
                                    system_prompt=st.session_state.combination_results['combined_prompt'],
                                    query=query_text,
                                    response=edited_combined_response,
                                    mode="Combination_Combined",
                                    remark="Edited and saved",
                                    status=combined_result['status'],
                                    status_code=combined_result.get('status_code', 'N/A'),
                                    combination_strategy=st.session_state.combination_results.get('strategy'),
                                    combination_temperature=st.session_state.combination_results.get('temperature'),
                                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                                    edited=True,
                                    rating=st.session_state.response_ratings.get(unique_id, 0)
                                )
                                st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.get(unique_id, 0)
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'response'] = edited_combined_response
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'edited'] = True
                                st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'remark'] = 'Edited and saved'
                                st.success("Combined response updated!")
                                st.rerun()
                        with col_reverse:
                            if st.button("🔄 Reverse Prompt for Combined"):
                                with st.spinner("Generating updated prompt..."):
                                    genai.configure(api_key=gemini_api_key)
                                    suggestion = suggest_func(edited_combined_response, query_text)
                                    st.session_state.combination_results['combined_prompt'] = suggestion
                                    st.session_state.combination_results['combined_result']['system_prompt'] = suggestion
                                    st.session_state.combination_results['combined_result']['edited'] = True
                                    st.session_state.combination_results['combined_result']['remark'] = 'Reverse prompt generated'
                                    saved_unique_id = save_export_entry(
                                        prompt_name="AI_Combined",
                                        system_prompt=suggestion,
                                        query=query_text,
                                        response=edited_combined_response,
                                        mode="Combination_Combined",
                                        remark="Reverse prompt generated",
                                        status=combined_result['status'],
                                        status_code=combined_result.get('status_code', 'N/A'),
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=st.session_state.combination_results.get('temperature'),
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        edited=True,
                                        rating=st.session_state.response_ratings.get(unique_id, 0)
                                    )
                                    st.session_state.response_ratings[saved_unique_id] = st.session_state.response_ratings.get(unique_id, 0)
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == unique_id, 'unique_id'] = saved_unique_id
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'system_prompt'] = suggestion
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'edited'] = True
                                    st.session_state.test_results.loc[st.session_state.test_results['unique_id'] == saved_unique_id, 'remark'] = 'Reverse prompt generated'
                                    st.success("Combined prompt updated based on edited response!")
                                    st.rerun()

                    st.write("**Details:**")
                    st.write(
                        f"Status: {combined_result['status']} | "
                        f"Time: {combined_result['timestamp']} | "
                        f"Rating: {st.session_state.response_ratings.get(unique_id, combined_result.get('rating', 0))}/10"
                    )
                else:
                    st.info("Combined prompt has not been tested yet.")
                
                if st.button("🔮 Suggest Prompt for Combined Response"):
                    with st.spinner("Generating prompt suggestion..."):
                        genai.configure(api_key=gemini_api_key)
                        suggestion = suggest_func(edited_combined_response, query_text)
                        st.session_state.suggested_prompt = suggestion
                        st.session_state.suggested_prompt_name = f"Suggested Prompt {len(st.session_state.prompts) + 1}"
                        st.write("**Suggested System Prompt:**")
                        st.text_area("Suggested Prompt:", value=suggestion, height=100, key="suggested_combined", disabled=True)
                
                if st.session_state.get('suggested_prompt'):
                    col_save, col_save_run, col_edit = st.columns(3)
                    with col_save:
                        prompt_name = st.text_input("Prompt Name:", value=st.session_state.suggested_prompt_name, key="suggest_combined_name")
                        if st.button("💾 Save as Prompt", key="save_suggest_combined"):
                            if prompt_name.strip():
                                st.session_state.prompts.append(st.session_state.suggested_prompt)
                                st.session_state.prompt_names.append(prompt_name.strip())
                                unique_id = save_export_entry(
                                    prompt_name=prompt_name.strip(),
                                    system_prompt=st.session_state.suggested_prompt,
                                    query=query_text,
                                    response='Prompt saved but not executed',
                                    mode='Combination_Individual',
                                    remark='Save only',
                                    status='Not Executed',
                                    status_code='N/A',
                                    combination_strategy=st.session_state.combination_results.get('strategy'),
                                    combination_temperature=st.session_state.combination_results.get('temperature'),
                                    slider_weights=st.session_state.combination_results.get('slider_weights'),
                                    rating=0
                                )
                                st.session_state.response_ratings[unique_id] = 0
                                new_result = pd.DataFrame([{
                                    'unique_id': unique_id,
                                    'prompt_name': edit_prompt_name.strip(),
                                    'system_prompt': st.session_state.suggested_prompt,
                                    'query': query_text,
                                    'response': 'Prompt saved but not executed',
                                    'status': 'Not Executed',
                                    'status_code': 'N/A',
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'edited': False,
                                    'remark': 'Save only',
                                    'rating': 0
                                }])
                                st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                st.session_state.suggested_prompt = None
                                st.session_state.suggested_prompt_name = None
                                st.success(f"Saved as new prompt: {prompt_name.strip()}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")
                    with col_save_run:
                        run_prompt_name = st.text_input("Prompt Name:", value=st.session_state.suggested_prompt_name, key="suggest_combined_run_name")
                        if st.button("🏃 Save as Prompt and Run", key="save_run_suggest_combined"):
                            if run_prompt_name.strip():
                                st.session_state.prompts.append(st.session_state.suggested_prompt)
                                st.session_state.prompt_names.append(run_prompt_name.strip())
                                with st.spinner("Running new prompt..."):
                                    result = call_api_func(st.session_state.suggested_prompt, query_text, body_template, headers, response_path)
                                    unique_id = save_export_entry(
                                        prompt_name=run_prompt_name.strip(),
                                        system_prompt=st.session_state.suggested_prompt,
                                        query=query_text,
                                        response=result['response'] if 'response' in result else None,
                                        mode='Combination_Individual',
                                        remark='Saved and ran',
                                        status=result['status'],
                                        status_code=result.get('status_code', 'N/A'),
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=st.session_state.combination_results.get('temperature'),
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        rating=0
                                    )
                                    st.session_state.response_ratings[unique_id] = 0
                                    new_result = pd.DataFrame([{
                                        'unique_id': unique_id,
                                        'prompt_name': run_prompt_name.strip(),
                                        'system_prompt': st.session_state.suggested_prompt,
                                        'query': query_text,
                                        'response': result['response'] if 'response' in result else None,
                                        'status': result['status'],
                                        'status_code': str(result.get('status_code', 'N/A')),
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'edited': False,
                                        'remark': 'Saved and ran',
                                        'rating': 0
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                st.session_state.suggested_prompt = None
                                st.session_state.suggested_prompt_name = None
                                st.success(f"Saved and ran new prompt: {run_prompt_name.strip()}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")
                    with col_edit:
                        if st.button("✏️ Edit", key="edit_suggest_combined"):
                            st.session_state.edit_suggest_combined_active = True
                        
                        if st.session_state.get("edit_suggest_combined_active", False):
                            edited_suggestion = st.text_area("Edit Suggested Prompt:", value=st.session_state.suggested_prompt, height=100, key="edit_suggested_combined")
                            edit_prompt_name = st.text_input("Prompt Name for Edited Prompt:", value=st.session_state.suggested_prompt_name, key="edit_suggest_combined_name")
                            if st.button("💾 Save Edited Prompt", key="save_edited_suggest_combined"):
                                if edit_prompt_name.strip():
                                    st.session_state.prompts.append(edited_suggestion)
                                    st.session_state.prompt_names.append(edit_prompt_name.strip())
                                    unique_id = save_export_entry(
                                        prompt_name=edit_prompt_name.strip(),
                                        system_prompt=edited_suggestion,
                                        query=query_text,
                                        response='Prompt saved but not executed',
                                        mode='Combination_Individual',
                                        remark='Save only',
                                        status='Not Executed',
                                        status_code='N/A',
                                        combination_strategy=st.session_state.combination_results.get('strategy'),
                                        combination_temperature=st.session_state.combination_results.get('temperature'),
                                        slider_weights=st.session_state.combination_results.get('slider_weights'),
                                        rating=0
                                    )
                                    st.session_state.response_ratings[unique_id] = 0
                                    new_result = pd.DataFrame([{
                                        'unique_id': unique_id,
                                        'prompt_name': edit_prompt_name.strip(),
                                        'system_prompt': edited_suggestion,
                                        'query': query_text,
                                        'response': 'Prompt saved but not executed',
                                        'status': 'Not Executed',
                                        'status_code': 'N/A',
                                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                        'edited': False,
                                        'remark': 'Save only',
                                        'rating': 0
                                    }])
                                    st.session_state.test_results = pd.concat([st.session_state.test_results, new_result], ignore_index=True)
                                    st.session_state.edit_suggest_combined_active = False
                                    st.session_state.suggested_prompt = None
                                    st.session_state.suggested_prompt_name = None
                                    st.success(f"Saved edited prompt as: {edit_prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")