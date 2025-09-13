import streamlit as st
from datetime import datetime
from app.prompt_management import ensure_prompt_names
from app.api_utils import call_api, suggest_prompt_from_response

def render_prompt_chaining(api_url, query_text, body_template, headers, response_path, call_api_func, suggest_func):
    st.header("🔗 Prompt Chaining")
    
    if st.session_state.prompts:
        ensure_prompt_names()
        st.write("**Current Chain Order:**")
        for i, (prompt, name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
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
    
    if st.button("⛓️ Execute Chain", type="primary", disabled=not (api_url and st.session_state.prompts and query_text)):
        if not api_url:
            st.error("Please enter an API endpoint URL")
        elif not st.session_state.prompts:
            st.error("Please add at least one system prompt")
        elif not query_text:
            st.error("Please enter a query")
        else:
            ensure_prompt_names()
            st.session_state.chain_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            current_query = query_text
            
            for i, (system_prompt, prompt_name) in enumerate(zip(st.session_state.prompts, st.session_state.prompt_names)):
                status_text.text(f"Executing step {i+1}: {prompt_name}...")
                
                result = call_api_func(system_prompt, current_query, body_template, headers, response_path)
                result.update({
                    'step': i + 1,
                    'prompt_name': prompt_name,
                    'system_prompt': system_prompt,
                    'input_query': current_query,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'edited': False,
                    'remark': 'Saved and ran'
                })
                
                st.session_state.chain_results.append(result)
                
                if result['status'] != 'Success':
                    break
                
                current_query = result['response']
                progress_bar.progress((i + 1) / len(st.session_state.prompts))
            
            status_text.text("Chain execution completed!")
            st.success(f"Executed {len(st.session_state.chain_results)} chain steps!")

    if st.session_state.chain_results:
        st.subheader("🔗 Chain Results")
        
        final_result = st.session_state.chain_results[-1]
        if final_result['status'] == 'Success':
            st.success("✅ Chain completed successfully!")
            st.subheader("🎯 Final Result")
            
            edited_final = st.text_area("Final Output (editable):", value=final_result['response'], height=150, key="edit_final_chain")
            
            rating = st.slider(
                "Rate this response (0-10):",
                min_value=0,
                max_value=10,
                value=st.session_state.response_ratings.get("chain_final", 5),
                key="rating_chain_final"
            )
            st.session_state.response_ratings["chain_final"] = rating
            
            if edited_final != final_result['response']:
                col_save, col_reverse = st.columns(2)
                with col_save:
                    if st.button("💾 Save Final Response"):
                        st.session_state.chain_results[-1]['response'] = edited_final
                        st.session_state.chain_results[-1]['edited'] = True
                        st.success("Final response updated!")
                        st.rerun()
                with col_reverse:
                    if st.button("🔄 Reverse Prompt for Final"):
                        with st.spinner("Generating updated prompt..."):
                            suggestion = suggest_func(edited_final, final_result['input_query'])
                            last_index = len(st.session_state.prompts) - 1
                            st.session_state.prompts[last_index] = suggestion
                            st.session_state.chain_results[-1]['system_prompt'] = suggestion
                            st.session_state.chain_results[-1]['edited'] = True
                            st.session_state.chain_results[-1]['remark'] = 'Saved and ran'
                            st.success("Final prompt updated based on edited response!")
                            st.rerun()
            
            if st.button("🔮 Suggest Prompt for Final Response"):
                with st.spinner("Generating prompt suggestion..."):
                    suggestion = suggest_func(edited_final, final_result['input_query'])
                    st.write("**Suggested System Prompt:**")
                    suggested_prompt = st.text_area("Suggested Prompt:", value=suggestion, height=100, key="suggested_final_chain", disabled=True)
                    
                    col_save, col_save_run, col_edit = st.columns(3)
                    with col_save:
                        prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key="suggest_final_name")
                        if st.button("💾 Save as Prompt", key="save_suggest_final"):
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
                        run_prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key="suggest_final_run_name")
                        if st.button("🏃 Save as Prompt and Run", key="save_run_suggest_final"):
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
                        if st.button("✏️ Edit", key="edit_suggest_final"):
                            st.session_state["edit_suggest_final_active"] = True
                    
                    if st.session_state.get("edit_suggest_final_active", False):
                        edited_suggestion = st.text_area("Edit Suggested Prompt:", value=suggestion, height=100, key="edit_suggested_final")
                        if st.button("💾 Save Edited Prompt", key="save_edited_suggest_final"):
                            prompt_name = st.text_input("Prompt Name for Edited Prompt:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key="edit_suggest_final_name")
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
                                st.session_state["edit_suggest_final_active"] = False
                                st.success(f"Saved edited prompt as: {prompt_name.strip()}")
                                st.rerun()
                            else:
                                st.error("Please provide a prompt name")
            
        else:
            st.error(f"❌ Chain failed at step {final_result['step']}: {final_result['prompt_name']}")
        
        st.subheader("📋 Step-by-Step Results")
        for j, result in enumerate(st.session_state.chain_results):
            status_color = "🟢" if result['status'] == 'Success' else "🔴"
            with st.expander(f"{status_color} Step {result['step']}: {result['prompt_name']} - {result['status']}"):
                st.write("**System Prompt:**")
                st.text(result['system_prompt'])
                st.write("**Input Query:**")
                st.text(result['input_query'])
                st.write("**Response:**")
                
                edited_step_response = st.text_area(
                    "Response (editable):", 
                    value=result['response'], 
                    height=150, 
                    key=f"edit_chain_response_{j}"
                )
                
                rating = st.slider(
                    "Rate this response (0-10):",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.response_ratings.get(f"chain_{j}", 5),
                    key=f"rating_chain_{j}"
                )
                st.session_state.response_ratings[f"chain_{j}"] = rating
                
                if edited_step_response != result['response']:
                    col_save, col_reverse = st.columns(2)
                    with col_save:
                        if st.button(f"💾 Save Response", key=f"save_chain_response_{j}"):
                            st.session_state.chain_results[j]['response'] = edited_step_response
                            st.session_state.chain_results[j]['edited'] = True
                            st.success("Step response updated!")
                            st.rerun()
                    with col_reverse:
                        if st.button(f"🔄 Reverse Prompt", key=f"reverse_chain_{j}"):
                            with st.spinner("Generating updated prompt..."):
                                suggestion = suggest_func(edited_step_response, result['input_query'])
                                st.session_state.prompts[j] = suggestion
                                st.session_state.chain_results[j]['system_prompt'] = suggestion
                                st.session_state.chain_results[j]['edited'] = True
                                st.session_state.chain_results[j]['remark'] = 'Saved and ran'
                                st.success("Prompt updated based on edited response!")
                                st.rerun()
                
                if st.button(f"🔮 Suggest Prompt for This Response", key=f"suggest_chain_{j}"):
                    with st.spinner("Generating prompt suggestion..."):
                        suggestion = suggest_func(edited_step_response, result['input_query'])
                        st.write("**Suggested System Prompt:**")
                        suggested_prompt = st.text_area("Suggested Prompt:", value=suggestion, height=100, key=f"suggested_chain_{j}", disabled=True)
                        
                        col_save, col_save_run, col_edit = st.columns(3)
                        with col_save:
                            prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key=f"suggest_chain_name_{j}")
                            if st.button("💾 Save as Prompt", key=f"save_suggest_chain_{j}"):
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
                            run_prompt_name = st.text_input("Prompt Name:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key=f"suggest_chain_run_name_{j}")
                            if st.button("🏃 Save as Prompt and Run", key=f"save_run_suggest_chain_{j}"):
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
                            if st.button("✏️ Edit", key=f"edit_suggest_chain_{j}"):
                                st.session_state[f"edit_suggest_chain_{j}_active"] = True
                        
                        if st.session_state.get(f"edit_suggest_chain_{j}_active", False):
                            edited_suggestion = st.text_area("Edit Suggested Prompt:", value=suggestion, height=100, key=f"edit_suggested_chain_{j}")
                            if st.button("💾 Save Edited Prompt", key=f"save_edited_suggest_chain_{j}"):
                                prompt_name = st.text_input("Prompt Name for Edited Prompt:", placeholder=f"Suggested Prompt {len(st.session_state.prompts) + 1}", key=f"edit_suggest_chain_name_{j}")
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
                                    st.session_state[f"edit_suggest_chain_{j}_active"] = False
                                    st.success(f"Saved edited prompt as: {prompt_name.strip()}")
                                    st.rerun()
                                else:
                                    st.error("Please provide a prompt name")
                
                st.write("**Details:**")
                st.write(f"Status Code: {result['status_code']} | Time: {result['timestamp']} | Rating: {rating}/10 ({rating*10}%)")