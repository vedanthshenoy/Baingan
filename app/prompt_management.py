import streamlit as st
import uuid

def ensure_prompt_names():
    while len(st.session_state.prompt_names) < len(st.session_state.prompts):
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompt_names) + 1}")
    while len(st.session_state.prompt_names) > len(st.session_state.prompts):
        st.session_state.prompt_names.pop()

def add_prompt_section():
    st.subheader("✏️ Prompt Management")
    
    if 'prompt_input_key_suffix' not in st.session_state:
        st.session_state.prompt_input_key_suffix = str(uuid.uuid4())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_prompt = st.text_area("Enter your system prompt:", height=100, key=f"new_prompt_input_{st.session_state.prompt_input_key_suffix}")
    with col2:
        st.write("")  # spacing
        new_prompt_name = st.text_input("Prompt Name:", placeholder=f"Prompt {len(st.session_state.prompts) + 1}", key=f"new_prompt_name_{st.session_state.prompt_input_key_suffix}")
    
    col_add, col_clear = st.columns(2)
    with col_add:
        if st.button("➕ Add System Prompt", type="primary", key="add_prompt"):
            if new_prompt.strip():
                st.session_state.prompts.append(new_prompt.strip())
                prompt_name = new_prompt_name.strip() if new_prompt_name.strip() else f"Prompt {len(st.session_state.prompts)}"
                st.session_state.prompt_names.append(prompt_name)
                st.session_state.prompt_input_key_suffix = str(uuid.uuid4())
                st.success(f"Added: {prompt_name}")
                st.rerun()
            else:
                st.error("Please enter a prompt")
    
    with col_clear:
        if st.button("🗑️ Clear All Prompts", key="clear_prompts"):
            st.session_state.prompts = []
            st.session_state.prompt_names = []
            st.session_state.test_results = []
            st.session_state.chain_results = []
            st.session_state.combination_results = []
            st.session_state.slider_weights = {}
            st.session_state.last_selected_prompts = []
            st.session_state.response_ratings = {}
            st.session_state.prompt_input_key_suffix = str(uuid.uuid4())
            st.success("Cleared all prompts and results")
            st.rerun()
    
    if st.session_state.prompts:
        ensure_prompt_names()
        st.subheader(f"📋 Current Prompts ({len(st.session_state.prompts)})")
        
        for i in range(len(st.session_state.prompts)):
            with st.expander(f"{st.session_state.prompt_names[i]}: {st.session_state.prompts[i][:50]}..."):
                new_name = st.text_input("Name:", value=st.session_state.prompt_names[i], key=f"edit_name_{i}")
                if new_name != st.session_state.prompt_names[i]:
                    if st.button(f"💾 Update Name", key=f"update_name_{i}"):
                        st.session_state.prompt_names[i] = new_name
                        st.success(f"Updated name to: {new_name}")
                        st.rerun()
                
                edited_prompt = st.text_area("Content:", value=st.session_state.prompts[i], height=100, key=f"edit_prompt_{i}")
                if edited_prompt != st.session_state.prompts[i]:
                    if st.button(f"💾 Update Content", key=f"update_content_{i}"):
                        st.session_state.prompts[i] = edited_prompt
                        st.success("Updated prompt content")
                        st.rerun()
                
                if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                    st.session_state.prompts.pop(i)
                    st.session_state.prompt_names.pop(i)
                    if i < len(st.session_state.test_results):
                        st.session_state.test_results.pop(i)
                    if i < len(st.session_state.chain_results):
                        st.session_state.chain_results.pop(i)
                    if i in st.session_state.slider_weights:
                        del st.session_state.slider_weights[i]
                    for key in list(st.session_state.response_ratings.keys()):
                        if key.startswith(f"test_{i}_") or key.startswith(f"chain_{i}_") or key.startswith(f"combination_individual_{i}_"):
                            del st.session_state.response_ratings[key]
                    st.rerun()