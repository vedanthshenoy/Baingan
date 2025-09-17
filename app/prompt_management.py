import streamlit as st
import pandas as pd
import uuid
from datetime import datetime


def ensure_prompt_names():
    """Ensure prompts always have names"""
    while len(st.session_state.prompt_names) < len(st.session_state.prompts):
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompt_names) + 1}")


def add_new_prompt_callback():
    """Callback function to add a new prompt and clear the input field"""
    new_prompt = st.session_state.new_prompt_input
    new_prompt_name = st.session_state.new_prompt_name_input
    
    if new_prompt.strip():
        # Use the user-provided name, or a default name if it's empty
        name_to_add = new_prompt_name.strip() if new_prompt_name.strip() else f"Prompt {len(st.session_state.prompts) + 1}"
        
        st.session_state.prompts.append(new_prompt)
        st.session_state.prompt_names.append(name_to_add)

        # Add placeholder row to test_results with the correct prompt name
        st.session_state.test_results = pd.concat([
            st.session_state.test_results,
            pd.DataFrame([{
                "unique_id": str(uuid.uuid4()),
                "prompt_name": name_to_add,
                "system_prompt": new_prompt,
                "query": None,
                "response": None,
                "status": None,
                "status_code": None,
                "timestamp": datetime.now().isoformat(),
                "rating": None,
                "remark": None,
                "edited": False,
            }])
        ], ignore_index=True)

        # Clear the text area by setting its session state value to an empty string
        st.session_state.new_prompt_input = ""
        st.session_state.new_prompt_name_input = ""


def add_prompt_section():
    st.subheader("📜 Prompt Management")

    # --- Add new prompt section first ---
    st.text_input("📝 New Prompt Name", key="new_prompt_name_input", placeholder="e.g., Marketing Persona Prompt")
    st.text_area("➕ Add New Prompt", key="new_prompt_input")
    # The on_click callback is used to run the add_new_prompt_callback function
    st.button("Add Prompt", key="add_prompt_btn", on_click=add_new_prompt_callback)

    st.markdown("---")

    # --- Existing prompts section below ---
    ensure_prompt_names()
    for i, prompt in enumerate(st.session_state.prompts):
        with st.expander(f"Prompt {i+1}: {st.session_state.prompt_names[i]}", expanded=True):
            edited_name = st.text_input(
                f"Prompt {i+1} Name",
                value=st.session_state.prompt_names[i],
                key=f"prompt_name_{i}"
            )

            # Inline edit
            edited_text = st.text_area(
                f"Edit Prompt {i+1}",
                value=prompt,
                key=f"prompt_edit_{i}"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                # Use a single button to save both name and content
                if st.button("💾 Save", key=f"save_btn_{i}"):
                    # Update both the name and the content in session state
                    st.session_state.prompt_names[i] = edited_name
                    st.session_state.prompts[i] = edited_text
                    
                    # Also update the corresponding row in the DataFrame
                    # Find the index of the row to update based on the original prompt name
                    try:
                        # Assuming prompt names are unique enough for this purpose
                        row_index = st.session_state.test_results[
                            st.session_state.test_results['prompt_name'] == st.session_state.prompt_names[i]
                        ].index[0]
                        st.session_state.test_results.at[row_index, 'prompt_name'] = edited_name
                        st.session_state.test_results.at[row_index, 'system_prompt'] = edited_text
                    except IndexError:
                        # Handle case where the prompt name might not be in the DataFrame yet
                        pass

            with col2:
                if st.button("🗑️ Remove", key=f"remove_btn_{i}"):
                    # Drop corresponding row(s) from DataFrame
                    if not st.session_state.test_results.empty:
                        # Identify rows to drop based on the prompt name
                        rows_to_drop = st.session_state.test_results[
                            st.session_state.test_results['prompt_name'] == st.session_state.prompt_names[i]
                        ].index
                        if not rows_to_drop.empty:
                            st.session_state.test_results = (
                                st.session_state.test_results.drop(rows_to_drop).reset_index(drop=True)
                            )
                            
                    st.session_state.prompts.pop(i)
                    st.session_state.prompt_names.pop(i)
                    # break loop after remove to avoid index mismatch
                    break

    st.markdown("---")

    # --- Live preview of saved prompts ---
    if st.session_state.prompts:
        st.subheader("📊 Current Prompts")
        preview_df = pd.DataFrame({
            "Name": st.session_state.prompt_names,
            "Prompt": st.session_state.prompts
        })
        st.dataframe(preview_df, width='stretch')