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
    if new_prompt.strip():
        st.session_state.prompts.append(new_prompt)
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompts)}")

        # Add placeholder row to test_results
        st.session_state.test_results = pd.concat([
            st.session_state.test_results,
            pd.DataFrame([{
                "unique_id": str(uuid.uuid4()),
                "prompt_name": st.session_state.prompt_names[-1],
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


def add_prompt_section():
    st.subheader("📜 Prompt Management")

    # --- Add new prompt section first ---
    st.text_area("➕ Add New Prompt", key="new_prompt_input")
    # The on_click callback is used to run the add_new_prompt_callback function
    st.button("Add Prompt", key="add_prompt_btn", on_click=add_new_prompt_callback)

    st.markdown("---")

    # --- Existing prompts section below ---
    ensure_prompt_names()
    for i, prompt in enumerate(st.session_state.prompts):
        with st.expander(f"Prompt {i+1}: {st.session_state.prompt_names[i]}", expanded=True):
            st.text_input(
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
                if st.button("💾 Save", key=f"save_btn_{i}"):
                    st.session_state.prompts[i] = edited_text
            with col2:
                if st.button("🗑️ Remove", key=f"remove_btn_{i}"):
                    st.session_state.prompts.pop(i)
                    st.session_state.prompt_names.pop(i)

                    # Drop corresponding row(s) from DataFrame
                    if not st.session_state.test_results.empty and i < len(st.session_state.test_results):
                        st.session_state.test_results = (
                            st.session_state.test_results.drop(i).reset_index(drop=True)
                        )
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
        st.dataframe(preview_df, use_container_width=True)