import streamlit as st
import pandas as pd
import uuid
from datetime import datetime


def ensure_prompt_names():
    """Ensure prompts always have names"""
    while len(st.session_state.prompt_names) < len(st.session_state.prompts):
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompt_names) + 1}")


def remove_prompt(index):
    """Callback function to remove a prompt by index"""
    if index < len(st.session_state.prompts):
        st.session_state.prompts.pop(index)
        st.session_state.prompt_names.pop(index)


def add_new_prompt():
    """Callback function to add a new prompt and clear the input field"""
    new_prompt = st.session_state.new_prompt_input
    if new_prompt.strip():
        st.session_state.prompts.append(new_prompt)
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompts)}")
        # Clear the text area by setting its session state value to an empty string
        st.session_state.new_prompt_input = ""


def add_prompt_section():
    st.subheader("📜 Prompt Management")

    # --- Add new prompt section first ---
    # The key is used to store the text area's value in st.session_state
    st.text_area("➕ Add New Prompt", key="new_prompt_input")
    # The on_click callback is used to run the add_new_prompt function
    st.button("Add Prompt", key="add_prompt_btn", on_click=add_new_prompt)

    st.markdown("---")

    # --- Existing prompts section below ---
    if "prompts" in st.session_state and st.session_state.prompts:
        st.subheader("📝 Edit Saved Prompts")
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
                    st.button(
                        "🗑️ Remove",
                        key=f"remove_btn_{i}",
                        on_click=remove_prompt,
                        args=[i]
                    )

    st.markdown("---")


if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "prompt_names" not in st.session_state:
    st.session_state.prompt_names = []
if "test_results" not in st.session_state:
    st.session_state.test_results = pd.DataFrame(columns=[
        "unique_id", "prompt_name", "system_prompt", "query", "response", "status", "status_code", "timestamp", "rating", "remark", "edited"
    ])

add_prompt_section()