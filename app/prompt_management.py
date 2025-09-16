import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
import inspect


def ensure_prompt_names():
    """Ensure prompts always have names"""
    while len(st.session_state.prompt_names) < len(st.session_state.prompts):
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompt_names) + 1}")


def remove_prompt(index):
    """Callback function to remove a prompt by index"""
    if index < len(st.session_state.prompts):
        removed_prompt = st.session_state.prompts[index]
        st.session_state.prompts.pop(index)
        st.session_state.prompt_names.pop(index)
        # Drop corresponding row(s) from DataFrame
        if not st.session_state.test_results.empty:
            st.session_state.test_results = (
                st.session_state.test_results[
                    st.session_state.test_results['system_prompt'] != removed_prompt
                ]
            ).reset_index(drop=True)


def add_new_prompt(key_suffix):
    """Add a new prompt into session state"""
    key = f"add_prompt_input_{key_suffix}"
    new_prompt = st.session_state[key]
    if new_prompt.strip():
        st.session_state.prompts.append(new_prompt)
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompts)}")
        st.session_state.test_results = pd.concat([
            st.session_state.test_results,
            pd.DataFrame([{
                "unique_id": str(uuid.uuid4()),
                "prompt_name": st.session_state.prompt_names[-1],
                "system_prompt": new_prompt,
                "query": "",
                "response": "",
                "status": "Pending",
                "status_code": 0,
                "timestamp": datetime.now().isoformat(),
                "rating": 0,
                "remark": "",
                "edited": False,
            }])
        ], ignore_index=True)
        # Reset input
        st.session_state[key] = ""


def add_prompt_section():
    st.subheader("📜 Prompt Management")

    # Generate a key suffix based on the caller location (unique per call site)
    caller_line = inspect.currentframe().f_back.f_lineno
    key_suffix = f"line{caller_line}"

    input_key = f"add_prompt_input_{key_suffix}"
    button_key = f"add_prompt_btn_{key_suffix}"

    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    st.text_area("➕ Add New Prompt", key=input_key)
    st.button("Add Prompt", key=button_key, on_click=add_new_prompt, args=[key_suffix])

    st.markdown("---")

    # --- Existing prompts section ---
    if st.session_state.prompts:
        st.subheader("📝 Edit Saved Prompts")
        ensure_prompt_names()

        for i, prompt in enumerate(st.session_state.prompts):
            with st.expander(f"Prompt {i+1}: {st.session_state.prompt_names[i]}", expanded=True):
                st.text_input(
                    f"Prompt {i+1} Name",
                    value=st.session_state.prompt_names[i],
                    key=f"prompt_name_{i}"
                )

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


# --- Session state initialization ---
if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "prompt_names" not in st.session_state:
    st.session_state.prompt_names = []
if "test_results" not in st.session_state:
    st.session_state.test_results = pd.DataFrame(columns=[
        "unique_id", "prompt_name", "system_prompt", "query", "response",
        "status", "status_code", "timestamp", "rating", "remark", "edited"
    ])

add_prompt_section()
