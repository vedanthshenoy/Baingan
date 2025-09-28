import streamlit as st
import pandas as pd
import uuid
from datetime import datetime


def ensure_prompt_names():
    """Ensure prompts always have names"""
    while len(st.session_state.prompt_names) < len(st.session_state.prompts):
        st.session_state.prompt_names.append(f"Prompt {len(st.session_state.prompt_names) + 1}")


def ensure_prompt_ids():
    """Ensure there is a stable prompt_id for every prompt (keeps lists aligned)"""
    if 'prompt_ids' not in st.session_state:
        # Create an id for each existing prompt (keeps compatibility with older sessions)
        st.session_state.prompt_ids = [str(uuid.uuid4()) for _ in st.session_state.prompts]
    else:
        # If new prompts were added without ids, create ids for them
        while len(st.session_state.prompt_ids) < len(st.session_state.prompts):
            st.session_state.prompt_ids.append(str(uuid.uuid4()))
        # If some stray ids exist (shouldn't normally happen), trim to match prompts
        if len(st.session_state.prompt_ids) > len(st.session_state.prompts):
            st.session_state.prompt_ids = st.session_state.prompt_ids[: len(st.session_state.prompts)]


def add_new_prompt_callback():
    """Callback function to add a new prompt and clear the input field"""
    new_prompt = st.session_state.new_prompt_input
    new_prompt_name = st.session_state.new_prompt_name_input

    if new_prompt.strip():
        # Use the user-provided name, or a default name if it's empty
        name_to_add = new_prompt_name.strip() if new_prompt_name.strip() else f"Prompt {len(st.session_state.prompts) + 1}"

        # Create a stable id for this prompt
        pid = str(uuid.uuid4())

        # Append to parallel lists (keeps compatibility with other parts of the app)
        st.session_state.prompts.append(new_prompt)
        st.session_state.prompt_names.append(name_to_add)

        # Ensure prompt_ids exists and append the new id
        if 'prompt_ids' not in st.session_state:
            st.session_state.prompt_ids = []
        st.session_state.prompt_ids.append(pid)

        # Initialize tracking for new prompt (used by individual testing)
        if 'prompt_status' not in st.session_state:
            st.session_state.prompt_status = []
        if 'recently_added_indices' not in st.session_state:
            st.session_state.recently_added_indices = set()

        # Mark new prompt as added
        new_index = len(st.session_state.prompts) - 1
        st.session_state.prompt_status.append('added')
        st.session_state.recently_added_indices.add(new_index)

        # Add placeholder row to test_results with the correct prompt name and prompt_id
        # Keep same schema but include 'prompt_id' for precise mapping
        row = {
            "unique_id": str(uuid.uuid4()),
            "prompt_id": pid,
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
        }
        st.session_state.test_results = pd.concat(
            [st.session_state.test_results, pd.DataFrame([row])], ignore_index=True
        )

        # Clear the text area by setting its session state value to an empty string
        st.session_state.new_prompt_input = ""
        st.session_state.new_prompt_name_input = ""

        # Also clean any stray ephemeral keys that might collide later
        # (there shouldn't be any for a freshly created pid, but safe-guarding)
        for k in [f"prompt_name_input_{pid}", f"prompt_text_input_{pid}"]:
            if k in st.session_state:
                del st.session_state[k]


# Callback helpers (must be module-level so Streamlit can call them)
def save_prompt_callback(pid: str):
    """
    Save callback for the prompt identified by pid.
    Reads the user's current values from the widget keys and writes them
    back into parallel lists and the test_results DataFrame (using prompt_id).
    """
    # Derive widget keys (stable because they include pid)
    name_key = f"prompt_name_input_{pid}"
    text_key = f"prompt_text_input_{pid}"

    # Defensive checks: ensure we still have the lists and the pid exists
    if 'prompt_ids' not in st.session_state or pid not in st.session_state.prompt_ids:
        return

    idx = st.session_state.prompt_ids.index(pid)
    old_name = st.session_state.prompt_names[idx] if idx < len(st.session_state.prompt_names) else None

    # Get new values from the widget states (fall back to current values)
    new_name = st.session_state.get(name_key, st.session_state.prompt_names[idx])
    new_text = st.session_state.get(text_key, st.session_state.prompts[idx])

    # Update parallel lists
    st.session_state.prompt_names[idx] = new_name
    st.session_state.prompts[idx] = new_text

    # Mark prompt as updated for tracking (used by individual testing)
    if 'prompt_status' not in st.session_state:
        st.session_state.prompt_status = ['tested'] * len(st.session_state.prompts)
    if 'recently_updated_indices' not in st.session_state:
        st.session_state.recently_updated_indices = set()

    # Only mark as updated if it's not already marked as 'added'
    if len(st.session_state.prompt_status) > idx:
        if st.session_state.prompt_status[idx] != 'added':
            st.session_state.prompt_status[idx] = 'updated'
            st.session_state.recently_updated_indices.add(idx)

    # Update test_results rows that match this prompt_id if available, else fallback to old_name
    try:
        if not st.session_state.test_results.empty:
            if 'prompt_id' in st.session_state.test_results.columns:
                mask = st.session_state.test_results['prompt_id'] == pid
            else:
                # fallback (older rows may not have prompt_id); match by old_name
                mask = st.session_state.test_results['prompt_name'] == old_name

            if mask.any():
                st.session_state.test_results.loc[mask, 'prompt_name'] = new_name
                st.session_state.test_results.loc[mask, 'system_prompt'] = new_text
    except Exception:
        # avoid breaking UI if DataFrame updates fail
        pass

    # No need to explicitly refresh widget keys — pid-based keys are stable.


def remove_prompt_callback(pid: str):
    """
    Remove callback for the prompt identified by pid.
    Removes it from the parallel lists and removes any matching rows in test_results.
    Also removes ephemeral widget keys from session_state to avoid stale state later.
    """
    # Defensive checks
    if 'prompt_ids' not in st.session_state or pid not in st.session_state.prompt_ids:
        return

    idx = st.session_state.prompt_ids.index(pid)

    # Save the name for DataFrame filtering before popping the list
    name_to_remove = st.session_state.prompt_names[idx] if idx < len(st.session_state.prompt_names) else None

    # Remove DataFrame rows for this prompt (prefer prompt_id if present)
    try:
        if not st.session_state.test_results.empty:
            if 'prompt_id' in st.session_state.test_results.columns:
                mask = st.session_state.test_results['prompt_id'] == pid
            else:
                mask = st.session_state.test_results['prompt_name'] == name_to_remove

            if mask.any():
                st.session_state.test_results = st.session_state.test_results.loc[~mask].reset_index(drop=True)
    except Exception:
        # keep silent on dataframe errors; still proceed with list removal
        pass

    # Remove from parallel lists (prompts, prompt_names, prompt_ids)
    try:
        st.session_state.prompts.pop(idx)
    except Exception:
        pass
    try:
        st.session_state.prompt_names.pop(idx)
    except Exception:
        pass
    try:
        st.session_state.prompt_ids.pop(idx)
    except Exception:
        pass

    # Update tracking indices after removal (used by individual testing)
    if 'prompt_status' in st.session_state and 'recently_added_indices' in st.session_state and 'recently_updated_indices' in st.session_state:
        # Remove the status for this index
        if idx < len(st.session_state.prompt_status):
            st.session_state.prompt_status.pop(idx)
        
        # Update all tracking indices that are greater than the removed index
        st.session_state.recently_added_indices = {i-1 if i > idx else i for i in st.session_state.recently_added_indices if i != idx}
        st.session_state.recently_updated_indices = {i-1 if i > idx else i for i in st.session_state.recently_updated_indices if i != idx}

    # Clean up any widget keys that used the pid (to avoid stale values being reused)
    for k in [f"prompt_name_input_{pid}", f"prompt_text_input_{pid}", f"save_btn_{pid}", f"remove_btn_{pid}"]:
        if k in st.session_state:
            del st.session_state[k]


def add_prompt_section():
    st.subheader("📜 Prompt Management")

    # --- Add new prompt section first ---
    # Initialize inputs if missing
    if 'new_prompt_input' not in st.session_state:
        st.session_state.new_prompt_input = ""
    if 'new_prompt_name_input' not in st.session_state:
        st.session_state.new_prompt_name_input = ""

    st.text_input("🏷 New Prompt Name", key="new_prompt_name_input", placeholder="e.g., Marketing Persona Prompt")
    st.text_area("➕ Add New Prompt", key="new_prompt_input")
    # The on_click callback is used to run the add_new_prompt_callback function
    st.button("Add Prompt", key="add_prompt_btn", on_click=add_new_prompt_callback)

    st.markdown("---")

    # --- Existing prompts section below ---
    # Ensure lists are aligned and stable ids exist
    ensure_prompt_names()
    ensure_prompt_ids()

    # Render each prompt using stable pid-based keys
    # Use a snapshot copy of prompt_ids so iteration isn't affected by in-loop changes
    for display_index, pid in enumerate(list(st.session_state.prompt_ids)):
        # Determine the current index mapped to this pid (parallel lists)
        try:
            idx = st.session_state.prompt_ids.index(pid)
        except ValueError:
            # skip ids that disappeared unexpectedly
            continue

        # Defensive: obtain current values from lists or defaults
        current_name = st.session_state.prompt_names[idx] if idx < len(st.session_state.prompt_names) else f"Prompt {display_index+1}"
        current_prompt_text = st.session_state.prompts[idx] if idx < len(st.session_state.prompts) else ""

        # Stable widget keys use pid
        name_key = f"prompt_name_input_{pid}"
        edit_key = f"prompt_text_input_{pid}"
        save_key = f"save_btn_{pid}"
        remove_key = f"remove_btn_{pid}"

        with st.expander(f"Prompt {display_index+1}: {current_name}", expanded=True):
            # Text & area widgets bound to pid-based keys
            st.text_input(f"Prompt {display_index+1} Name", value=current_name, key=name_key)
            st.text_area(f"Edit Prompt {display_index+1}", value=current_prompt_text, key=edit_key)

            col1, col2 = st.columns([1, 1])
            with col1:
                # Save uses a direct callback with the stable pid
                st.button("💾 Save", key=save_key, on_click=save_prompt_callback, args=(pid,))
            with col2:
                # Remove uses a direct callback with the stable pid
                st.button("🗑️ Remove", key=remove_key, on_click=remove_prompt_callback, args=(pid,))

    st.markdown("---")

    # --- Live preview of saved prompts ---
    if st.session_state.prompts:
        st.subheader("📊 Current Prompts")
        preview_df = pd.DataFrame({
            "Name": st.session_state.prompt_names,
            "Prompt": st.session_state.prompts
        })
        st.dataframe(preview_df, width='stretch')