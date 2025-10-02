import streamlit as st
import pandas as pd
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import io


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


def process_single_prompt_row(row_data):
    """Process a single row from Excel into prompt data structure"""
    name, prompt_text = row_data
    
    if pd.isna(prompt_text) or not str(prompt_text).strip():
        return None
    
    # Generate unique ID for this prompt
    pid = str(uuid.uuid4())
    
    # Use provided name or generate default
    prompt_name = str(name).strip() if not pd.isna(name) and str(name).strip() else f"Imported Prompt"
    prompt_text = str(prompt_text).strip()
    
    # Create test result row
    test_row = {
        "unique_id": str(uuid.uuid4()),
        "prompt_id": pid,
        "prompt_name": prompt_name,
        "system_prompt": prompt_text,
        "query": None,
        "response": None,
        "status": None,
        "status_code": None,
        "timestamp": datetime.now().isoformat(),
        "rating": None,
        "remark": None,
        "edited": False,
    }
    
    return {
        'pid': pid,
        'name': prompt_name,
        'text': prompt_text,
        'test_row': test_row
    }


def upload_prompts_from_excel(uploaded_file):
    """Process Excel file and add all prompts in parallel"""
    try:
        # Read Excel file
        df = pd.read_excel(uploaded_file)
        
        # Validate columns - support multiple column name variations
        name_col = None
        prompt_col = None
        
        # Check for name column
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in ['name', 'prompt name', 'prompt_name', 'promptname']:
                name_col = col
                break
        
        # Check for prompt text column
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if col_lower in ['prompt', 'system prompt', 'system_prompt', 'systemprompt', 'text', 'prompt text', 'prompt_text']:
                prompt_col = col
                break
        
        # If exact columns not found, try to infer from first 2 columns
        if not prompt_col:
            if len(df.columns) >= 2:
                name_col = df.columns[0]
                prompt_col = df.columns[1]
                st.info(f"Using columns: '{name_col}' for names and '{prompt_col}' for prompts")
            elif len(df.columns) == 1:
                prompt_col = df.columns[0]
                st.info(f"Using single column: '{prompt_col}' for prompts (auto-generating names)")
            else:
                st.error("Excel file must have at least one column with prompt text")
                return 0
        
        # Prepare data for parallel processing
        if name_col and prompt_col:
            rows_data = [(row[name_col], row[prompt_col]) for _, row in df.iterrows()]
        elif prompt_col:
            rows_data = [(None, row[prompt_col]) for _, row in df.iterrows()]
        else:
            st.error("Could not identify prompt column")
            return 0
        
        # Process all rows in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_single_prompt_row, rows_data))
        
        # Filter out None results (invalid rows)
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            st.warning("No valid prompts found in the Excel file")
            return 0
        
        # Initialize tracking structures if needed
        if 'prompt_status' not in st.session_state:
            st.session_state.prompt_status = []
        if 'recently_added_indices' not in st.session_state:
            st.session_state.recently_added_indices = set()
        if 'recently_updated_indices' not in st.session_state:
            st.session_state.recently_updated_indices = set()
        if 'prompt_ids' not in st.session_state:
            st.session_state.prompt_ids = []
        
        # Ensure prompt_status is aligned with current prompts
        while len(st.session_state.prompt_status) < len(st.session_state.prompts):
            st.session_state.prompt_status.append('tested')  # Default for existing prompts
        
        # Batch add all prompts
        base_index = len(st.session_state.prompts)
        new_statuses = ['added'] * len(valid_results)  # Prepare statuses for new prompts
        
        for i, result in enumerate(valid_results):
            st.session_state.prompts.append(result['text'])
            st.session_state.prompt_names.append(result['name'])
            st.session_state.prompt_ids.append(result['pid'])
            st.session_state.recently_added_indices.add(base_index + i)
        
        # Extend prompt_status with new statuses
        st.session_state.prompt_status.extend(new_statuses)
        
        # Batch add test result rows
        test_rows_df = pd.DataFrame([r['test_row'] for r in valid_results])
        st.session_state.test_results = pd.concat(
            [st.session_state.test_results, test_rows_df], 
            ignore_index=True
        )
        
        # Reset prompts_just_added to avoid interfering with button logic
        st.session_state.prompts_just_added = False
        
        # Success message without immediate rerun to preserve state
        st.success(f"✅ Successfully imported {len(valid_results)} prompts!")
        
        return len(valid_results)
        
    except Exception as e:
        st.error(f"Error processing Excel file: {str(e)}")
        return 0


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

    # No need to explicitly refresh widget keys – pid-based keys are stable.


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

    # --- Excel Upload Section ---
    st.markdown("### 📤 Bulk Import from Excel")
    uploaded_file = st.file_uploader(
        "Upload Excel file with prompts (Make sure the column has a Heading)",
        type=['xlsx', 'xls'],
        help="Excel should have columns: 'Name' and 'Prompt' (or similar). If only one column, it will be used for prompts."
    )
    
    if uploaded_file is not None:
        if st.button("🚀 Import Prompts", type="primary"):
            with st.spinner("Processing prompts in parallel..."):
                count = upload_prompts_from_excel(uploaded_file)
                if count > 0:
                    st.success(f"✅ Successfully imported {count} prompts!")
                    st.rerun()
    
    st.markdown("---")

    # --- Add new prompt section ---
    st.markdown("### ➕ Add Single Prompt")
    
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
    st.markdown("### 📝 Manage Existing Prompts")
    
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

        with st.expander(f"Prompt {display_index+1}: {current_name}", expanded=False):
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