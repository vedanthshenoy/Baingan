import streamlit as st
import pandas as pd
import io
import uuid
from datetime import datetime
from app.score_predictor_llm import predict_scores_llm
from db_operations import DatabaseManager

import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'baingan_db',
    'user': 'root',
    'password': os.getenv("DB_PASSWORD")  # Update with your MySQL password
}

def get_db_connection():
    """Get database connection."""
    db = DatabaseManager(**DB_CONFIG)
    if db.connect():
        return db
    return None

def check_if_guest_user(user_name=None):
    """Check if the current user is a guest."""
    # First check session state user_type (most reliable)
    if 'user_type' in st.session_state and st.session_state.user_type == 'guest':
        return True
    
    # Fallback: Check if guest_username exists in session
    if 'guest_username' in st.session_state:
        return True
    
    # Last resort: Query database with actual username
    if user_name:
        db = get_db_connection()
        if db:
            # Use guest_username if available, otherwise use user_name
            username_to_check = st.session_state.get('guest_username', user_name)
            user_type = db.check_user_type(username_to_check)
            db.disconnect()
            return user_type == 'guest'
    
    return False

def initialize_guest_session():
    """Initialize or reset session state for guest users to ensure clean slate."""
    st.session_state.export_data = pd.DataFrame(columns=[
        'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt',
        'query', 'response', 'status', 'status_code', 'timestamp',
        'edited', 'step', 'combination_strategy', 'combination_temperature',
        'slider_weights', 'rating', 'remark', 'created_at', 'updated_at'
    ]).astype({'rating': 'Int64'})
    st.session_state.test_results = pd.DataFrame(columns=[
        'user_name', 'unique_id', 'prompt_name', 'system_prompt', 'query', 'response', 
        'status', 'status_code', 'timestamp', 'rating', 'remark', 'edited'
    ])
    st.session_state.chain_results = []
    st.session_state.combination_results = {}
    st.session_state.response_ratings = {}
    # Mark that guest session has been initialized
    st.session_state.guest_session_initialized = True

def save_export_entry(
    prompt_name,
    system_prompt,
    query,
    response,
    mode,
    remark,
    status,
    status_code,
    combination_strategy=None,
    combination_temperature=None,
    slider_weights=None,
    edited=False,
    step=None,
    rating=None,
    user_name="Unknown"
):
    """Save export entry to database and session state."""
    # Initialize session state if needed
    if 'export_data' not in st.session_state:
        st.session_state.export_data = pd.DataFrame(columns=[
            'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'edited', 'step',
            'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark',
            'created_at', 'updated_at'
        ]).astype({'rating': 'Int64'})
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unique_id = f"{mode}_{prompt_name}_{timestamp}_{uuid.uuid4()}"
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Create new entry dictionary
    entry_data = {
        'user_name': user_name,
        'unique_id': unique_id,
        'test_type': mode,
        'prompt_name': prompt_name,
        'system_prompt': system_prompt,
        'query': query,
        'response': response,
        'status': status,
        'status_code': str(status_code),
        'timestamp': timestamp,
        'edited': edited,
        'step': str(step) if step is not None else '',
        'combination_strategy': combination_strategy if combination_strategy is not None else '',
        'combination_temperature': str(combination_temperature) if combination_temperature is not None else '',
        'slider_weights': str(slider_weights) if slider_weights is not None else '',
        'rating': rating,
        'remark': remark,
        'created_at': now,
        'updated_at': now
    }
    
    # Save to database for all users (including guests)
    db = get_db_connection()
    if db:
        success = db.save_export_result(**entry_data)
        db.disconnect()
        
        if not success:
            st.warning("Failed to save to database, but saved to session.")
    else:
        st.warning("Database connection failed, saved to session only.")
    
    # Add to session state
    new_entry = pd.DataFrame([entry_data])
    
    if st.session_state.export_data.empty:
        st.session_state.export_data = new_entry
    else:
        st.session_state.export_data = pd.concat([st.session_state.export_data, new_entry], ignore_index=True).astype({'rating': 'Int64'})

    st.write(f"Added {mode} result: {unique_id}")
    return unique_id

def load_user_results(user_name):
    """Load all results for a specific user from database."""
    db = get_db_connection()
    if db:
        results = db.get_user_export_results(user_name)
        db.disconnect()
        return results
    return pd.DataFrame()

def load_all_results_from_db():
    """Load all export results from database to session state for non-guest users ONLY."""
    if 'user_name' in st.session_state:
        user_name = st.session_state.user_name

        # Check if user is a guest - if so, NEVER load from DB
        if check_if_guest_user(user_name):
            # Ensure guest session is properly initialized with empty data
            if 'guest_session_initialized' not in st.session_state:
                initialize_guest_session()
            return False

        # Non-guest users: load results from DB
        # Initialize empty DataFrame first
        st.session_state.export_data = pd.DataFrame(columns=[
            'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt',
            'query', 'response', 'status', 'status_code', 'timestamp',
            'edited', 'step', 'combination_strategy', 'combination_temperature',
            'slider_weights', 'rating', 'remark', 'created_at', 'updated_at'
        ]).astype({'rating': 'Int64'})
        
        df = load_user_results(user_name)
        if not df.empty:
            st.session_state.export_data = df.astype({'rating': 'Int64'})
            return True
    return False

def update_rating_in_db(unique_id, rating, remark):
    """Update rating and remark for a specific entry in database."""
    db = get_db_connection()
    if db:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        success = db.update_export_rating(unique_id, rating, remark, updated_at=now)
        db.disconnect()
        return success
    return False

def delete_entry_from_db(unique_id):
    """Delete a specific entry from database."""
    db = get_db_connection()
    if db:
        success = db.delete_export_result(unique_id)
        db.disconnect()
        return success
    return False

def get_export_statistics(user_name):
    """Get statistics about user's export results."""
    db = get_db_connection()
    if db:
        stats = db.get_export_statistics(user_name)
        db.disconnect()
        return stats
    return None

def reset_guest_session():
    """Reset session state for guest users."""
    initialize_guest_session()

def render_export_section(query_text):
    """Render the export section with database integration."""
    st.header("📊 Export Results")
    
    # Initialize export_data if needed
    if 'export_data' not in st.session_state:
        st.session_state.export_data = pd.DataFrame(columns=[
            'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response', 
            'status', 'status_code', 'timestamp', 'edited', 'step',
            'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark',
            'created_at', 'updated_at'
        ]).astype({'rating': 'Int64'})
    
    # Check for guest user and ensure clean session
    if check_if_guest_user():
        # Initialize guest session if not already done
        if 'guest_session_initialized' not in st.session_state:
            initialize_guest_session()
        
        st.info("🔒 Guest Mode: You can run tests in this session, but previous session data won't be loaded on re-login. Predict Scores feature only available for Signed up users as of now.")
        
        # Display only current session data for guests
        display_guest_session_data()
        # Render clear results section for guests
        render_clear_results_section(guest_mode=True)
        return
    
    # For non-guest users: Load results from database on first render
    if 'user_name' in st.session_state:
        if 'export_data_loaded' not in st.session_state:
            load_all_results_from_db()
            st.session_state.export_data_loaded = True
    
    # Display statistics for non-guest users only
    if 'user_name' in st.session_state:
        stats = get_export_statistics(st.session_state.user_name)
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Results", stats.get('total_results', 0))
            with col2:
                st.metric("Rated Results", stats.get('rated_results', 0))
            with col3:
                # st.metric("Avg Rating", f"{stats.get('avg_rating', 0):.2f}" if stats.get('avg_rating') else "N/A")
                st.metric("Median Rating", f"{stats.get('median_rating', 0):.2f}" if stats.get('median_rating') else "N/A")
            with col4:
                st.metric("Test Types", stats.get('unique_test_types', 0))
    
    if not st.session_state.export_data.empty:
        st.subheader("📋 DataFrame Preview")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            test_types = ['All'] + list(st.session_state.export_data['test_type'].unique())
            selected_type = st.selectbox("Filter by Test Type", test_types)
        with col2:
            rating_filter = st.selectbox("Filter by Rating", ['All', 'Rated', 'Unrated'])
        
        # Apply filters
        filtered_df = st.session_state.export_data.copy()
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['test_type'] == selected_type]
        if rating_filter == 'Rated':
            filtered_df = filtered_df[filtered_df['rating'].notna()]
        elif rating_filter == 'Unrated':
            filtered_df = filtered_df[filtered_df['rating'].isna()]

        # Drop id and edited from display
        drop_cols = [col for col in ['id', 'edited'] if col in filtered_df.columns]
        filtered_df = filtered_df.drop(columns=drop_cols, errors='ignore')

        # Ensure created_at and updated_at are always visible
        if 'created_at' in filtered_df.columns:
            filtered_df['created_at'] = filtered_df['created_at'].fillna(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        if 'updated_at' in filtered_df.columns:
            filtered_df['updated_at'] = filtered_df['updated_at'].fillna(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        st.dataframe(filtered_df, width='stretch')
        
        # Predict Scores button
        st.subheader("🔮 Predict Scores")
        # if (st.session_state.export_data['rating'].notna()).any():
        rated_count = len(st.session_state.export_data[(st.session_state.export_data['rating'].notna()) & (st.session_state.export_data['edited'])])
        st.info(f"Currently {rated_count} rated prompts available for prediction")
        
        if st.button("🔮 Predict Scores for Unrated Entries"):
            if rated_count < 2:
                st.warning("Need at least two rated prompts (with non-null ratings) to predict scores.")
            else:
                # Ensure test_results has necessary columns
                if 'test_results' in st.session_state and not st.session_state.test_results.empty:
                    required_columns = ['system_prompt', 'response', 'rating', 'edited', 'remark']
                    for col in required_columns:
                        if col not in st.session_state.test_results.columns:
                            st.session_state.test_results[col] = None
                    st.session_state.test_results['rating'] = st.session_state.test_results['rating'].astype('Int64')
                    st.session_state.test_results['edited'] = st.session_state.test_results['edited'].fillna(False).astype(bool)
                    st.session_state.test_results['remark'] = st.session_state.test_results['remark'].fillna('')
                
                # Apply predictions using LLM
                try:
                    with st.spinner("Predicting scores using LLM..."):
                        st.session_state.export_data = predict_scores_llm(st.session_state.export_data)
                        
                        # Update predictions in database (for all users, including guests)
                        db = get_db_connection()
                        if db:
                            for idx, row in st.session_state.export_data.iterrows():
                                if pd.notna(row['rating']) and row['edited'] == False:
                                    db.update_export_rating(
                                        row['unique_id'], 
                                        int(row['rating']), 
                                        row['remark'] if pd.notna(row['remark']) else '',
                                        updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    )
                            db.disconnect()
                        
                        if 'test_results' in st.session_state and not st.session_state.test_results.empty:
                            st.session_state.test_results = predict_scores_llm(st.session_state.test_results)
                        
                        st.success("Predicted ratings for unrated prompts using LLM!")
                except Exception as e:
                    st.error(f"Error predicting scores: {str(e)}")
                st.rerun()
        
        # Download buttons
        st.subheader("📥 Download Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"prompt_results_{st.session_state.get('user_name', 'user')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            filtered_df.to_excel(excel_buffer, index=False, sheet_name="Results", engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"prompt_results_{st.session_state.get('user_name', 'user')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            # Download filtered results
            csv_filtered = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered CSV",
                data=csv_filtered,
                file_name=f"filtered_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No results to export yet. Run some tests first!")
    
    # Render clear results section for non-guests
    render_clear_results_section(guest_mode=False)

def display_guest_session_data():
    """Display current session data for guest users (no DB loading)."""
    if not st.session_state.export_data.empty:
        st.subheader("📋 Current Session Data")
        
        # Simplified filters for guests
        col1, col2 = st.columns(2)
        with col1:
            test_types = ['All'] + list(st.session_state.export_data['test_type'].unique())
            selected_type = st.selectbox("Filter by Test Type", test_types)
        with col2:
            rating_filter = st.selectbox("Filter by Rating", ['All', 'Rated', 'Unrated'])
        
        # Apply filters
        filtered_df = st.session_state.export_data.copy()
        if selected_type != 'All':
            filtered_df = filtered_df[filtered_df['test_type'] == selected_type]
        if rating_filter == 'Rated':
            filtered_df = filtered_df[filtered_df['rating'].notna()]
        elif rating_filter == 'Unrated':
            filtered_df = filtered_df[filtered_df['rating'].isna()]

        # Drop id and edited from display
        drop_cols = [col for col in ['id', 'edited'] if col in filtered_df.columns]
        filtered_df = filtered_df.drop(columns=drop_cols, errors='ignore')

        # Ensure created_at and updated_at are always visible
        if 'created_at' in filtered_df.columns:
            filtered_df['created_at'] = filtered_df['created_at'].fillna(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        if 'updated_at' in filtered_df.columns:
            filtered_df['updated_at'] = filtered_df['updated_at'].fillna(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        
        st.dataframe(filtered_df, width='stretch')
        
        # Download buttons for session data
        st.subheader("📥 Download Current Session")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"guest_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            filtered_df.to_excel(excel_buffer, index=False, sheet_name="Session Results", engine='openpyxl')
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name=f"guest_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.info("No current session results yet. Run some tests!")

def render_clear_results_section(guest_mode=False):
    """Render clear results section, with modifications for guest mode."""
    st.subheader("🗑️ Clear Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Session Results", help="Clear results from current session only"):
            if guest_mode:
                initialize_guest_session()
            else:
                reset_guest_session()  # Uses same logic but different context
            st.success("Session results cleared!")
            st.rerun()
    
    # with col2:
    #     if st.button("🗑️ Clear All Database Results", type="secondary", 
    #                 help="Delete all your results from database permanently"):
    #         if not guest_mode:
    #             db = get_db_connection()
    #             if db:
    #                 success = db.delete_all_user_export_results(st.session_state.user_name)
    #                 db.disconnect()
                    
    #                 if success:
    #                     st.session_state.export_data = pd.DataFrame(columns=[
    #                         'user_name', 'unique_id', 'test_type', 'prompt_name', 'system_prompt', 'query', 'response', 
    #                         'status', 'status_code', 'timestamp', 'edited', 'step',
    #                         'combination_strategy', 'combination_temperature', 'slider_weights', 'rating', 'remark',
    #                         'created_at', 'updated_at'
    #                     ]).astype({'rating': 'Int64'})
    #                     st.success("All database results cleared!")
    #                     st.rerun()
    #                 else:
    #                     st.error("Failed to clear database results")
    #             else:
    #                 st.error("Database connection failed")
    #         else:
    #             st.warning("Guest users: Database results are not loaded, so this action has no effect. Use 'Clear Session Results' instead.")

def sync_session_to_db():
    """Sync all session state results to database."""
    if 'export_data' in st.session_state and not st.session_state.export_data.empty:
        db = get_db_connection()
        if db:
            success_count = 0
            for idx, row in st.session_state.export_data.iterrows():
                if db.save_export_result(**row.to_dict()):
                    success_count += 1
            db.disconnect()
            return success_count
    return 0

def export_to_shared_dataset(unique_ids, dataset_name="shared_dataset"):
    """Export selected results to a shared dataset for team collaboration."""
    if not unique_ids:
        return False
    
    db = get_db_connection()
    if db:
        success = db.create_shared_dataset(dataset_name, unique_ids, st.session_state.get('user_name', 'Unknown'))
        db.disconnect()
        return success
    return False