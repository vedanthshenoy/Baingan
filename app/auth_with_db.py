import streamlit as st
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

def render_auth_page():
    """Render the authentication page and return True if authenticated, False otherwise."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "Unknown"
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_type' not in st.session_state:
        st.session_state.user_type = None

    if st.session_state.authenticated:
        return True

    st.title("Welcome to BainGan")

    # Add tabs for Login, Sign Up, and Continue as Guest
    tabs = st.tabs(["Login", "Sign Up", "Continue as Guest"])

    with tabs[0]:  # Login tab
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if not username or not password:
                st.error("Please provide both username and password")
                return False
            
            # Connect to database and authenticate
            db = get_db_connection()
            if db:
                success, user, message = db.authenticate_user(username, password)
                db.disconnect()
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_name = user['username']
                    st.session_state.user_id = user['id']
                    st.session_state.user_type = user['user_type']
                    st.success(f"Logged in as {username}")
                    st.rerun()
                    return True
                else:
                    st.error(message)
                    return False
            else:
                st.error("Database connection failed")
                return False

    with tabs[1]:  # Sign Up tab
        st.subheader("Sign Up")
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up"):
            if not new_username or not new_password:
                st.error("Please provide both username and password")
                return False
            
            if new_password != confirm_password:
                st.error("Passwords do not match")
                return False
            
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
                return False
            
            if len(new_username) < 3:
                st.error("Username must be at least 3 characters long")
                return False
            
            # Connect to database and register user
            db = get_db_connection()
            if db:
                success, message = db.register_user(new_username, new_password)
                
                if success:
                    # Auto-login after successful registration
                    success_auth, user, _ = db.authenticate_user(new_username, new_password)
                    db.disconnect()
                    
                    if success_auth:
                        st.session_state.authenticated = True
                        st.session_state.user_name = user['username']
                        st.session_state.user_id = user['id']
                        st.session_state.user_type = user['user_type']
                        st.success(f"Signed up successfully as {new_username}")
                        st.rerun()
                        return True
                else:
                    db.disconnect()
                    st.error(message)
                    return False
            else:
                st.error("Database connection failed")
                return False

    with tabs[2]:  # Continue as Guest tab
        st.subheader("Continue as Guest")
        st.info("Guest sessions are temporary and data may not be saved permanently")
        guest_name = st.text_input("Enter Your Name", key="guest_name")
        
        if st.button("Continue"):
            if not guest_name.strip():
                st.error("Please enter a valid name")
                return False
            
            if len(guest_name.strip()) < 2:
                st.error("Name must be at least 2 characters long")
                return False
            
            # Connect to database and create guest session
            db = get_db_connection()
            if db:
                success, guest_data = db.create_guest_session(guest_name.strip())
                db.disconnect()
                
                if success:
                    st.session_state.authenticated = True
                    st.session_state.user_name = guest_data['display_name']
                    st.session_state.user_id = guest_data['id']
                    st.session_state.user_type = 'guest'
                    st.session_state.guest_username = guest_data['username']
                    st.success(f"Continuing as {guest_name.strip()}")
                    st.rerun()
                    return True
                else:
                    st.error("Failed to create guest session")
                    return False
            else:
                st.error("Database connection failed")
                return False

    return False


def logout():
    """Logout the current user and clear all session data."""
    # Clear all session state variables
    keys_to_clear = [
        'authenticated', 'user_name', 'user_id', 'user_type', 'guest_username',
        'export_data', 'export_data_loaded', 'guest_session_initialized',
        'test_results', 'chain_results', 'combination_results', 'response_ratings'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    st.rerun()


def get_current_user_info():
    """Get current user information."""
    return {
        'username': st.session_state.get('user_name', 'Unknown'),
        'user_id': st.session_state.get('user_id', None),
        'user_type': st.session_state.get('user_type', None),
        'is_guest': st.session_state.get('user_type', None) == 'guest'
    }