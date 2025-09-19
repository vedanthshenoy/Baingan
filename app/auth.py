import streamlit as st

def render_auth_page():
    """Renders the signup/login page and manages user authentication."""
    if "user_name" in st.session_state and st.session_state.user_name:
        return True  # User is already logged in

    # Initialize users dictionary in session state if not present
    if "users" not in st.session_state:
        st.session_state.users = {}  # Dictionary to store username: password

    st.title("🔐 Welcome to BainGan")
    st.markdown("Please sign up or log in to continue.")

    # Tabs for Signup and Login
    tab1, tab2 = st.tabs(["Sign Up", "Login"])

    with tab1:
        st.subheader("Sign Up")
        signup_name = st.text_input("Name", key="signup_name")
        signup_username = st.text_input("Username", key="signup_username")
        signup_password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            if signup_username and signup_password and signup_name:
                if signup_username in st.session_state.users:
                    st.error("Username already exists. Please choose a different username.")
                else:
                    st.session_state.users[signup_username] = signup_password
                    st.session_state.user_name = signup_name
                    st.success(f"Welcome, {signup_name}! You have signed up successfully.")
                    st.rerun()
            else:
                st.error("Please fill in all fields.")

    with tab2:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if login_username in st.session_state.users and st.session_state.users[login_username] == login_password:
                # Retrieve or set name (default to username if no name stored)
                st.session_state.user_name = login_username  # Simplified: using username as name for login
                st.success(f"Welcome back, {st.session_state.user_name}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    return False  # User is not logged in