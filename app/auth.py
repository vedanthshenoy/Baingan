import streamlit as st

def render_auth_page():
    """Render the authentication page and return True if authenticated, False otherwise."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "Unknown"

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
            # Placeholder for actual login logic (e.g., check credentials)
            if username and password:  # Simplified check
                st.session_state.authenticated = True
                st.session_state.user_name = username
                st.success(f"Logged in as {username}")
                st.rerun()
                return True
            else:
                st.error("Please provide valid credentials")
                return False

    with tabs[1]:  # Sign Up tab
        st.subheader("Sign Up")
        new_username = st.text_input("New Username", key="signup_username")
        new_password = st.text_input("New Password", type="password", key="signup_password")
        if st.button("Sign Up"):
            # Placeholder for actual signup logic (e.g., store credentials)
            if new_username and new_password:  # Simplified check
                st.session_state.authenticated = True
                st.session_state.user_name = new_username
                st.success(f"Signed up as {new_username}")
                st.rerun()
                return True
            else:
                st.error("Please provide valid username and password")
                return False

    with tabs[2]:  # Continue as Guest tab
        st.subheader("Continue as Guest")
        guest_name = st.text_input("Enter Your Name", key="guest_name")
        if st.button("Continue"):
            if guest_name.strip():
                st.session_state.authenticated = True
                st.session_state.user_name = guest_name.strip()
                st.success(f"Continuing as {guest_name.strip()}")
                st.rerun()
                return True
            else:
                st.error("Please enter a valid name")
                return False

    return False