import streamlit as st
from db_request import *


def main():
    """Main function to render the login page."""
    st.title("Welcome to FDX AI Assistant")

    # Input fields for username and password
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login") or password:
        # Check if the credentials are valid
        if check_login(username,password) :

            if "username" not in st.session_state:
                st.session_state.username = username

            st.switch_page(r'pages/test.py')
            st.success("Login successful!")
            st.write("Welcome, {}".format(username))
            # Add your app functionality here after login is successful
        else:
            st.error("Invalid username or password.")

if __name__ == "__main__":
    main()
