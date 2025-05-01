import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL")
if(not API_URL or API_URL == None or API_URL == ""):
    API_URL = "http://localhost:8000"  

def init_session_state():
    """Initialize session state variables."""
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def login(username: str, password: str) -> bool:
    """Attempt to login and store the access token."""
    try:
        response = requests.post(
            f"{API_URL}/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.access_token = data["access_token"]
            st.session_state.username = username
            return True
        return False
    except Exception as e:
        st.error(f"Error during login: {str(e)}")
        return False

def register(username: str, password: str) -> bool:
    """Attempt to register a new user."""
    try:
        response = requests.post(
            f"{API_URL}/register",
            json={"username": username, "password": password}
        )
        return response.status_code == 201
    except Exception as e:
        st.error(f"Error during registration: {str(e)}")
        return False

def send_chat_message(message: str, use_gpt4: bool = True) -> str:
    """Send a message to the chatbot API."""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.post(
            f"{API_URL}/chat",
            json={"user_input": message, "use_gpt4": use_gpt4},
            headers=headers
        )
        if response.status_code == 200:
            return response.json()["bot_response"]
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def login_page():
    """Display the login page."""
    st.title("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if login(username, password):
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")

def register_page():
    """Display the registration page."""
    st.title("Register")
    
    with st.form("register_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            elif len(username) < 3:
                st.error("Username must be at least 3 characters long")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                if register(username, password):
                    st.success("Registration successful! Please login.")
                    st.rerun()
                else:
                    st.error("Registration failed. Username might already exist.")

def chat_page():
    """Display the chat interface."""
    st.title("Chat with AI")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(prompt)
                if response:
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

def main():
    """Main application entry point."""
    init_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    if st.session_state.access_token is None:
        # Not logged in
        page = st.sidebar.radio("Go to", ["Login", "Register"])
        if page == "Login":
            login_page()
        else:
            register_page()
    else:
        # Logged in
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.access_token = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.rerun()
        
        chat_page()

if __name__ == "__main__":
    main() 