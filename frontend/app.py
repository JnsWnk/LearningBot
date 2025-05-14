import streamlit as st
import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import uuid
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
    if 'quiz_answers' not in st.session_state:
        st.session_state.quiz_answers = {}
    if 'quiz_evaluations' not in st.session_state:
        st.session_state.quiz_evaluations = {}

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

def send_chat_message(message: str, use_gpt4: bool = True) -> dict:
    """Send a message to the chatbot API."""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.post(
            f"{API_URL}/chat",
            json={"user_input": message, "use_gpt4": use_gpt4},
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending message: {str(e)}")
        return None

def send_quiz_answer(answer: str, message_id: str, question: str, topic: str) -> dict:
    """Send a quiz answer to the API."""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.post(
            f"{API_URL}/quiz-answer",
            json={"answer": answer, "message_id": message_id, "question": question, "topic": topic},
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error sending quiz answer: {str(e)}")
        return None

def get_user_knowledge() -> dict:
    """Get the user's knowledge profile."""
    try:
        headers = {"Authorization": f"Bearer {st.session_state.access_token}"}
        response = requests.get(
            f"{API_URL}/users/knowledge",
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return {"knowledge_profile": {}}
    except Exception as e:
        st.error(f"Error getting knowledge profile: {str(e)}")
        return {"knowledge_profile": {}}

def display_knowledge_profile():
    """Display the user's knowledge profile in the sidebar."""
    st.sidebar.markdown("### Your Knowledge Profile")
    
    # Get the latest knowledge profile
    knowledge_data = get_user_knowledge()
    knowledge_profile = knowledge_data.get("knowledge_profile", {})
    
    if not knowledge_profile:
        st.sidebar.info("You haven't learned any topics yet. Start by asking about a topic!")
        return
    
    # Sort topics by level (highest first) and then alphabetically
    sorted_topics = sorted(
        knowledge_profile.items(),
        key=lambda x: (-x[1].get("level", 0), x[0])
    )
    
    for topic, data in sorted_topics:
        level = data.get("level", 0)
        last_updated = data.get("last_updated", "")
        
        # Create a progress bar for the level
        st.sidebar.markdown(f"**{topic.replace('_', ' ').title()}**")
        st.sidebar.progress(level / 5)
        st.sidebar.markdown(f"Level: {level}/5")
        if last_updated:
            st.sidebar.markdown(f"Last updated: {last_updated}")
        st.sidebar.markdown("---")

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
    st.title("Chat with AI")
    
    # Display knowledge profile in sidebar
    display_knowledge_profile()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # If this is a quiz message, show the answer input and any evaluation
            if message.get("is_quiz", False):
                message_id = message.get("id", str(uuid.uuid4()))
                
                # Initialize quiz state if not exists
                if message_id not in st.session_state.quiz_answers:
                    st.session_state.quiz_answers[message_id] = ""
                if message_id not in st.session_state.quiz_evaluations:
                    st.session_state.quiz_evaluations[message_id] = None
                
                # Create a container for the quiz interaction
                quiz_container = st.container()
                with quiz_container:
                    # Show input field if no evaluation exists
                    if not st.session_state.quiz_evaluations[message_id]:
                        answer = st.text_input(
                            "Your answer:",
                            key=f"quiz_answer_{message_id}",
                            value=st.session_state.quiz_answers[message_id]
                        )
                        
                        if st.button("Submit Answer", key=f"submit_{message_id}"):
                            response = send_quiz_answer(
                                answer, 
                                message_id, 
                                message["content"], 
                                message.get("topic", "")
                            )
                            if response:
                                # Store the answer and evaluation
                                st.session_state.quiz_answers[message_id] = answer
                                st.session_state.quiz_evaluations[message_id] = response["evaluation"]
                                # Update knowledge profile display
                                st.rerun()
                    else:
                        # Display stored evaluation
                        evaluation = st.session_state.quiz_evaluations[message_id]
                        st.markdown("---")
                        st.markdown("### Evaluation")
                        st.markdown(f"**Score:** {evaluation['score']}/5")
                        st.markdown("**Feedback:**")
                        st.markdown(evaluation['evaluation'])
                        st.markdown("**Sample Solution:**")
                        st.markdown(evaluation['sample_solution'])
                        
                        # Add a button to retry the quiz
                        if st.button("Try Again", key=f"retry_{message_id}"):
                            # Clear the evaluation and answer
                            st.session_state.quiz_evaluations[message_id] = None
                            st.session_state.quiz_answers[message_id] = ""
                            st.rerun()
    
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
                    message_id = str(uuid.uuid4())
                    st.write(response["bot_response"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["bot_response"],
                        "is_quiz": response.get("is_quiz", False),
                        "id": message_id,
                        "topic": response.get("topic", "")
                    })
                    # Force a rerun to show the quiz input field immediately and update knowledge profile
                    st.rerun()

def get_statistics() -> dict:
    """Get statistics about all users' knowledge profiles."""
    try:
        response = requests.get(f"{API_URL}/statistics")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting statistics: {str(e)}")
        return None

def statistics_page():
    """Display the statistics page."""
    st.title("Learning Statistics")
    
    # Get statistics
    stats = get_statistics()
    if not stats:
        st.info("No statistics available yet.")
        return
    
    # Display total users
    st.metric("Total Users", stats["total_users"])
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Topic Popularity", "Mastery Distribution", "Topic Details"])
    
    with tab1:
        # Topic popularity chart
        topics = []
        percentages = []
        for topic, data in stats["topic_statistics"].items():
            topics.append(topic.replace("_", " ").title())
            percentages.append(data["percentage_of_users"])
        
        df_popularity = pd.DataFrame({
            "Topic": topics,
            "Percentage of Users": percentages
        })
        
        fig_popularity = px.bar(
            df_popularity,
            x="Topic",
            y="Percentage of Users",
            title="Topic Popularity Across Users",
            labels={"Topic": "Topic", "Percentage of Users": "% of Users"},
            color="Percentage of Users",
            color_continuous_scale="Viridis"
        )
        fig_popularity.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_popularity, use_container_width=True)
    
    with tab2:
        # Overall mastery distribution
        levels = list(stats["overall_level_distribution"].keys())
        percentages = list(stats["overall_level_distribution"].values())
        
        fig_distribution = px.pie(
            values=percentages,
            names=[f"Level {level}" for level in levels],
            title="Overall Mastery Level Distribution",
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_distribution, use_container_width=True)
    
    with tab3:
        # Detailed topic statistics
        st.subheader("Detailed Topic Statistics")
        
        # Create a DataFrame for the detailed view
        detailed_data = []
        for topic, data in stats["topic_statistics"].items():
            detailed_data.append({
                "Topic": topic.replace("_", " ").title(),
                "Total Users": data["total_users"],
                "Average Level": round(data["average_level"], 2),
                "% of Users": round(data["percentage_of_users"], 2)
            })
        
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(
            df_detailed.sort_values("Total Users", ascending=False),
            use_container_width=True
        )
        
        # Level distribution heatmap
        topics = []
        levels = []
        percentages = []
        
        for topic, data in stats["topic_statistics"].items():
            for level, percentage in data["level_distribution"].items():
                topics.append(topic.replace("_", " ").title())
                levels.append(f"Level {level}")
                percentages.append(percentage)
        
        df_heatmap = pd.DataFrame({
            "Topic": topics,
            "Level": levels,
            "Percentage": percentages
        })
        
        fig_heatmap = px.density_heatmap(
            df_heatmap,
            x="Level",
            y="Topic",
            z="Percentage",
            title="Topic Mastery Level Distribution",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def main():
    """Main application entry point."""
    init_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    if st.session_state.access_token is None:
        # Not logged in
        page = st.sidebar.radio("Go to", ["Login", "Register", "Statistics"])
        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()
        else:
            statistics_page()
    else:
        # Logged in
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.access_token = None
            st.session_state.username = None
            st.session_state.chat_history = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_evaluations = {}
            st.rerun()
        
        page = st.sidebar.radio("Go to", ["Chat", "Statistics"])
        if page == "Chat":
            chat_page()
        else:
            statistics_page()

if __name__ == "__main__":
    main() 