import os
import logging
import pytz
from datetime import datetime

import streamlit as st
import google.generativeai as genai

from orchestrator import OrchestratorAgent

def setup_logging():
    """Configure logging with a single log file per session."""
    if not logging.getLogger().handlers:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        local_tz = pytz.timezone('America/Los_Angeles')
        local_time = datetime.now(local_tz)
        timestamp = local_time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"app_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info("Application logging configured.")

class ChatInterface:
    def __init__(self, agent: OrchestratorAgent):
        self.agent = agent
        self._setup_session_state()
        
    def _setup_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Agent Smith. How can I help?"}]
            
    def _display_messages(self):
        """Display all messages in the chat history."""
        for message in st.session_state.messages:
            avatar_path = "images/neo.jpg" if message["role"] == "user" else "images/agent_smith.jpg"
            if not os.path.exists(avatar_path):
                avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])
            else:
                with st.chat_message(message["role"], avatar=avatar_path):
                    st.write(message["content"])

    def _process_user_input(self, prompt: str):
        """Process user input, run the agent, and update the chat state."""
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Processing..."):
            try:
                final_state = self.agent.process_message(st.session_state.messages)
                st.session_state.messages = final_state["messages"]
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logging.getLogger(__name__).error(f"Error in UI processing loop: {e}", exc_info=True)
        st.rerun()

    def display_chat(self):
        """Main function to display the chat interface."""
        if os.path.exists("images/agent_smith.jpg"):
            st.image("images/agent_smith.jpg", width=150)
        else:
            st.title("🤖 Agent Smith")
        
        chat_container = st.container(height=500, border=True)
        with chat_container:
            self._display_messages()
        
        if prompt := st.chat_input("What's on your mind?"):
            self._process_user_input(prompt)

def main():
    """Main function to configure and run the Streamlit app."""
    # Setup logging and ensure asset directory exists
    setup_logging()
    if not os.path.exists("images"):
        os.makedirs("images")
        logging.getLogger(__name__).info("Created 'images' directory. Please add 'agent_smith.jpg' and 'neo.jpg'.")

    # Configure page
    st.set_page_config(page_title="Agent Smith", layout="centered")

    # Configure Gemini API
    try:
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCQJAurEYDJJQrXfYRnYWswteGFauYWI28")
        if GOOGLE_API_KEY == "your_google_api_key_here":
            st.warning("Google API Key is not set. The agent may not function correctly.")
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Failed to configure Google API: {e}")
        return

    # Initialize and run the UI
    try:
        agent = OrchestratorAgent()
        chat_interface = ChatInterface(agent)
        chat_interface.display_chat()
    except Exception as e:
        st.error(f"Failed to initialize the agent. Please check your API keys and configuration.")
        logging.getLogger(__name__).critical(f"Fatal error on startup: {e}", exc_info=True)

if __name__ == "__main__":
    main()
