import os
import json
import logging
import time
import pytz
from enum import Enum
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any, Optional, TypedDict, Literal, Annotated

# Streamlit and UI
import streamlit as st
from PIL import Image
import io
import base64

# AI and Graph Components
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# Tools and External Agents
from duckduckgo_search import DDGS
# from sed_agent import run_sed_agent

def setup_logging():
    """Configure logging with a single log file per session."""
    # Only set up logging if it hasn't been configured yet
    if not logging.getLogger().handlers:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a log file with timestamp in local time
        local_tz = pytz.timezone('America/Los_Angeles')  # Change this to your local timezone
        local_time = datetime.now(local_tz)
        timestamp = local_time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"app_{timestamp}.log")
        
        # Configure logging with local time
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create logger
        logger = logging.getLogger(__name__)
        logger.info("Application started")

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Constants
PAGE_ICON = "🤖"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCQJAurEYDJJQrXfYRnYWswteGFauYWI28")
# SED_API_KEY = os.getenv("SED_API_KEY", "PUT API KEY HERE")


# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class AgentState(TypedDict):
    """Type definition for the agent's state."""
    messages: List[Dict[str, str]]
    current_step: str
    search_results: List[Dict[str, str]]
    should_search: bool
    # status: Optional[st.status]  # Add status to state

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class SearchError(AgentError):
    """Exception raised when search fails."""
    pass

class LLMError(AgentError):
    """Exception raised when LLM fails."""
    pass

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on failure."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

class SearchTool:
    """Tool for performing internet searches using DuckDuckGo."""
    
    def __init__(self):
        """Initialize the search tool."""
        self.ddgs = DDGS()
        self.last_search_time = 0
        self.min_search_interval = 2  # Minimum seconds between searches
        self.llm = genai.GenerativeModel('gemini-2.0-flash')
        
    def _transform_query(self, query: str) -> str:
        """Transform a question into a search-friendly query using LLM."""
        try:
            prompt = f"""Transform the following question into an optimal search query for a search engine.
            The query should be concise, focused, and likely to return relevant results.
            
            Guidelines:
            1. Remove question words and unnecessary phrases
            2. Keep key terms and important context
            3. Add any necessary search operators or modifiers
            4. Make it specific enough to find relevant results
            5. Keep it under 10 words if possible
            
            Question: {query}
            
            Search query:"""
            
            response = self.llm.generate_content(prompt)
            transformed_query = response.text.strip()
            
            # Fallback to basic transformation if LLM fails
            if not transformed_query:
                # Basic cleanup
                query = query.lower()
                question_words = ['what', 'who', 'how', 'when', 'where', 'why', 'tell me about', 'can you', 'could you']
                for word in question_words:
                    if query.startswith(word):
                        query = query[len(word):].strip()
                query = query.replace('?', '').strip()
                transformed_query = query
            
            logger.debug(f"Transformed query from '{query}' to '{transformed_query}'")
            return transformed_query
            
        except Exception as e:
            logger.error(f"Error transforming query: {str(e)}")
            # Fallback to original query if transformation fails
            return query
        
    @retry_on_error(max_retries=3)
    def run(self, query: str, num_results: int = 12) -> List[Dict[str, str]]:
        """Perform an internet search with rate limiting."""
        try:
            # Transform the query
            search_query = self._transform_query(query)
            
            # Rate limiting
            current_time = time.time()
            time_since_last_search = current_time - self.last_search_time
            if time_since_last_search < self.min_search_interval:
                wait_time = self.min_search_interval - time_since_last_search
                logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            
            logger.debug(f"Performing search for query: {search_query}")
            results = []
            for r in self.ddgs.text(search_query, max_results=num_results):
                result = {
                    "title": r.get("title", "No title"),
                    "snippet": r.get("body", "No description"),
                    "url": r.get("href", "No URL")
                }
                results.append(result)
                logger.debug(f"Found result: {result['title']}")
            
            self.last_search_time = time.time()
            if not results:
                logger.warning("No results found for query")
                return [{
                    "title": "No Results",
                    "snippet": "No results found for your query.",
                    "url": ""
                }]
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise SearchError(f"Search failed: {str(e)}")

class Agent:
    """Agent that can process messages and perform searches."""
    
    def __init__(self):
        """Initialize the agent with tools and LLM."""
        # Configure the Gemini API
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize the model
        self.llm = genai.GenerativeModel('gemini-2.0-flash')
        self.search_tool = SearchTool()
        self._initialize_graph()
        
    def _initialize_graph(self):
        """Initialize the agent's workflow graph."""
        # Create the graph
        graph = StateGraph(AgentState)
        
        # Add nodes with message handling
        graph.add_node("process_input", self._process_input)
        graph.add_node("check_search_need", self._should_search_node)
        graph.add_node("perform_search", self._perform_search)
        graph.add_node("generate_response", self._generate_response)
        
        # Add edges with message handling
        graph.add_edge("process_input", "check_search_need")
        graph.add_conditional_edges(
            "check_search_need",
            lambda x: "perform_search" if x["should_search"] else "generate_response",
            {
                "perform_search": "perform_search",
                "generate_response": "generate_response"
            }
        )
        graph.add_edge("perform_search", "generate_response")
        graph.add_edge("generate_response", END)
        
        # Set the entry point
        graph.set_entry_point("process_input")
        
        # Compile the graph
        self.graph = graph.compile()

    def _process_input(self, state: AgentState) -> AgentState:
        """Process the user input."""
        logger.debug(f"Processing input: {state}")
        return state

    def _should_search(self, message: str) -> bool:
        """Use the LLM to determine if a search is needed."""
        try:
            prompt = f"""Given the following user message, determine if it requires searching the internet for current information.
            Return only 'yes' or 'no'.

            Message: {message}

            Consider:
            1. Does it ask for factual information?
            2. Does it ask about current events?
            3. Does it ask for how-to information?
            4. Does it ask about recent developments?
            5. Could the answer be found in a search engine?

            Answer (yes/no):"""

            logger.debug(f"Calling LLM with prompt: {prompt}")
            response = self.llm.generate_content(prompt)
            decision = response.text.strip().lower()
            logger.debug(f"Search decision for '{message}': {decision}")
            return decision == "yes"
        except Exception as e:
            logger.error(f"Error determining if search is needed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            raise LLMError(f"Failed to determine if search is needed: {str(e)}")

    def _should_search_node(self, state: AgentState) -> AgentState:
        """Node to determine if a search is needed."""
        try:
            last_message = state["messages"][-1]
            should_search = self._should_search(last_message["content"])
            state["should_search"] = should_search
            return state
        except Exception as e:
            logger.error(f"Error in search decision: {str(e)}")
            state["should_search"] = False
            return state

    def _perform_search(self, state: AgentState) -> AgentState:
        """Node to perform the search."""
        try:
            last_message = state["messages"][-1]
            query = last_message["content"]
            
            # Update status if available
            if state.get("status"):
                state["status"].update(label="🔍 Searching DuckDuckGo...", state="running")
            
            # Perform search
            results = self.search_tool.run(query)
            state["search_results"] = results
            
            # Update status if available
            if state.get("status"):
                state["status"].update(label="✅ Search complete!", state="complete")
            
            return state
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            if state.get("status"):
                state["status"].update(label="❌ Search failed", state="error")
            raise SearchError(f"Search failed: {str(e)}")

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate a response using the LLM."""
        try:
            # Get all messages for context
            messages = state["messages"]
            
            # Create a prompt that includes the conversation history
            conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt = f"""You are a helpful AI assistant. Here is the conversation history:

{conversation}

Please provide a helpful response to the last message, taking into account the full conversation context."""
            
            # Generate response
            response = self.llm.generate_content(prompt)
            
            # Add the response to messages
            state["messages"].append({
                "role": "assistant",
                "content": response.text
            })
            
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise LLMError(f"Failed to generate response: {str(e)}")

    def process_message(self, state: AgentState) -> AgentState:
        """Process a message through the graph workflow."""
        try:
            logger.debug(f"Starting message processing with state: {state}")
            
            # Run the graph with the current state
            final_state = None
            for event in self.graph.stream(state):
                logger.debug(f"Graph event: {event}")
                for node_name, value in event.items():
                    logger.debug(f"Processing node {node_name} with value: {value}")
                    if node_name == "generate_response":
                        final_state = value
                        break
            
            if final_state is None:
                raise AgentError("No response was generated")
                
            logger.debug(f"Final state: {final_state}")
            if final_state and "messages" in final_state:
                st.session_state.messages = final_state["messages"]
            return final_state
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            # Add error message to the conversation
            state["messages"].append({
                "role": "assistant",
                "content": f"I apologize, but I encountered an error: {str(e)}"
            })
            return state

class ChatInterface:
    """Class to handle the Streamlit UI and user interactions."""
    def __init__(self, agent: Agent):
        self.agent = agent
        self._setup_session_state()
        
    def _setup_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_step" not in st.session_state:
            st.session_state.current_step = "process_input"
        if "search_results" not in st.session_state:
            st.session_state.search_results = []
        if "should_search" not in st.session_state:
            st.session_state.should_search = False
        if "searching" not in st.session_state:
            st.session_state.searching = False
        if "status" not in st.session_state:
            st.session_state.status = None
            
    def _is_search_query(self, prompt: str) -> bool:
        """Determine if a prompt is likely a search query."""
        search_indicators = [
            "what is", "who is", "how to", "when", "where", "why",
            "tell me about", "search for", "find"
        ]
        return any(indicator in prompt.lower() for indicator in search_indicators)

    def _create_new_state(self, prompt: str) -> AgentState:
        """Create a new state with the current messages and prompt."""
        return {
            "messages": st.session_state.messages.copy(),
            "current_step": "process_input",
            "search_results": [],
            "should_search": self._is_search_query(prompt)
        }

    def _display_messages(self, container):
        """Display all messages in the given container."""
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user", avatar="images/neo.jpg").write(message["content"])
            else:
                # Use Agent Smith image as the assistant's avatar
                st.chat_message("assistant", avatar="images/agent_smith.jpg").write(message["content"])

    def _process_user_input(self, prompt: str):
        """Process user input and update the chat state."""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Create new state and process message
        try:
            new_state = self._create_new_state(prompt)
            final_state = self.agent.process_message(new_state)
            
            if final_state and "messages" in final_state:
                st.session_state.messages = final_state["messages"]
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
        
        # Force a rerun to update the display
        st.rerun()

    def display_chat(self):
        """Display the chat interface."""
        # Display the Agent Smith image instead of text title
        st.image("images/agent_smith.jpg", width=200)
        
        # Create two columns
        left_col, right_col = st.columns([2, 1])
        
        with left_col:
            # Create a container for chat messages
            chat_box = st.container(height=400, border=True)
            
            # Display chat messages in the container
            with chat_box:
                self._display_messages(chat_box)
            
            # Add text input at the bottom
            if prompt := st.chat_input("What's on your mind?", key="chat_input"):
                self._process_user_input(prompt)
        
        # Right column is left empty for future use
        with right_col:
            pass

def get_image_as_base64(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def main():
    # Convert Agent Smith image to base64 for the page icon
    icon_base64 = get_image_as_base64("images/agent_smith.jpg")
    
    st.set_page_config(
        page_title="Agent Smith",
        page_icon=f"data:image/jpeg;base64,{icon_base64}",
        layout="wide"
    )

    # Initialize and display chat interface
    chat_interface = ChatInterface(Agent())
    chat_interface.display_chat()

if __name__ == "__main__":
    main()
