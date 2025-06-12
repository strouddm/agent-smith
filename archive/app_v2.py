# Agent Smith: A Conversational AI Agent with Planning and Search Capabilities version 2.0

import os
import json
import logging
import time
import pytz
import base64
from enum import Enum
from datetime import datetime
from functools import wraps
from typing import List, Dict, TypedDict, Literal

# Streamlit and UI
import streamlit as st
from PIL import Image

# AI and Graph Components
import google.generativeai as genai
from langgraph.graph import StateGraph, END

# Tools
from duckduckgo_search import DDGS

def setup_logging():
    """Configure logging with a single log file per session."""
    if not logging.getLogger().handlers:
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Use a timezone-aware timestamp for the log file
        local_tz = pytz.timezone('America/New_York') # Changed to a common US East timezone
        local_time = datetime.now(local_tz)
        timestamp = local_time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"app_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Reduce noise from verbose libraries
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.info("Application logging configured.")

setup_logging()
logger = logging.getLogger(__name__)

# Constants
PAGE_ICON = "🤖"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCQJAurEYDJJQrXfYRnYWswteGFauYWI28")
# SED_API_KEY = os.getenv("SED_API_KEY", "PUT API KEY HERE")

# Configure the Gemini client
genai.configure(api_key=GOOGLE_API_KEY)

class AgentState(TypedDict):
    """
    Defines the state of the agent, passed between nodes in the graph.
    """
    messages: List[Dict[str, str]]
    search_query: str  # The self-contained query for the search tool
    search_results: List[Dict[str, str]]
    should_search: bool # The decision from the planner node

class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass

class SearchError(AgentError):
    """Exception raised when search fails."""
    pass

class LLMError(AgentError):
    """Exception raised when an LLM call fails."""
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
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts.")
                        raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

class SearchTool:
    """
    A tool for performing internet searches using DuckDuckGo.
    This tool is now simpler, as the query formulation logic is handled by the agent's planner.
    """
    def __init__(self):
        self.ddgs = DDGS()
        self.last_search_time = 0
        self.min_search_interval = 2  # Seconds

    @retry_on_error(max_retries=3)
    def run(self, query: str, num_results: int = 8) -> List[Dict[str, str]]:
        """Perform an internet search with rate limiting."""
        try:
            # Enforce rate limiting
            current_time = time.time()
            if (current_time - self.last_search_time) < self.min_search_interval:
                time.sleep(self.min_search_interval - (current_time - self.last_search_time))
            
            logger.info(f"Performing search for query: '{query}'")
            results = []
            for r in self.ddgs.text(query, max_results=num_results):
                result = {
                    "title": r.get("title", "No title"),
                    "snippet": r.get("body", "No description"),
                    "url": r.get("href", "No URL")
                }
                results.append(result)
            
            self.last_search_time = time.time()
            if not results:
                logger.warning(f"No results found for query: '{query}'")
                return [{"title": "No Results", "snippet": "No results found.", "url": ""}]
            
            return results
            
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}", exc_info=True)
            raise SearchError(f"Search failed for query '{query}': {e}")

class Agent:
    """The conversational agent with planning and search capabilities."""
    
    def __init__(self):
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        self.search_tool = SearchTool()
        self.graph = self._initialize_graph()
        
    def _initialize_graph(self) -> StateGraph:
        """Initializes the agent's workflow graph."""
        graph = StateGraph(AgentState)
        
        # Add the nodes that represent steps in the agent's logic
        graph.add_node("plan_and_route", self._plan_and_route)
        graph.add_node("perform_search", self._perform_search)
        graph.add_node("generate_response", self._generate_response)
        
        # The entry point is now the planner, which decides the route
        graph.set_entry_point("plan_and_route")
        
        # Define conditional edges based on the planner's output
        graph.add_conditional_edges(
            "plan_and_route",
            lambda state: "perform_search" if state["should_search"] else "generate_response",
            {
                "perform_search": "perform_search",
                "generate_response": "generate_response"
            }
        )
        
        # Define the regular edges connecting the rest of the graph
        graph.add_edge("perform_search", "generate_response")
        graph.add_edge("generate_response", END)
        
        # Compile the graph into a runnable workflow
        return graph.compile()

    def _plan_and_route(self, state: AgentState) -> AgentState:
        """
        Analyzes the conversation to decide if a search is needed and, if so, 
        formulates a context-aware, self-contained search query. This uses a
        "chain of thought" style prompt and a JSON output format for reliability.
        """
        # Use a specific model instance configured to return JSON
        json_llm = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config={"response_mime_type": "application/json"}
        )

        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["messages"]])
        latest_question = state["messages"][-1]["content"]

        prompt = f"""You are an expert planner for an AI assistant. Your task is to analyze a conversation and determine if an internet search is required to answer the user's latest question. If a search is needed, you must formulate a precise, self-contained search query.

**Conversation History:**
{conversation_history}

**Instructions:**
1.  **Analyze Context:** Carefully examine the user's latest question: "{latest_question}". Is it a question that requires up-to-date information, facts, or details not commonly known? Or is it a simple greeting, statement, or a question about the AI's capabilities?
2.  **Resolve Ambiguity:** In the context of the full conversation, rewrite the user's latest question to be a standalone, self-contained question. For example, if the user asks "how old is he?" after talking about "Elon Musk", the rewritten question is "how old is Elon Musk?".
3.  **Decide on Search:** Based on the rewritten, self-contained question, decide if an internet search is necessary. A search is needed for facts, current events, definitions, etc. Do not search for conversational pleasantries.
4.  **Formulate Query:** If a search is needed, create a concise, effective search engine query (under 10 words) for the rewritten question.
5.  **Output:** Provide your response in a valid JSON object with two keys: "search_needed" (a boolean: true or false) and "query" (a string: the search query, or an empty string if no search is needed).

**Example:**
Conversation History:
user: Who is the current president of France?
assistant: The current president of France is Emmanuel Macron.
user: What is his wife's name?

Your JSON Output:
{{
  "search_needed": true,
  "query": "Emmanuel Macron's wife's name"
}}
"""
        try:
            response = json_llm.generate_content(prompt)
            plan = json.loads(response.text)
            
            state["should_search"] = plan.get("search_needed", False)
            state["search_query"] = plan.get("query", "")
            
            logger.info(f"Planner decision: Search needed = {state['should_search']}. Query = '{state['search_query']}'")

        except (json.JSONDecodeError, AttributeError, Exception) as e:
            logger.error(f"Failed to parse planning response: {e}. Defaulting to no search.")
            state["should_search"] = False
            state["search_query"] = ""
            
        return state

    def _perform_search(self, state: AgentState) -> AgentState:
        """Node to perform the search using the planned query."""
        try:
            query = state["search_query"]
            logger.info(f"Executing search for: '{query}'")
            results = self.search_tool.run(query)
            state["search_results"] = results
            return state
        except Exception as e:
            logger.error(f"Error performing search: {e}", exc_info=True)
            raise SearchError(f"Search failed: {e}")

    def _generate_response(self, state: AgentState) -> AgentState:
        """
        Generates the final response to the user, incorporating search results if they were fetched.
        """
        try:
            messages = state["messages"]
            search_results = state.get("search_results")
            conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            # Base prompt for the AI assistant
            prompt_template = """You are a helpful AI assistant named Agent Smith. Your goal is to answer the user's last message based on the provided conversation history and, if available, search results.

**Conversation History:**
{conversation}
"""

            # Conditionally add search results to the prompt
            if search_results and state.get("should_search"):
                formatted_results = "\n\n".join(
                    [f"Title: {res['title']}\nSnippet: {res['snippet']}\nURL: {res['url']}" for res in search_results]
                )
                prompt_template += """
**Relevant Search Results:**
<search_results>
{search_results}
</search_results>

Based on the conversation and the provided search results, synthesize a comprehensive and accurate answer to the user's last question. If you use information from a source, cite the URL in parentheses at the end of the relevant sentence (e.g., "Paris is the capital of France (https://en.wikipedia.org/wiki/France)").
"""
                prompt = prompt_template.format(conversation=conversation, search_results=formatted_results)
            else:
                prompt_template += """
Please provide a helpful and conversational response to the last message, taking into account the full conversation context. Do not use external tools or search results.
"""
                prompt = prompt_template.format(conversation=conversation)

            logger.info("Generating final response...")
            response = self.llm.generate_content(prompt)
            
            # Append the assistant's response to the message history
            state["messages"].append({"role": "assistant", "content": response.text})
            return state
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            raise LLMError(f"Failed to generate final response: {e}")

    def process_message(self, messages: List[Dict[str, str]]) -> Dict:
        """Processes a list of messages through the graph workflow."""
        initial_state: AgentState = {
            "messages": messages,
            "search_query": "",
            "search_results": [],
            "should_search": False
        }
        try:
            logger.debug("Starting message processing...")
            final_state = self.graph.invoke(initial_state)
            logger.debug("Message processing complete.")
            return final_state
        except Exception as e:
            logger.error(f"Error processing message in graph: {e}", exc_info=True)
            return {
                "messages": messages + [{"role": "assistant", "content": f"I'm sorry, I encountered an error: {e}"}]
            }

class ChatInterface:
    """Handles the Streamlit UI and user interactions."""
    def __init__(self, agent: Agent):
        self.agent = agent
        self._setup_session_state()
        
    def _setup_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Add a default message if the history is empty
        if not st.session_state.messages:
            st.session_state.messages.append(
                {"role": "assistant", "content": "My name is Smith. Agent Smith. How can I assist you today?"}
            )
            
    def _display_messages(self):
        """Display all messages in the chat history."""
        for message in st.session_state.messages:
            avatar_path = "images/neo.jpg" if message["role"] == "user" else "images/agent_smith.jpg"
            # Use a placeholder if the image doesn't exist to prevent errors
            if not os.path.exists(avatar_path):
                avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
                with st.chat_message(message["role"], avatar=avatar):
                    st.write(message["content"])
            else:
                with st.chat_message(message["role"], avatar=avatar_path):
                    st.write(message["content"])

    def _process_user_input(self, prompt: str):
        """Process user input, run the agent, and update the chat state."""
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Show a spinner while the agent is thinking
        with st.spinner("Processing..."):
            try:
                # The agent now takes the full message history
                final_state = self.agent.process_message(st.session_state.messages)
                st.session_state.messages = final_state["messages"]
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Error in UI processing loop: {e}", exc_info=True)
                st.session_state.messages.append({"role": "assistant", "content": "I've hit a snag. Please try again."})

        # Rerun to display the new messages
        st.rerun()

    def display_chat(self):
        """Main function to display the chat interface."""
        # It's good practice to check if images exist
        if os.path.exists("images/agent_smith.jpg"):
            st.image("images/agent_smith.jpg", width=150)
        else:
            st.title("🤖 Agent Smith")

        # Container for chat messages
        chat_container = st.container(height=500, border=True)
        with chat_container:
            self._display_messages()
        
        # User input field
        if prompt := st.chat_input("What's on your mind?"):
            self._process_user_input(prompt)

def get_image_as_base64(image_path: str) -> str:
    """Safely convert an image file to a base64 string for the page icon."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        logger.warning(f"Icon image not found at {image_path}. Using default emoji.")
        return "🤖" # Return a default emoji if the file is not found

def main():
    """Main function to run the Streamlit app."""
    # Ensure asset directory exists
    if not os.path.exists("images"):
        os.makedirs("images")
        logger.info("Created 'images' directory. Please add 'agent_smith.jpg' and 'neo.jpg'.")

    icon = get_image_as_base64("images/agent_smith.jpg")
    icon_data = f"data:image/jpeg;base64,{icon}" if icon != "🤖" else "🤖"
    
    st.set_page_config(
        page_title="Agent Smith",
        page_icon=icon_data,
        layout="centered"
    )

    # Initialize and display chat interface
    try:
        agent = Agent()
        chat_interface = ChatInterface(agent)
        chat_interface.display_chat()
    except Exception as e:
        st.error("Failed to initialize the agent. Please check your API key and configuration.")
        logger.critical(f"Fatal error on startup: {e}", exc_info=True)

if __name__ == "__main__":
    main()