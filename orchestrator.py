import os
import json
import logging
from typing import List, Dict, TypedDict, Literal

import google.generativeai as genai
from langgraph.graph import StateGraph, END

from tools import WebSearchTool, SEDSearchTool
from sed_agent import SEDAgent

# Get logger instance
logger = logging.getLogger(__name__)

# --- State Definition ---
class MainAgentState(TypedDict):
    """Defines the state for the main orchestrator agent."""
    messages: List[Dict[str, str]]
    tool_choice: Literal["web_search", "sed_search", "none"]
    search_query: str
    search_results: List[Dict[str, str]]
    sed_summary: str

# --- Main Agent Class ---
class OrchestratorAgent:
    """The main agent that plans, routes to tools/agents, and generates responses."""
    
    def __init__(self):
        self.llm = genai.GenerativeModel('gemini-1.5-flash')
        self.web_search_tool = WebSearchTool()
        try:
            SED_API_KEY = os.getenv("SED_API_KEY", "your_sed_api_key_here")
            SED_API_BASE_URL = "https://sed-api.com/api/v1"
            sed_search_tool = SEDSearchTool(api_key=SED_API_KEY, base_url=SED_API_BASE_URL)
            self.sed_agent = SEDAgent(llm=self.llm, sed_tool=sed_search_tool)
        except ValueError as e:
            logger.error(e)
            self.sed_agent = None
        self.graph = self._initialize_graph()
        
    def _initialize_graph(self) -> StateGraph:
        """Initializes the main agent's workflow graph."""
        graph = StateGraph(MainAgentState)
        
        graph.add_node("plan_and_route", self._plan_and_route)
        graph.add_node("perform_web_search", self._perform_web_search)
        graph.add_node("call_sed_agent", self._call_sed_agent)
        graph.add_node("generate_response", self._generate_response)
        
        graph.set_entry_point("plan_and_route")
        
        graph.add_conditional_edges(
            "plan_and_route",
            lambda state: state["tool_choice"],
            {
                "web_search": "perform_web_search",
                "sed_search": "call_sed_agent",
                "none": "generate_response"
            }
        )
        
        graph.add_edge("perform_web_search", "generate_response")
        graph.add_edge("call_sed_agent", "generate_response")
        graph.add_edge("generate_response", END)
        
        return graph.compile()

    def _plan_and_route(self, state: MainAgentState) -> MainAgentState:
        json_llm = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["messages"]])
        latest_question = state["messages"][-1]["content"]

        prompt = f"""You are an expert planner for an AI assistant. You have two tools:
1.  `web_search`: For general knowledge, current events, and public information.
2.  `sed_search`: For internal, proprietary documents (project plans, specs, policies).

Analyze the user's latest question in the context of the conversation and choose the single best tool to answer it.

**Conversation History:**
{conversation_history}

**User's Latest Question:** "{latest_question}"

**Instructions:**
1.  **Analyze Intent:** Determine if the user is asking about a public topic or an internal one.
2.  **Rewrite for Clarity:** Rewrite the question to be a self-contained query.
3.  **Choose Tool:** Select "sed_search" for internal topics, "web_search" for public topics, or "none" for greetings/chitchat.
4.  **Output:** Respond in a JSON object with "tool_choice" (string) and "query" (string).
"""
        if not self.sed_agent:
            prompt += "\n**Note:** The `sed_search` tool is currently unavailable. Do not choose it."
        try:
            response = json_llm.generate_content(prompt)
            plan = json.loads(response.text)
            tool_choice = plan.get("tool_choice", "none")
            if tool_choice == "sed_search" and not self.sed_agent:
                tool_choice = "web_search"
            
            state["tool_choice"] = tool_choice
            state["search_query"] = plan.get("query", "")
            logger.info(f"Planner decision: Tool = {state['tool_choice']}, Query = '{state['search_query']}'")
        except Exception as e:
            logger.error(f"Failed to parse planning response: {e}. Defaulting to no tool.")
            state["tool_choice"] = "none"
            state["search_query"] = ""
        return state

    def _perform_web_search(self, state: MainAgentState) -> MainAgentState:
        query = state["search_query"]
        state["search_results"] = self.web_search_tool.run(query)
        return state

    def _call_sed_agent(self, state: MainAgentState) -> MainAgentState:
        logger.info("Orchestrator: Calling SEDAgent.")
        query = state["search_query"]
        state["sed_summary"] = self.sed_agent.run(query)
        return state

    def _generate_response(self, state: MainAgentState) -> MainAgentState:
        web_results = state.get("search_results")
        sed_summary = state.get("sed_summary")
        conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["messages"]])
        
        prompt_template = f"You are Agent Smith, a helpful AI assistant. Answer the user's last question based on the conversation and any provided search results.\n\n**Conversation History:**\n{conversation}"

        if web_results:
            formatted_results = "\n\n".join([f"Title: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}" for r in web_results])
            prompt_template += f"\n\n**Web Search Results:**\n<web_search_results>\n{formatted_results}\n</web_search_results>\n\nSynthesize an answer from the web results. Cite URLs."
        elif sed_summary:
            prompt_template += f"\n\n**Internal Document Summary (from SED Agent):**\n<sed_summary>\n{sed_summary}\n</sed_summary>\n\nSynthesize an answer based on the internal document summary."
        else:
            prompt_template += "\n\nProvide a conversational response based on the history."
        
        logger.info("Generating final response...")
        response = self.llm.generate_content(prompt_template)
        state["messages"].append({"role": "assistant", "content": response.text})
        return state

    def process_message(self, messages: List[Dict[str, str]]) -> Dict:
        initial_state = MainAgentState(
            messages=messages, tool_choice="none", search_query="",
            search_results=[], sed_summary=""
        )
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.error(f"Error in graph invocation: {e}", exc_info=True)
            return {"messages": messages + [{"role": "assistant", "content": f"I'm sorry, an error occurred: {e}"}]}
