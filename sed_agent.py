import logging
import json
from typing import List, Dict, TypedDict

import google.generativeai as genai
from langgraph.graph import StateGraph, START, END

from tools import SEDSearchTool

# Get logger instance
logger = logging.getLogger(__name__)

class SEDAgentState(TypedDict):
    """Defines the state for the multi-step SED agent workflow."""
    query: str
    search_results: List[Dict[str, str]]
    selected_doc_ids: List[str]
    full_documents_content: str
    final_summary: str

class SEDAgent:
    """
    A specialized agent that performs a multi-step process to answer questions
    using the SED API: search, select, fetch, and synthesize.
    """
    def __init__(self, llm: genai.GenerativeModel, sed_tool: SEDSearchTool):
        self.llm = llm
        self.sed_tool = sed_tool
        self.graph = self._initialize_graph()

    def _initialize_graph(self) -> StateGraph:
        """Defines the sophisticated 4-step workflow for the agent."""
        graph = StateGraph(SEDAgentState)
        graph.add_node("search_for_documents", self._search_step)
        graph.add_node("select_documents", self._select_documents_step)
        graph.add_node("fetch_full_documents", self._fetch_documents_step)
        graph.add_node("synthesize_answer", self._synthesize_answer_step)
        
        graph.add_edge(START, "search_for_documents")
        graph.add_edge("search_for_documents", "select_documents")
        graph.add_edge("select_documents", "fetch_full_documents")
        graph.add_edge("fetch_full_documents", "synthesize_answer")
        graph.add_edge("synthesize_answer", END)
        return graph.compile()

    def _search_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 1: Use the SED tool to search for relevant document summaries."""
        logger.info(f"SEDAgent (Step 1 - Search): Searching for query '{state['query']}'")
        state['search_results'] = self.sed_tool.search(state['query'])
        return state

    def _select_documents_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 2: Use an LLM to select the most relevant documents from the search results."""
        logger.info("SEDAgent (Step 2 - Select): Selecting top documents from search results.")
        if not state['search_results'] or state['search_results'][0]['title'] == "No Results":
            state['selected_doc_ids'] = []
            return state

        json_llm = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
        formatted_results = json.dumps(state['search_results'], indent=2)

        prompt = f"""You are a selection expert. Based on the user's query, select the top 1-3 most relevant documents from the provided search results to answer the query.

**User's Original Question:** "{state['query']}"

**Search Results (JSON):**
{formatted_results}

**Your Task:**
Return a JSON object containing a single key "doc_ids" with a list of the string IDs of the most relevant documents. For example: {{"doc_ids": ["doc-123", "doc-456"]}}
"""
        response = json_llm.generate_content(prompt)
        try:
            selection = json.loads(response.text)
            state['selected_doc_ids'] = selection.get("doc_ids", [])
            logger.info(f"SEDAgent: Selected document IDs: {state['selected_doc_ids']}")
        except (json.JSONDecodeError, AttributeError):
            logger.warning("SEDAgent: Failed to parse document selection, proceeding with no selections.")
            state['selected_doc_ids'] = []
        
        return state

    def _fetch_documents_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 3: Fetch the full content for the selected document IDs."""
        logger.info(f"SEDAgent (Step 3 - Fetch): Fetching full content for IDs: {state['selected_doc_ids']}")
        doc_contents = []
        for doc_id in state.get('selected_doc_ids', []):
            try:
                full_doc = self.sed_tool.get_document_by_id(doc_id)
                # Assuming the full document object has 'title', 'docId', and 'content' keys
                content_str = f"--- Document Start ---\nTitle: {full_doc.get('title')}\nID: {full_doc.get('docId')}\n\nContent:\n{full_doc.get('content')}\n--- Document End ---\n\n"
                doc_contents.append(content_str)
            except Exception as e:
                logger.error(f"SEDAgent: Failed to fetch document {doc_id}: {e}")
        
        state['full_documents_content'] = "\n".join(doc_contents)
        return state

    def _synthesize_answer_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 4: Synthesize a final answer from the full document content."""
        logger.info("SEDAgent (Step 4 - Synthesize): Creating final summary from full content.")
        if not state['full_documents_content']:
            state['final_summary'] = "I found some initial document results, but was unable to retrieve their full content for a detailed answer."
            if not state.get('selected_doc_ids'):
                 state['final_summary'] = "I searched for relevant documents but did not find any that seemed to match your query."
            return state

        prompt = f"""You are a research analyst. You have been given a user's question and the full text of several relevant internal documents. Your task is to synthesize this information into a single, comprehensive answer.

**User's Original Question:** "{state['query']}"

**Full Text of Retrieved Documents:**
<documents>
{state['full_documents_content']}
</documents>

**Your Task:**
Based *only* on the provided full-text documents, write a clear and detailed answer that directly addresses the user's question. Cite the Document ID for any specific pieces of information you use.
"""
        response = self.llm.generate_content(prompt)
        state['final_summary'] = response.text
        return state

    def run(self, query: str) -> str:
        """Runs the SED agent's complete workflow and returns the final, synthesized summary."""
        initial_state = SEDAgentState(
            query=query, search_results=[], selected_doc_ids=[],
            full_documents_content="", final_summary=""
        )
        final_state = self.graph.invoke(initial_state)
        logger.info("SEDAgent: Run complete.")
        return final_state['final_summary']

