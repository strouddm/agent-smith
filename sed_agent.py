import logging
import json
import sqlite3
import os
from typing import List, Dict, TypedDict

import google.generativeai as genai
from langgraph.graph import StateGraph, START, END

from tools import SEDSearchTool

# Get logger instance
logger = logging.getLogger(__name__)

class SEDAgentState(TypedDict):
    """Defines the state for the multi-step SED agent workflow."""
    query: str
    original_query: str 
    search_results: List[Dict[str, str]]
    doc_evaluations: List[Dict]
    final_summary: str

class SEDAgent:
    """
    A specialized agent that performs a multi-step process to answer questions
    using the SED API with document storage and evaluation.
    """
    def __init__(self, llm: genai.GenerativeModel, sed_tool: SEDSearchTool):
        self.llm = llm
        self.sed_tool = sed_tool
        self.db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sed_documents.db')
        self._initialize_database()
        self.graph = self._initialize_graph()

    def _initialize_database(self):
        """Set up SQLite database for document storage."""
        db_exists = os.path.exists(self.db_path)
        if db_exists:
            logger.info(f"SQLite database already exists at {self.db_path}")
            # Connect and verify tables exist to ensure schema is correct
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents'")
                documents_exists = cursor.fetchone() is not None
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='evaluations'")
                evaluations_exists = cursor.fetchone() is not None
                
                conn.close()
                
                # If tables don't exist, proceed with initialization
                if not (documents_exists and evaluations_exists):
                    logger.info("Database exists but schema is incomplete. Initializing schema...")
                    self._create_database_schema()
                else:
                    logger.info("Database schema verified.")
                    return
            except Exception as e:
                logger.warning(f"Error checking database schema: {e}. Will recreate schema.")
                self._create_database_schema()
        else:
            logger.info(f"Initializing new SQLite database at {self.db_path}")
            self._create_database_schema()

    def _create_database_schema(self):
        """Create database tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create documents table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    query TEXT,
                    title TEXT,
                    content TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create evaluations table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT,
                    query TEXT,
                    relevance_score REAL,
                    insight_score REAL,
                    evaluation_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database schema creation complete")
        except Exception as e:
            logger.error(f"Database schema creation failed: {e}")
            raise

    def _initialize_graph(self) -> StateGraph:
        """Defines the workflow for the agent with keyword optimization."""
        graph = StateGraph(SEDAgentState)
        
        # Add new step for query optimization
        graph.add_node("optimize_query", self._optimize_query_step)
        graph.add_node("search_documents", self._search_documents_step)
        graph.add_node("store_documents", self._store_documents_step)
        graph.add_node("evaluate_documents", self._evaluate_documents_step)
        graph.add_node("synthesize_results", self._synthesize_results_step)
        
        # Define the workflow sequence with the new step
        graph.add_edge(START, "optimize_query")
        graph.add_edge("optimize_query", "search_documents")
        graph.add_edge("search_documents", "store_documents")
        graph.add_edge("store_documents", "evaluate_documents")
        graph.add_edge("evaluate_documents", "synthesize_results")
        graph.add_edge("synthesize_results", END)
        
        return graph.compile()

    def _optimize_query_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 0: Convert natural language query to optimized keyword search."""
        logger.info(f"SEDAgent (Step 0 - Optimize): Converting query '{state['query']}' to keyword format")
        
        optimization_prompt = f"""Convert this natural language question into an optimized keyword search query for an intelligence database.

Question: "{state['query']}"

Guidelines:
1. Focus ONLY on extracting critical entities (people, organizations, locations, specific terms)
2. Use boolean operators (AND, OR) and parentheses for structure
3. Include common variations of names and acronyms with OR
4. For names, try both full name and parts ("john smith" OR "john" OR "smith")
5. For complex terms, try variations and acronyms ("supply chain" OR "supplier")
6. Return ONLY the optimized search query, nothing else

Example 1:
Question: "What's the relationship between John Smith and Acme Corporation in the healthcare sector?"
Result: ("john smith" OR "smith, john") AND ("acme" OR "acme corporation") AND ("healthcare" OR "health care" OR "medical")

Example 2:
Question: "Tell me about cyberattacks from North Korea targeting financial institutions"
Result: ("cyberattack" OR "cyber attack" OR "hack") AND ("north korea" OR "dprk" OR "democratic people's republic of korea") AND ("financial" OR "bank" OR "finance")

Optimized search query:"""
        
        try:
            response = self.llm.generate_content(optimization_prompt)
            optimized_query = response.text.strip()
            
            # Store both original and optimized queries
            state['original_query'] = state['query']
            state['query'] = optimized_query
            
            logger.info(f"SEDAgent: Optimized query to '{optimized_query}'")
        except Exception as e:
            logger.error(f"SEDAgent: Failed to optimize query: {e}")
            # Keep original query if optimization fails
            state['original_query'] = state['query']
            
        return state

    def _search_documents_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 1: Use the SED tool to search for relevant documents."""
        logger.info(f"SEDAgent (Step 1 - Search): Searching for query '{state['query']}'")
        state['search_results'] = self.sed_tool.search(state['query'])
        logger.info(f"SEDAgent: Found {len(state['search_results'])} document results")
        return state

    def _store_documents_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 2: Store the documents in SQLite database."""
        if not state['search_results'] or state['search_results'][0].get('title') == "No Results":
            logger.warning("SEDAgent: No documents to store")
            return state
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for doc in state['search_results']:
                doc_id = doc.get('docId', 'unknown-' + str(hash(json.dumps(doc))))
                title = doc.get('title', 'No Title')
                
                # Convert document to JSON string for storage
                content_json = json.dumps(doc)
                metadata = json.dumps({'source': 'SED API', 'query': state['query']})
                
                # Check if document already exists in database
                cursor.execute("SELECT doc_id FROM documents WHERE doc_id = ?", (doc_id,))
                if cursor.fetchone():
                    # Update existing document
                    cursor.execute(
                        "UPDATE documents SET query = ?, content = ?, metadata = ? WHERE doc_id = ?",
                        (state['query'], content_json, metadata, doc_id)
                    )
                    logger.info(f"SEDAgent: Updated existing document: {doc_id}")
                else:
                    # Insert new document
                    cursor.execute(
                        "INSERT INTO documents (doc_id, query, title, content, metadata) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, state['query'], title, content_json, metadata)
                    )
                    logger.info(f"SEDAgent: Stored new document: {doc_id}")
            
            conn.commit()
            conn.close()
            logger.info(f"SEDAgent (Step 2 - Store): Stored {len(state['search_results'])} documents in database")
        except Exception as e:
            logger.error(f"SEDAgent: Failed to store documents: {e}")
            
        return state

    def _evaluate_documents_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 3: Evaluate each document's relevance and add evaluations to database."""
        if not state['search_results'] or state['search_results'][0].get('title') == "No Results":
            logger.warning("SEDAgent: No documents to evaluate")
            state['doc_evaluations'] = []
            return state
            
        state['doc_evaluations'] = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Use a model optimized for structured evaluation
            eval_llm = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json"})
            
            for doc in state['search_results']:
                doc_id = doc.get('docId', 'unknown-' + str(hash(json.dumps(doc))))
                title = doc.get('title', 'No Title')
                summary = doc.get('summary', 'No Summary')
                
                # Build evaluation prompt
                eval_prompt = f"""You are a document evaluation expert. Evaluate the relevance and insights of this document to the user's query.

**User Query:** {state['query']}

**Document Title:** {title}

**Document Summary:** {summary}

Evaluate on two dimensions:
1. Relevance (0-10): How directly relevant is this document to the query?
2. Insight Value (0-10): How much valuable insight does this document provide?

Return a JSON object with the following keys:
- relevance_score: numerical score 0-10
- insight_score: numerical score 0-10
- evaluation: text explanation of the evaluation (2-3 sentences)
"""

                # Generate evaluation
                eval_response = eval_llm.generate_content(eval_prompt)
                try:
                    evaluation = json.loads(eval_response.text)
                    relevance_score = float(evaluation.get('relevance_score', 0))
                    insight_score = float(evaluation.get('insight_score', 0))
                    evaluation_text = evaluation.get('evaluation', 'No evaluation provided')
                    
                    # Store evaluation in database
                    cursor.execute(
                        "INSERT INTO evaluations (doc_id, query, relevance_score, insight_score, evaluation_text) VALUES (?, ?, ?, ?, ?)",
                        (doc_id, state['query'], relevance_score, insight_score, evaluation_text)
                    )
                    
                    # Add evaluation to state
                    doc_with_eval = doc.copy()
                    doc_with_eval['evaluation'] = {
                        'relevance_score': relevance_score,
                        'insight_score': insight_score,
                        'evaluation_text': evaluation_text
                    }
                    state['doc_evaluations'].append(doc_with_eval)
                    
                    logger.info(f"SEDAgent: Evaluated document {doc_id} - Relevance: {relevance_score}, Insight: {insight_score}")
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.error(f"SEDAgent: Failed to parse evaluation for document {doc_id}: {e}")
                    state['doc_evaluations'].append(doc)
            
            conn.commit()
            conn.close()
            logger.info(f"SEDAgent (Step 3 - Evaluate): Evaluated {len(state['doc_evaluations'])} documents")
        except Exception as e:
            logger.error(f"SEDAgent: Failed during document evaluation: {e}")
            # If evaluation fails, continue with original documents
            state['doc_evaluations'] = state['search_results']
            
        return state

    def _synthesize_results_step(self, state: SEDAgentState) -> SEDAgentState:
        """Step 4: Synthesize a final answer based on document evaluations."""
        logger.info("SEDAgent (Step 4 - Synthesize): Creating final summary from evaluations")
        
        if not state['doc_evaluations']:
            state['final_summary'] = "I searched for relevant documents but did not find any that matched your query."
            return state
        
        # Sort documents by combined score
        sorted_docs = sorted(
            [doc for doc in state['doc_evaluations'] if 'evaluation' in doc],
            key=lambda x: x.get('evaluation', {}).get('relevance_score', 0) + x.get('evaluation', {}).get('insight_score', 0),
            reverse=True
        )
        
        # Create a consolidated report
        doc_summaries = []
        for i, doc in enumerate(sorted_docs[:5]):  # Use top 5 documents
            title = doc.get('title', 'No Title')
            summary = doc.get('summary', 'No Summary')
            relevance = doc.get('evaluation', {}).get('relevance_score', 'N/A')
            insight = doc.get('evaluation', {}).get('insight_score', 'N/A')
            evaluation = doc.get('evaluation', {}).get('evaluation_text', 'No evaluation')
            
            doc_summaries.append(f"""
Document {i+1}: {title}
Relevance Score: {relevance}/10 | Insight Score: {insight}/10
Summary: {summary}
Evaluation: {evaluation}
---
""")
        
        # Create synthesis prompt - now includes both queries
        synthesis_prompt = f"""You are an intelligence analyst synthesizing information from evaluated documents.

**User's Original Question:** "{state['original_query']}"

**Query Used for Search:** "{state['query']}"

**Evaluated Documents:**
{' '.join(doc_summaries)}

**Your Task:**
Write a comprehensive synthesis that:
1. Directly answers the user's question
2. Integrates insights from the most relevant documents
3. Highlights confidence levels based on document evaluations
4. Notes any significant gaps or contradictions

Focus on creating actionable intelligence from these evaluated sources.
"""
        
        # Generate synthesis
        try:
            response = self.llm.generate_content(synthesis_prompt)
            state['final_summary'] = response.text
        except Exception as e:
            logger.error(f"SEDAgent: Failed to generate synthesis: {e}")
            # Fallback summary if synthesis fails
            state['final_summary'] = f"I found {len(sorted_docs)} documents related to your query, but encountered an issue creating a synthesis. The most relevant document was titled: {sorted_docs[0].get('title', 'Untitled') if sorted_docs else 'Unknown'}"
            
        return state

    def run(self, query: str) -> str:
        """Runs the SED agent's complete workflow and returns the final, synthesized summary."""
        initial_state = SEDAgentState(
            query=query,
            original_query="",  # Will be set in optimize_query_step
            search_results=[],
            doc_evaluations=[],
            final_summary=""
        )
        try:
            final_state = self.graph.invoke(initial_state)
            logger.info("SEDAgent: Run complete.")
            return final_state['final_summary']
        except Exception as e:
            logger.error(f"SEDAgent: Workflow execution failed: {e}")
            return f"I encountered an error while processing your query: {e}"

