# prototype.py
#
# A threat intelligence analysis agent that ADAPTIVELY parses data based on file type.
# It implements a "Text-First" principle, treating unknown formats as text by default
# while using the mime_type to provide critical context to the AI analyst.

# ==============================================================================
# PART 1: IMPORTS & CONFIGURATION
# ==============================================================================

import os
import requests
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

# LangChain and LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API keys from environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SED_API_KEY = os.getenv("SED_API_KEY")

# Initialize the Gemini model for later use
try:
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"[!] Failed to initialize Gemini model. Please check your GOOGLE_API_KEY. Error: {e}")
    model = None

# The Target Profile: Defines the mission for the agent.
TARGET_PROFILE = {
    "description": "Investigate connections related to 'elon musk'",
    "query": "elon musk",
    "size": 30,
    "include": {}
}

# ==============================================================================
# PART 2: STATE DEFINITION & HELPER FUNCTIONS
# ==============================================================================

class WorkflowState(TypedDict):
    """Defines the state that flows through the graph."""
    profile: Dict[str, Any]
    raw_chunks: List[Dict]
    parsed_records: List[Dict[str, Any]]
    analysis_results: List[str]
    final_report: str

def find_relevant_json_object(data: Any, query: str) -> List[Dict]:
    """Recursively searches a JSON structure to find the smallest object(s) containing the query."""
    # This function remains the same
    results = []
    if isinstance(data, dict):
        if any(query in str(v) for v in data.values()):
            for k, v in data.items():
                results.extend(find_relevant_json_object(v, query))
            if not results:
                return [data]
        else:
            for k, v in data.items():
                results.extend(find_relevant_json_object(v, query))
    elif isinstance(data, list):
        for item in data:
            results.extend(find_relevant_json_object(item, query))
    return results

def isolate_relevant_line(chunk_content: str, query: str) -> str:
    """Finds the specific line in a text-based chunk containing the query."""
    # This function remains the same
    for line in chunk_content.splitlines():
        if query in line:
            return line.strip()
    return ""

# ==============================================================================
# PART 3: NODE DEFINITIONS (Workflow Steps)
# ==============================================================================

def search_node(state: WorkflowState) -> Dict[str, Any]:
    """Node that calls the Flashpoint API to get raw data chunks."""
    # This node remains the same
    profile = state["profile"]
    print(f"--- Node: search_node ---")
    print(f"[*] Calling Flashpoint API with query: '{profile['query']}'")
    api_url = "https://api.flashpoint.io/sources/v2/strategic-entities/chunks/search"
    headers = {"Authorization": f"Bearer {SED_API_KEY}", "Content-Type": "application/json"}
    body = {"query": profile["query"], "size": profile["size"], "include": profile["include"]}
    try:
        response = requests.post(api_url, headers=headers, json=body)
        response.raise_for_status()
        results = response.json().get("items", [])
        print(f"[+] Found {len(results)} items.")
        return {"raw_chunks": results}
    except requests.exceptions.RequestException as e:
        print(f"[!] API call failed: {e}")
        return {"raw_chunks": []}

def intelligent_parser_node(state: WorkflowState) -> Dict[str, Any]:
    """
    REFINED NODE: Implements the "Text-First" principle. It attempts to parse known
    structured types (JSON) and safely defaults to line-by-line parsing for all other types.
    """
    profile = state["profile"]
    raw_chunks = state["raw_chunks"]
    query = profile["query"]
    print(f"--- Node: intelligent_parser_node ---")
    
    parsed_records = []
    for i, chunk in enumerate(raw_chunks):
        content_str = chunk.get("chunk_content", "")
        file_meta = chunk.get("file", {})
        mime_type = file_meta.get("mime_type", "text/plain")
        file_path = file_meta.get("file_path", "N/A")
        
        if not content_str:
            continue
            
        print(f"  > Parsing chunk {i+1}/{len(raw_chunks)} from '{file_path}' (type: {mime_type})...")
        
        relevant_record = None
        
        # ** REFINED ROUTING LOGIC **
        # We apply special parsing ONLY for known structured formats.
        if 'json' in mime_type:
            try:
                content_json = json.loads(content_str)
                relevant_objects = find_relevant_json_object(content_json, query)
                if relevant_objects:
                    print(f"    [*] JSON format detected. Isolated {len(relevant_objects)} specific object(s).")
                    relevant_record = relevant_objects[0] # Take the most specific one
            except json.JSONDecodeError:
                # If parsing fails despite the mime_type, we fall back to text.
                print(f"    [!] Mime_type was json, but parsing failed. Defaulting to text-based line search.")
                relevant_record = isolate_relevant_line(content_str, query)
        else:
            # ** DEFAULT BEHAVIOR FOR ALL OTHER 44+ TEXT-BASED TYPES **
            # For 'text/plain', 'text/csv', 'text/html', etc., we use the robust line isolation.
            print(f"    [*] Non-JSON text format detected. Performing line-by-line search.")
            relevant_record = isolate_relevant_line(content_str, query)

        if relevant_record:
            parsed_records.append({
                "source_file": file_path,
                "file_type": mime_type,
                "record": relevant_record
            })

    print(f"[+] Parsing complete. Found {len(parsed_records)} relevant records to analyze.")
    return {"parsed_records": parsed_records}


def targeted_analysis_node(state: WorkflowState) -> Dict[str, Any]:
    """
    REFINED NODE: The prompt is now much more aware of the data's structure, leading
    to more accurate and relevant AI analysis.
    """
    parsed_records = state["parsed_records"]
    query = state["profile"]["query"]
    print(f"--- Node: targeted_analysis_node ---")

    if not parsed_records or not model:
        return {"analysis_results": []}

    # This prompt is now highly contextual, telling the AI what kind of file it's looking at.
    analysis_prompt = ChatPromptTemplate.from_template(
        """You are a professional threat analyst investigating the query "{query}".
You have found a record in a file with the mime type: "{file_type}".
Your task is to analyze ONLY the following data record and explain its significance in one sentence.
- If it's a JSON object, explain the key fields.
- If it's a line from a text or CSV file, identify what the other values in the line are.
Focus on the most important connection or piece of information for an investigator.

DATA RECORD:
---
{record}
---

ANALYST SUMMARY:"""
    )
    analysis_chain = analysis_prompt | model | StrOutputParser()

    print(f"[*] Performing context-aware analysis on {len(parsed_records)} records...")
    analysis_results = []
    for i, item in enumerate(parsed_records):
        print(f"  > Analyzing record {i+1}/{len(parsed_records)} from {item['source_file']}...")
        try:
            record_str = json.dumps(item['record'], indent=2) if isinstance(item['record'], dict) else str(item['record'])
            
            summary = analysis_chain.invoke({
                "query": query,
                "file_type": item['file_type'],
                "record": record_str
            })
            
            analysis_results.append(
                f"Finding:\n{summary}\n\nSource Evidence (from file: '{item['source_file']}', type: {item['file_type']}):\n{record_str}\n"
            )
        except Exception as e:
            analysis_results.append(f"Failed to analyze record {i+1}. Error: {e}")
    
    print("[+] Targeted analysis complete.")
    return {"analysis_results": analysis_results}

# The report_node and the graph execution code remain the same as the previous full script.
# The report_node is already designed to handle the improved output from the analysis node.
def report_node(state: WorkflowState) -> Dict[str, Any]:
    # This node remains the same as the previous version.
    analysis_results = state["analysis_results"]
    profile = state["profile"]
    print(f"--- Node: report_node ---")
    if not analysis_results:
        return {"final_report": "No relevant findings were generated from the data."}
    all_findings = "\n---\n".join(analysis_results)
    report_prompt = ChatPromptTemplate.from_template(
        """You are a lead intelligence analyst. Your team has conducted an investigation into "{query}".
They have produced a set of specific findings, each with its source evidence and file type.
Your job is to synthesize all of this information into a final intelligence report.
Start with a concise, high-level "Executive Summary" of 2-4 sentences that captures the most important takeaways. What does the analyst need to know right away?
Then, list each individual "Detailed Finding" along with its supporting evidence.

INVESTIGATION FINDINGS:
---
{findings}
---

FINAL REPORT:"""
    )
    report_chain = report_prompt | model | StrOutputParser()
    print("[*] Generating final intelligence report...")
    try:
        final_report_str = report_chain.invoke({"query": profile["description"],"findings": all_findings})
    except Exception as e:
        final_report_str = f"Could not generate report due to an error: {e}"
    print("[*] Final report compiled.")
    return {"final_report": final_report_str}

# ==============================================================================
# PART 4: GRAPH DEFINITION & EXECUTION
# ==============================================================================
workflow = StateGraph(WorkflowState)
workflow.add_node("search", search_node)
workflow.add_node("parse_records", intelligent_parser_node)
workflow.add_node("analyze_records", targeted_analysis_node)
workflow.add_node("generate_report", report_node)
workflow.set_entry_point("search")
workflow.add_edge("search", "parse_records")
workflow.add_edge("parse_records", "analyze_records")
workflow.add_edge("analyze_records", "generate_report")
workflow.add_edge("generate_report", END)
app = workflow.compile()

def run_investigation(user_query: str) -> str:
    """
    This function takes a user's query, runs the entire LangGraph workflow,
    and returns the final formatted report.
    """
    if not model:
        return "Workflow halted because the Gemini model could not be initialized. Please check your GOOGLE_API_KEY."

    print(f"--- Running investigation for query: {user_query} ---")
    
    # Dynamically create the TARGET_PROFILE from the user's input
    target_profile = {
        "description": f"Investigate connections related to '{user_query}'",
        "query": user_query,
        "size": 15,  # Keep size reasonable for web app speed
        "include": {}
    }
    
    initial_input = {"profile": target_profile}
    
    # We use .invoke() here as we just need the final result for the UI
    final_state = app.invoke(initial_input)
    
    final_report = final_state.get("final_report", "Report generation failed or no results found.")
    
    return final_report
