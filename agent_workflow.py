# agent_workflow.py

import os
import requests
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PROP_API_KEY = os.getenv("PROP_API_KEY") # Proprietary API

try:
    # Using a more advanced model for better reasoning and JSON formatting.
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"[!] Failed to initialize Gemini model: {e}")
    model = None

class WorkflowState(TypedDict):
    """Defines the state that flows through the graph."""
    profile: Dict[str, Any]
    raw_chunks: List[Dict]
    parsed_records: List[Dict[str, Any]]
    discard_log: List[Dict[str, str]]
    triaged_findings: List[Dict[str, Any]]
    final_report: str

def find_relevant_json_object(data: Any, query: str) -> List[Dict]:
    """Recursively finds ALL specific JSON objects containing the query, CASE-INSENSITIVE."""
    results = []
    query_lower = query.lower()
    if isinstance(data, dict):
        child_results = []
        for v in data.values():
            child_results.extend(find_relevant_json_object(v, query))
        if child_results:
            results.extend(child_results)
        elif any(query_lower in str(v).lower() for v in data.values()):
            results.append(data)
    elif isinstance(data, list):
        for item in data:
            results.extend(find_relevant_json_object(item, query))
    return results

def isolate_relevant_line(chunk_content: str, query: str, context_lines: int = 0) -> List[str]:
    """Finds ALL lines in a text-based chunk containing the query, CASE-INSENSITIVE."""
    lines = chunk_content.splitlines()
    matching_windows = []
    query_lower = query.lower()
    for i, line in enumerate(lines):
        if query_lower in line.lower():
            start_index = max(0, i - context_lines)
            end_index = min(len(lines), i + context_lines + 1)
            context_window = lines[start_index:end_index]
            matching_windows.append("\n".join(context_window))
    return matching_windows

def search_node(state: WorkflowState) -> Dict[str, Any]:
    profile = state["profile"]
    print(f"--- Node: search_node ---")
    api_url = "[INSERT API URL HERE]" # Proprietary API URL
    headers = {"Authorization": f"Bearer {PROP_API_KEY}", "Content-Type": "application/json"}
    body = {"query": profile["query"], "size": profile["size"], "include": profile["include"]}
    response = requests.post(api_url, headers=headers, json=body)
    response.raise_for_status()
    results = response.json().get("items", [])
    print(f"[+] Found {len(results)} items.")
    return {"raw_chunks": results}

def intelligent_parser_node(state: WorkflowState) -> Dict[str, Any]:
    profile = state["profile"]
    raw_chunks = state["raw_chunks"]
    query = profile["query"]
    context_lines = profile.get("context_lines", 0)
    print(f"--- Node: intelligent_parser_node ---")
    parsed_records, discard_log = [], []
    for i, chunk in enumerate(raw_chunks):
        content_str, file_meta = chunk.get("chunk_content", ""), chunk.get("file", {})
        mime_type, file_path = file_meta.get("mime_type", "text/plain"), file_meta.get("file_path", "N/A")
        if not content_str:
            discard_log.append({"reason": "Chunk content was empty.", "file_path": file_path, "content_preview": ""})
            continue
        relevant_records = []
        if 'json' in mime_type:
            try:
                relevant_records = find_relevant_json_object(json.loads(content_str), query)
            except json.JSONDecodeError:
                relevant_records = isolate_relevant_line(content_str, query, context_lines)
        else:
            relevant_records = isolate_relevant_line(content_str, query, context_lines)
        if relevant_records:
            for record in relevant_records:
                parsed_records.append({"source_file": file_path, "file_type": mime_type, "record": record})
        else:
            discard_log.append({"reason": "Query not found in content.", "file_path": file_path, "content_preview": content_str[:1000] + "..."})
    print(f"[+] Parsing complete. Found {len(parsed_records)} relevant records. Discarded {len(discard_log)} chunks.")
    return {"parsed_records": parsed_records, "discard_log": discard_log}

def triage_and_extract_node(state: WorkflowState) -> Dict[str, Any]:
    parsed_records = state["parsed_records"]
    query = state["profile"]["query"]
    print(f"--- Node: triage_and_extract_node ---")
    if not parsed_records or not model: return {"triaged_findings": []}
    
    triage_prompt = ChatPromptTemplate.from_template(
        """You are a data triage analyst. Assess if the following DATA RECORD is a "Primary Record" directly about the query '{query}', or just a "Contextual Mention" where the query is one item in a larger list.
A "Primary Record" is a user profile, credential pair, or article where the main subject IS the query.
A "Contextual Mention" is a log file or list where the query appears alongside many unrelated items.
Then, extract any PII (full name, username, email, phone, password) that is clearly part of the SAME logical entry as the query.

Return your analysis as a single, clean JSON object with these exact keys: "assessment", "confidence", "justification", and "associated_pii".
- assessment: "Primary Record" or "Contextual Mention".
- confidence: "High" or "Medium".
- justification: A one-sentence explanation.
- associated_pii: A JSON object of the PII you extracted, or an empty object if none.

DATA RECORD (from a file of type '{file_type}'):
---
{record}
---
"""
    )
    triage_chain = triage_prompt | model | StrOutputParser()
    print(f"[*] Triaging and tagging {len(parsed_records)} records...")
    triaged_findings = []
    for i, item in enumerate(parsed_records):
        print(f"  > Triaging record {i+1}/{len(parsed_records)}...")
        try:
            record_str = json.dumps(item['record'], indent=2) if isinstance(item['record'], dict) else str(item['record'])
            response_str = triage_chain.invoke({"query": query, "file_type": item['file_type'], "record": record_str})
            triage_data = json.loads(response_str.strip('` \njson'))
            triage_data['source_file'] = item['source_file']
            triaged_findings.append(triage_data)
        except Exception as e:
            print(f"  [!] Failed to triage record {i+1}. Error: {e}")
    print("[+] Triage complete.")
    return {"triaged_findings": triaged_findings}

def report_synthesis_node(state: WorkflowState) -> Dict[str, Any]:
    triaged_findings = state["triaged_findings"]
    profile = state["profile"]
    discard_log = state.get("discard_log", [])
    print(f"--- Node: report_synthesis_node ---")
    if not triaged_findings:
        final_report_str = "Investigation Complete: No Primary Records or meaningful connections were found for this query."
        # ... (rest of discard log handling)
        return {"final_report": final_report_str}

    primary_records = [f for f in triaged_findings if f.get("assessment") == "Primary Record"]
    source_map = {source: i + 1 for i, source in enumerate(sorted(list(set(f['source_file'] for f in primary_records))))}
    
    primary_findings_str = ""
    for finding in primary_records:
        pii_str = ", ".join([f"{k}: {v}" for k, v in finding.get("associated_pii", {}).items()])
        citation = f"[^{source_map[finding['source_file']]}]"
        primary_findings_str += f"- Justification: {finding.get('justification')} {citation}\n"
        if pii_str: primary_findings_str += f"  - Associated PII: {pii_str}\n"

    source_list_str = "\n".join([f"[{v}] {k}" for k, v in source_map.items()])
    discard_log_str = "\n".join([f"- {log['reason']} (File: {log['file_path']})" for log in discard_log]) if discard_log else "None"

    report_prompt = ChatPromptTemplate.from_template(
        """You are a lead intelligence analyst writing a final report on "{query}".
You have received a pre-triaged list of primary findings. Synthesize this into a report.

**FINAL REPORT DRAFT:**

**Executive Summary:**
(Based on the findings, write a 2-4 sentence summary of the key takeaways. Focus on what kind of PII and relationships were discovered.)

---
**Key Findings & Associated PII:**
(Based on the provided findings, create a clean, bulleted list. For each point, state the justification and list any associated PII with its source citation. Combine related findings if appropriate.)
{primary_findings}

---
**Sources:**
(List the numbered source files exactly as provided separated by a new line.)
{source_files}

---
**Discarded Chunks Log:**
(List the discard reasons exactly as provided.)
{discard_log}
"""
    )
    report_chain = report_prompt | model | StrOutputParser()
    print("[*] Generating final, triaged intelligence report...")
    try:
        final_report_str = report_chain.invoke({
            "query": profile["description"],
            "primary_findings": primary_findings_str if primary_findings_str else "No primary records were identified.",
            "source_files": source_list_str,
            "discard_log": discard_log_str
        })
    except Exception as e:
        final_report_str = f"Could not generate report due to an error: {e}"
    print(f"[*] Final report compiled.")
    return {"final_report": final_report_str}

workflow = StateGraph(WorkflowState)
workflow.add_node("search", search_node)
workflow.add_node("parse_records", intelligent_parser_node)
workflow.add_node("triage_and_extract", triage_and_extract_node)
workflow.add_node("generate_report", report_synthesis_node)
workflow.set_entry_point("search")
workflow.add_edge("search", "parse_records")
workflow.add_edge("parse_records", "triage_and_extract")
workflow.add_edge("triage_and_extract", "generate_report")
workflow.add_edge("generate_report", END)
app = workflow.compile()

def run_investigation(user_query: str):
    if not model: return {"report": "ERROR: Gemini model not initialized.", "discard_log": []}
    if not PROP_API_KEY: return {"report": "ERROR: Proprietary API key not set.", "discard_log": []}
    
    target_profile = {"description": f"Investigate '{user_query}'", "query": user_query, "size": 30, "include": {}, "context_lines": 1}
    initial_input = {"profile": target_profile, "discard_log": []}
    
    try:
        final_state = app.invoke(initial_input)
        return {"report": final_state.get("final_report"), "discard_log": final_state.get("discard_log", [])}
    except Exception as e:
        return {"report": f"ERROR: Workflow failed: {e}", "discard_log": []}