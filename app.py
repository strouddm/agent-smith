# app.py

import streamlit as st
import time
from agent_workflow import run_investigation # Import the agent's brain

# --- Function to load local CSS file ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Function for "hacker" typing effect ---
def type_effect(text, speed=0.005): # Sped up the default typing
    text_placeholder = st.empty()
    displayed_text = ""
    # Add a cursor effect
    for char in text:
        displayed_text += char
        text_placeholder.markdown(displayed_text + "▌")
        time.sleep(speed)
    # Remove the cursor at the end
    text_placeholder.markdown(displayed_text)

# --- Page Configuration ---
st.set_page_config(
    page_title="Threat Intel // M4TR1X Interface",
    page_icon="🤖",
    layout="wide"
)

# Load the custom CSS
local_css("style.css")

# --- Initialize Session State (for storing the final report only) ---
if 'final_report' not in st.session_state:
    st.session_state.final_report = ""

# --- UI Components ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("images/neo.jpg", caption="The Analyst")
with col2:
    st.title("Threat Intel // M4TR1X Interface")
    st.write(">> Accessing clandestine data streams...")
with col3:
    st.image("images/agent_smith.jpg", caption="The Anomaly")

st.divider()

with st.form("search_form"):
    query_input = st.text_input(
        "**TARGET IDENTIFIER:**",
        placeholder="e.g., david.m.stroud@gmail.com or Mark Zuckerberg"
    )
    submitted = st.form_submit_button("EXECUTE QUERY")

# --- This block handles the entire investigation on a single submission ---
if submitted and query_input:
    # Use st.spinner for a simple loading animation. It will disappear when done.
    with st.spinner(">> Agent is investigating... Analyzing data packets... This may take a few moments..."):
        # The entire investigation runs here, and we wait for the final result.
        report = run_investigation(query_input)
        st.session_state.final_report = report

# --- Display Final Report ---
# This block will only run after the investigation is complete and a report is available.
if st.session_state.final_report:
    st.divider()
    st.subheader(">> INCOMING TRANSMISSION // ANALYST REPORT")
    
    # Check if the report is an error message
    if st.session_state.final_report.startswith("ERROR:"):
        st.error(st.session_state.final_report)
    else:
        # Use the typing effect for the final display
        type_effect(st.session_state.final_report)