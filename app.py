# app.py

import streamlit as st
import time
from agent_workflow import run_investigation # Import the agent's brain

# --- Function to load local CSS file ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Function for "hacker" typing effect ---
def type_effect(text, speed=0.01):
    # Create a placeholder for the text
    text_placeholder = st.empty()
    # Display the text character by character
    displayed_text = ""
    for char in text:
        displayed_text += char
        text_placeholder.markdown(displayed_text)
        time.sleep(speed)

# --- Page Configuration ---
st.set_page_config(
    page_title="Threat Intel // M4TR1X Interface",
    page_icon="🤖",
    layout="wide"
)

# Load the custom CSS
local_css("style.css")

# --- UI Components ---

# Header with Agent Smith and Neo images
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("images/neo.jpg", caption="The Analyst")
with col2:
    st.title("Threat Intel // M4TR1X Interface")
    st.write(">> Accessing clandestine data streams...")
with col3:
    st.image("images/agent_smith.jpg", caption="The Anomaly")

st.divider()

# Use a form to prevent the app from re-running on every keystroke
with st.form("search_form"):
    query_input = st.text_input(
        "**TARGET IDENTIFIER:**",
        placeholder="e.g., david.m.stroud@gmail.com or Mark Zuckerberg"
    )
    submitted = st.form_submit_button("EXECUTE QUERY")

# --- Workflow Execution ---
if submitted and query_input:
    # Show a spinner with a custom message
    with st.spinner(">> Connecting to the source... Analyzing rogue data packets..."):
        try:
            report = run_investigation(query_input)
            # Store the report in the session state so it persists
            st.session_state['report'] = report
        except Exception as e:
            st.error(f">>> SYSTEM FAILURE // Anomaly detected: {e}")
            st.session_state['report'] = None

# --- Display Results ---
if 'report' in st.session_state and st.session_state['report']:
    st.divider()
    st.subheader(">> INCOMING TRANSMISSION // ANALYST REPORT")
    
    # Use the custom typing effect to display the report
    type_effect(st.session_state['report'])