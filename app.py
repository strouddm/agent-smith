# app.py

import streamlit as st
from agent_workflow import run_investigation

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

st.set_page_config(
    page_title="Agent Smith",
    page_icon="X",
    layout="wide"
)

local_css("style.css")

if 'final_report' not in st.session_state:
    st.session_state.final_report = ""
if 'discard_log' not in st.session_state:
    st.session_state.discard_log = []

st.title("agent smith")
st.write("hello mister anderson.")

with st.form("search_form"):
    query_input = st.text_input(
        "**entity of interest**",
        placeholder="e.g., zuck@fb.com or elon musk"
    )
    submitted = st.form_submit_button("start search")

if submitted and query_input:
    st.session_state.final_report = ""
    st.session_state.discard_log = []
    
    with st.spinner("Searching the Matrix... This may take a moment..."):
        results = run_investigation(query_input)
        st.session_state.final_report = results.get("report", "An unknown error occurred.")
        st.session_state.discard_log = results.get("discard_log", [])

if st.session_state.final_report:
    st.divider()

    if st.session_state.discard_log:
        with st.expander("View Discarded Chunk Log (Why some results were bypassed)"):
            for i, log_item in enumerate(st.session_state.discard_log):
                st.info(f"{log_item['reason']} in file: `{log_item['file_path']}`", icon="ℹ️")
                if st.checkbox("Show Content Preview", key=f"preview_{i}"):
                    st.code(log_item['content_preview'], language=None)
                if i < len(st.session_state.discard_log) - 1:
                    st.divider()

    st.subheader(">> Incoming Transmission // Agent Smith")
    
    if st.session_state.final_report.startswith("ERROR:"):
        st.error(st.session_state.final_report)
    else:
        st.markdown(st.session_state.final_report)