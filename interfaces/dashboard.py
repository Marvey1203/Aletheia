# interfaces/dashboard.py
# This is the Streamlit web application for the Aletheia Observatory.
# (Version 1.1 - Import Path Fix)

# --- FIX STARTS HERE: Add project root to the Python path ---
import sys
from pathlib import Path

# This adds the parent directory (e.g., 'aletheia/') to the system path
# so we can import from the 'core' module.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
# --- FIX ENDS HERE ---

import streamlit as st
import json

# Now this import should work correctly
from core.memory import MemoryGalaxy
from core.schemas import Trace, Attempt

# --- Page Configuration ---
# This MUST be the first Streamlit command.
st.set_page_config(
    page_title="Aletheia Observatory",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching ---
@st.cache_data
def load_traces_from_memory():
    """Loads all traces from the MemoryGalaxy."""
    memory = MemoryGalaxy()
    return memory.load_all_traces()

# --- UI Helper Functions ---
def display_trace(trace: Trace):
    """Renders a single, detailed view of a Trace object."""
    
    st.header(f"Trace: `{trace.trace_id}`", divider="rainbow")
    st.subheader(f"Query: \"{trace.query}\"")
    
    st.info(f"""
    **Answer:** {trace.summary.answer}  
    **Reasoning:** {trace.summary.reasoning}  
    **Next Action:** {trace.summary.next_action}
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Score", f"{trace.best.total:.4f}")
    col2.metric("Winning Attempt", f"#{trace.best.attempt_id}")
    col3.metric("Total Attempts", len(trace.attempts))

    with st.expander("Show AI's Reflection"):
        st.markdown(f"**What Worked:** {trace.reflection.what_worked}")
        st.markdown(f"**What Failed:** {trace.reflection.what_failed}")
        st.markdown(f"**Next Adjustment:** {trace.reflection.next_adjustment}")

    attempt_tabs = st.tabs([f"Attempt #{a.id}" for a in trace.attempts])
    for i, attempt in enumerate(trace.attempts):
        with attempt_tabs[i]:
            display_attempt(attempt)
    
    with st.expander("Show Raw Trace JSON"):
        trace_dict = trace.model_dump()
        st.json(json.dumps(trace_dict, indent=2))

def display_attempt(attempt: Attempt):
    """Renders a single Attempt object."""
    if attempt.plan:
        st.markdown("**Plan:**")
        plan_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(attempt.plan))
        st.code(plan_str, language="markdown")
    
    st.markdown("**Candidate:**")
    st.markdown(f"> {attempt.candidate}")

    st.markdown("**Scores:**")
    score_cols = st.columns(len(attempt.scores))
    for i, (key, value) in enumerate(attempt.scores.items()):
        score_cols[i].metric(key.capitalize(), f"{value:.2f}")
    
    if attempt.tool_calls:
        with st.expander("Show Tool Calls"):
            tool_calls_dict = [t.model_dump() for t in attempt.tool_calls]
            st.json(json.dumps(tool_calls_dict, indent=2))

# --- Main Page Application ---
def run_app():
    st.sidebar.title("🌌 Aletheia Observatory")
    st.sidebar.markdown("The window into the mind of your sovereign AI.")

    all_traces = load_traces_from_memory()

    if not all_traces:
        st.title("Welcome to the Observatory")
        st.warning("No traces found in the Memory Galaxy yet.")
        st.info("To generate your first trace, run the CLI from your terminal:")
        st.code("echo \"What is Aletheia?\" | python -m interfaces.cli reason --dummy")
    else:
        st.sidebar.header("Cognitive Traces")
        
        trace_options = {f"{trace.timestamp[:19]}Z - {trace.query[:50]}...": trace for trace in all_traces}
        
        selected_trace_key = st.sidebar.selectbox(
            "Select a trace to view",
            options=trace_options.keys(),
            label_visibility="collapsed"
        )
        
        if selected_trace_key:
            selected_trace = trace_options[selected_trace_key]
            display_trace(selected_trace)

    st.sidebar.markdown("---")
    if st.sidebar.button("Refresh Traces"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    run_app()