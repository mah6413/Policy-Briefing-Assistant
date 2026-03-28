import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

from rag_pipeline import run_policy_assistant


st.set_page_config(
    page_title="AI Policy Briefing Assistant",
    page_icon="📘",
    layout="wide",
)

st.title("📘 AI Policy Briefing Assistant")
st.caption(
    "Upload policy documents and ask grounded questions, or generate policy outputs."
)


# =========================
# Sidebar
# =========================

st.sidebar.header("Settings")

mode_label = st.sidebar.selectbox(
    "Choose output type",
    options=[
        "Question Answering",
        "Policy Memo",
        "Policy Brief",
        "Legislative Letter",
        "Executive Summary",
    ],
)

mode_map = {
    "Question Answering": "qa",
    "Policy Memo": "policy_memo",
    "Policy Brief": "policy_brief",
    "Legislative Letter": "legislative_letter",
    "Executive Summary": "executive_summary",
}

selected_mode = mode_map[mode_label]

st.sidebar.markdown("### Supported file types")
st.sidebar.write("PDF, DOCX, PPTX, XLSX")

api_key_present = bool(os.getenv("OPENAI_API_KEY"))
if api_key_present:
    st.sidebar.success("OPENAI_API_KEY detected")
else:
    st.sidebar.warning("OPENAI_API_KEY not found in environment")


# =========================
# Session state
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# =========================
# Helpers
# =========================

def save_uploaded_files(uploaded_files) -> List[str]:
    """
    Save uploaded Streamlit files to a temporary directory
    and return the local file paths.
    """
    saved_paths: List[str] = []

    temp_dir = Path(tempfile.mkdtemp())

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(str(file_path))

    return saved_paths


def render_sources(sources):
    if not sources:
        st.info("No source chunks available.")
        return

    for i, source in enumerate(sources, start=1):
        with st.expander(
            f"Source {i}: {source['document_title']} "
            f"(score: {source['score']:.3f})"
        ):
            st.write(source["text"])


# =========================
# File uploader
# =========================

uploaded_files = st.file_uploader(
    "Upload your policy documents",
    type=["pdf", "docx", "pptx", "xlsx"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded.")


# =========================
# Main input
# =========================

default_prompt = {
    "qa": "Ask a grounded question about the uploaded documents.",
    "policy_memo": "Enter the topic or request for the policy memo.",
    "policy_brief": "Enter the topic or request for the policy brief.",
    "legislative_letter": "Enter the topic or request for the legislative letter.",
    "executive_summary": "Enter the topic or request for the executive summary.",
}

user_query = st.text_area(
    "Your prompt",
    placeholder=default_prompt[selected_mode],
    height=150,
)

generate_clicked = st.button("Generate", type="primary")


# =========================
# Run assistant
# =========================

if generate_clicked:
    if not uploaded_files:
        st.error("Please upload at least one file first.")
    elif not user_query.strip():
        st.error("Please enter a question or request.")
    else:
        try:
            with st.spinner("Analyzing documents and generating response..."):
                st.write("✅ Button clicked")

                saved_file_paths = save_uploaded_files(uploaded_files)
                st.write("📂 Saved files:", saved_file_paths)

                result = run_policy_assistant(
                    user_query=user_query,
                    uploaded_file_paths=saved_file_paths,
                    mode=selected_mode,
                )

                st.write("🤖 Backend returned:", result)

                st.session_state.last_result = result
                st.session_state.chat_history.append(
                    {
                        "mode": mode_label,
                        "query": user_query,
                        "result": result,
                    }
                )

        except Exception as e:
            st.exception(e)

# =========================
# Show latest result
# =========================

if st.session_state.last_result:
    result = st.session_state.last_result

    st.subheader("Response")
    st.write(result["answer"])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Confidence", result.get("confidence", "unknown"))
    with col2:
        fallback_used = result.get("fallback_used", False)
        st.metric("Fallback Used", "Yes" if fallback_used else "No")

    st.subheader("Sources")
    render_sources(result.get("sources", []))


# =========================
# Chat/request history
# =========================

if st.session_state.chat_history:
    st.subheader("History")

    for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"{idx}. {item['mode']} — {item['query'][:80]}"):
            st.markdown(f"**Mode:** {item['mode']}")
            st.markdown(f"**Prompt:** {item['query']}")
            st.markdown("**Answer:**")
            st.write(item["result"]["answer"])

            st.markdown("**Sources:**")
            render_sources(item["result"].get("sources", []))