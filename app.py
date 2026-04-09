import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import run_policy_assistant

load_dotenv()

st.set_page_config(
    page_title="AI Policy Briefing Assistant",
    page_icon="📘",
    layout="wide",
)

st.title("AI Policy Briefing Assistant")
st.caption(
    "Generate policy answers, memos, briefs, letters, and executive summaries from uploaded documents."
)

with st.sidebar:
    st.header("Settings")

    demo_mode = st.toggle("Demo Mode (no OpenAI tokens)", value=True)

    if demo_mode:
        st.success("Demo Mode is ON")
        st.info(
            "The app uses uploaded document text to create sample outputs without calling the OpenAI API."
        )
    else:
        if os.getenv("OPENAI_API_KEY"):
            st.success("Live AI Mode is ON")
        else:
            st.error(
                "OPENAI_API_KEY not found. Add it to your environment or switch Demo Mode back on."
            )

    selected_mode = st.selectbox(
        "Output type",
        options=[
            "qa",
            "policy_memo",
            "policy_brief",
            "legislative_letter",
            "executive_summary",
        ],
        format_func=lambda x: {
            "qa": "Q&A",
            "policy_memo": "Policy Memo",
            "policy_brief": "Policy Brief",
            "legislative_letter": "Legislative Letter",
            "executive_summary": "Executive Summary",
        }[x],
    )

    st.markdown("---")
    st.markdown("**Tip:** Keep Demo Mode on for public sharing.")

st.info(
    "This demo can use uploaded document text for preview outputs without spending tokens. "
    "Turn Demo Mode off only when you want live AI generation."
)

uploaded_files = st.file_uploader(
    "Upload policy documents",
    type=["pdf", "docx", "pptx", "xlsx"],
    accept_multiple_files=True,
)

user_query = st.text_area(
    "Enter your prompt",
    placeholder="Example: Create a policy memo from the uploaded document.",
    height=140,
)

generate = st.button("Generate", type="primary")


def save_uploaded_files(files):
    saved_paths = []
    for uploaded_file in files:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            saved_paths.append(tmp_file.name)
    return saved_paths


if generate:
    if not user_query.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one document.")
        st.stop()

    saved_file_paths = save_uploaded_files(uploaded_files)

    try:
        with st.spinner("Generating response..."):
            result = run_policy_assistant(
                user_query=user_query,
                uploaded_file_paths=saved_file_paths,
                mode=selected_mode,
                demo_mode=demo_mode,
            )

        st.subheader("Response")
        st.write(result["answer"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", result.get("confidence", "unknown"))
        with col2:
            st.metric("Fallback used", "Yes" if result.get("fallback_used") else "No")
        with col3:
            st.metric("Sources returned", len(result.get("sources", [])))

        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources")
            for i, source in enumerate(sources, start=1):
                with st.expander(
                    f"Source {i}: {source.get('document_title', 'Untitled')}"
                ):
                    st.write(f"**Document type:** {source.get('document_type', 'unknown')}")
                    if "score" in source:
                        st.write(f"**Score:** {source['score']}")
                    st.write(source.get("text", ""))

    except Exception as e:
        st.error(f"Error: {e}")