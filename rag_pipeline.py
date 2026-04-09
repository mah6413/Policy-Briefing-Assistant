import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import openpyxl
from docx import Document
from openai import OpenAI
from pptx import Presentation
from pypdf import PdfReader

from dotenv import load_dotenv
load_dotenv()
# =========================
# OpenAI client setup
# =========================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your environment or Streamlit secrets."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Base system prompt
# =========================

BASE_SYSTEM_PROMPT = """
You are a Policy Briefing Assistant for nonprofit and advocacy organizations.

Your job is to:
- answer questions grounded in uploaded documents,
- summarize policy, legal, operational, and advocacy materials,
- draft professional policy memos, policy briefs, executive summaries, and legislative letters,
- clearly distinguish between what is supported by uploaded documents and what is not.

Rules:
- Do not fabricate facts, quotes, or citations.
- If the answer is not supported by the uploaded documents, say so clearly.
- Be professional, concise, and analytical.
- Do not present yourself as a lawyer and do not claim to provide formal legal advice.
""".strip()


# =========================
# Helper utilities
# =========================

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += max(1, chunk_size - overlap)

    return chunks


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def timestamp_str() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# =========================
# Document parsers
# =========================

def parse_pdf(file_path: str) -> str:
    text: List[str] = []
    reader = PdfReader(file_path)

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        text.append(f"\n[Page {i + 1}]\n{page_text}")

    return "\n".join(text)


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def parse_pptx(file_path: str) -> str:
    prs = Presentation(file_path)
    all_text: List[str] = []

    for i, slide in enumerate(prs.slides):
        slide_text = [f"[Slide {i + 1}]"]
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_text.append(shape.text)
        all_text.append("\n".join(slide_text))

    return "\n".join(all_text)


def parse_xlsx(file_path: str) -> str:
    wb = openpyxl.load_workbook(file_path, data_only=True)
    sheet_text: List[str] = []

    for sheet in wb.worksheets:
        sheet_text.append(f"[Sheet: {sheet.title}]")
        for row in sheet.iter_rows(values_only=True):
            values = [str(cell) for cell in row if cell is not None]
            if values:
                sheet_text.append(" | ".join(values))

    return "\n".join(sheet_text)


def parse_uploaded_file(file_path: str) -> Dict[str, str]:
    lower = file_path.lower()

    if lower.endswith(".pdf"):
        return {"document_type": "pdf", "text": parse_pdf(file_path)}
    if lower.endswith(".docx"):
        return {"document_type": "docx", "text": parse_docx(file_path)}
    if lower.endswith(".pptx"):
        return {"document_type": "pptx", "text": parse_pptx(file_path)}
    if lower.endswith(".xlsx"):
        return {"document_type": "xlsx", "text": parse_xlsx(file_path)}

    raise ValueError(f"Unsupported file format: {file_path}")


# =========================
# Build document library
# =========================

def build_document_library(file_paths: List[str]) -> List[Dict[str, Any]]:
    document_library: List[Dict[str, Any]] = []

    for idx, file_path in enumerate(file_paths, start=1):
        parsed = parse_uploaded_file(file_path)
        file_name = Path(file_path).name

        doc_record = {
            "document_id": f"doc_{idx}",
            "document_title": file_name,
            "document_type": parsed["document_type"],
            "raw_text": parsed["text"],
        }
        document_library.append(doc_record)

    return document_library


# =========================
# Build chunk store
# =========================

def build_chunk_store(
    document_library: List[Dict[str, Any]],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    chunk_store: List[Dict[str, Any]] = []

    for doc in document_library:
        chunks = chunk_text(doc["raw_text"], chunk_size=chunk_size, overlap=overlap)

        for i, chunk in enumerate(chunks, start=1):
            chunk_store.append(
                {
                    "chunk_id": f"{doc['document_id']}_chunk_{i}",
                    "document_id": doc["document_id"],
                    "document_title": doc["document_title"],
                    "document_type": doc["document_type"],
                    "text": chunk,
                    "embedding": None,
                }
            )

    return chunk_store


# =========================
# Embeddings
# =========================

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    response = client.embeddings.create(
        model=model,
        input=text,
    )
    return response.data[0].embedding


def embed_chunk_store(
    chunk_store: List[Dict[str, Any]],
    embedding_model: str = "text-embedding-3-small",
) -> List[Dict[str, Any]]:
    for chunk in chunk_store:
        if chunk["embedding"] is None:
            chunk["embedding"] = get_embedding(chunk["text"], model=embedding_model)

    return chunk_store


# =========================
# Retrieval
# =========================

def retrieve_relevant_chunks(
    question: str,
    chunk_store: List[Dict[str, Any]],
    top_k: int = 5,
    embedding_model: str = "text-embedding-3-small",
) -> List[Dict[str, Any]]:
    question_embedding = get_embedding(question, model=embedding_model)

    scored: List[Dict[str, Any]] = []
    for chunk in chunk_store:
        if chunk["embedding"] is None:
            continue

        score = cosine_similarity(question_embedding, chunk["embedding"])
        scored.append(
            {
                "score": score,
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "document_title": chunk["document_title"],
                "document_type": chunk["document_type"],
                "text": chunk["text"],
            }
        )

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def evaluate_retrieval_quality(retrieved_chunks: List[Dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "no_relevant_context"

    top_score = retrieved_chunks[0]["score"]

    if top_score < 0.25:
        return "no_relevant_context"
    if top_score < 0.55:
        return "partial"
    return "strong"


# =========================
# Question answering
# =========================

def answer_question(
    question: str,
    chunk_store: List[Dict[str, Any]],
    top_k: int = 5,
    chat_model: str = "gpt-4.1",
) -> Dict[str, Any]:
    retrieved = retrieve_relevant_chunks(question, chunk_store, top_k=top_k)
    quality = evaluate_retrieval_quality(retrieved)

    if quality == "no_relevant_context":
        return {
            "answer": "I could not find enough support for that answer in the uploaded document repository.",
            "sources": [],
            "confidence": "low",
            "fallback_used": True,
        }

    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(
            f"[Source {i} | {item['document_title']} | score={item['score']:.3f}]\n{item['text']}"
        )

    context = "\n\n".join(context_blocks)

    user_prompt = f"""
Question:
{question}

Retrieved Context:
{context}

Instructions:
- Answer primarily from the retrieved context.
- Be explicit when evidence is limited.
- Include a short source-based summary at the end.
""".strip()

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": retrieved,
        "confidence": "high" if quality == "strong" else "medium",
        "fallback_used": False,
    }


# =========================
# Grounded document generation
# =========================

def generate_grounded_output(
    user_request: str,
    chunk_store: List[Dict[str, Any]],
    output_type: str = "policy memo",
    top_k: int = 6,
    chat_model: str = "gpt-4.1",
) -> str:
    retrieved = retrieve_relevant_chunks(user_request, chunk_store, top_k=top_k)
    quality = evaluate_retrieval_quality(retrieved)

    if quality == "no_relevant_context":
        return f"I could not find enough support in the uploaded repository to generate a reliable {output_type}."

    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(f"[Source {i}: {item['document_title']}]\n{item['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
Create a {output_type} based on the uploaded document context.

User request:
{user_request}

Document context:
{context}

Requirements:
- Ground the output in the provided document context.
- Use professional nonprofit/public policy language.
- Do not invent facts.
- Note when evidence is limited.
""".strip()

    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": BASE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content


def generate_policy_memo(prompt_text: str, chunk_store: List[Dict[str, Any]]) -> str:
    request = f"""
Draft a policy memo.

Topic:
{prompt_text}

Structure:
- Title
- To
- From
- Date
- Subject
- Executive Summary
- Background
- Key Issues
- Analysis
- Recommendations
""".strip()

    return generate_grounded_output(request, chunk_store, output_type="policy memo", top_k=6)


def generate_policy_brief(prompt_text: str, chunk_store: List[Dict[str, Any]]) -> str:
    request = f"""
Draft a policy brief.

Topic:
{prompt_text}

Structure:
- Title
- Issue Overview
- Why It Matters
- Key Findings
- Policy Context
- Recommendations
- Conclusion
""".strip()

    return generate_grounded_output(request, chunk_store, output_type="policy brief", top_k=6)


def generate_legislative_letter(prompt_text: str, chunk_store: List[Dict[str, Any]]) -> str:
    request = f"""
Draft a legislative advocacy letter.

Topic:
{prompt_text}

Structure:
- Date
- Recipient placeholder
- Opening
- Statement of concern/support
- Evidence and rationale
- Policy ask
- Closing
""".strip()

    return generate_grounded_output(
        request,
        chunk_store,
        output_type="legislative letter",
        top_k=6,
    )


def generate_executive_summary(prompt_text: str, chunk_store: List[Dict[str, Any]]) -> str:
    request = f"""
Draft an executive summary.

Topic:
{prompt_text}

Structure:
- Title
- Overview
- Key Findings
- Implications
- Recommended Next Steps
""".strip()

    return generate_grounded_output(
        request,
        chunk_store,
        output_type="executive summary",
        top_k=6,
    )


# =========================
# Main helper for Streamlit
# =========================

def run_policy_assistant(
    user_query: str,
    uploaded_file_paths: List[str],
    mode: str = "qa",
) -> Dict[str, Any]:
    """
    mode options:
    - qa
    - policy_memo
    - policy_brief
    - legislative_letter
    - executive_summary
    """

    document_library = build_document_library(uploaded_file_paths)
    chunk_store = build_chunk_store(document_library)
    chunk_store = embed_chunk_store(chunk_store)

    if mode == "qa":
        return answer_question(user_query, chunk_store)

    if mode == "policy_memo":
        text = generate_policy_memo(user_query, chunk_store)
    elif mode == "policy_brief":
        text = generate_policy_brief(user_query, chunk_store)
    elif mode == "legislative_letter":
        text = generate_legislative_letter(user_query, chunk_store)
    elif mode == "executive_summary":
        text = generate_executive_summary(user_query, chunk_store)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    sources = retrieve_relevant_chunks(user_query, chunk_store, top_k=6)
    quality = evaluate_retrieval_quality(sources)

    return {
        "answer": text,
        "sources": sources,
        "confidence": "high" if quality == "strong" else "medium",
        "fallback_used": quality == "no_relevant_context",
    }