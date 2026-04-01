from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the repo root is on the path so the src package is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compliance_engine import ComplianceEngine  # noqa: E402

DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

RULES_PATH = DATA_DIR / "ais175_rules.json"
CLAUSES_PATH = DATA_DIR / "processed" / "clauses.json"
CLAUSES_PATH.parent.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_engine() -> ComplianceEngine:
    return ComplianceEngine(rules_path=RULES_PATH, clauses_path=CLAUSES_PATH)


engine = get_engine()

# ── Page layout ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AIS-175 Compliance Q&A", page_icon="📄", layout="wide")
st.title("📄 AIS-175 Regulatory Compliance Engine")

# ── Sidebar – PDF ingestion ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Ingest PDF")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        target = UPLOADS_DIR / uploaded_file.name
        target.write_bytes(uploaded_file.read())
        with st.spinner("Ingesting PDF…"):
            chunk_count = engine.ingest(target)
        st.success(f"✅ Ingested **{chunk_count}** chunks from *{uploaded_file.name}*")

    st.divider()
    chunks_loaded = len(engine.chunks)
    rules_loaded = len(engine.rules)
    st.metric("Chunks loaded", chunks_loaded)
    st.metric("Rules loaded", rules_loaded)

# ── Main area – query ─────────────────────────────────────────────────────────
st.subheader("🔍 Ask a Question")

with st.form("query_form"):
    question = st.text_area(
        "Enter your compliance question:",
        placeholder="e.g. What does AIS-175 say about emergency braking?",
        height=100,
    )
    top_k = st.slider("Number of excerpts to return (top_k)", min_value=1, max_value=10, value=3)
    submitted = st.form_submit_button("Send Query")

if submitted:
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Searching document…"):
            result = engine.ask(question.strip(), top_k=top_k)

        st.divider()
        st.subheader("📝 Answer")
        st.write(result.answer)

        if result.sources:
            st.divider()
            st.subheader("📌 Source Excerpts")
            for i, source in enumerate(result.sources, start=1):
                with st.expander(
                    f"Excerpt {i} – Page {source['page']}  |  chunk: {source['chunk_id']}  |  score: {source['score']}"
                ):
                    st.write(source["snippet"])
