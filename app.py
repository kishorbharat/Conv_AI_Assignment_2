"""
app.py
------
Streamlit dashboard for the AIS-175 Regulatory Compliance Engine.

Run:
    streamlit run app.py
"""

import streamlit as st
from compliance_engine import ComplianceEngine
from predict_risk import predict_risk

st.set_page_config(page_title="AIS-175 Compliance Engine", layout="wide")

st.title("🚗 AIS-175 Regulatory Compliance Engine")
st.markdown(
    "Search AIS-175 rules and assess compliance risk for your clause text."
)

# ── Initialise the engine once per session ─────────────────────────────────
@st.cache_resource
def get_engine() -> ComplianceEngine:
    return ComplianceEngine()

engine = get_engine()

if not engine.rules:
    st.warning(
        "No rules loaded. Run `python save_clauses.py` first to ingest the AIS-175 PDF."
    )

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Max results", min_value=1, max_value=20, value=5)

# ── Query section ───────────────────────────────────────────────────────────
st.subheader("🔍 Query AIS-175 Rules")
question = st.text_input("Enter your compliance question:", placeholder="e.g. lighting requirements")

if st.button("Send Query") and question.strip():
    results = engine.query(question.strip(), top_k=top_k)
    if results:
        st.success(f"Found {len(results)} matching clause(s):")
        for rule in results:
            with st.expander(f"Clause {rule['id']}"):
                st.write(rule["text"])
    else:
        st.info("No matching clauses found. Try different keywords.")

# ── Risk assessment section ─────────────────────────────────────────────────
st.subheader("⚠️ Compliance Risk Assessment")
clause_input = st.text_area(
    "Enter clause text to assess:",
    placeholder="Paste or type the clause text you want to evaluate…",
    height=150,
)

if st.button("Assess Risk") and clause_input.strip():
    result = predict_risk(clause_input.strip())
    level = result["risk_level"]
    score = result["risk_score"]
    color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(level, "⚪")
    st.metric(label="Risk Level", value=f"{color} {level}")
    if level != "Unknown":
        st.progress(score)
        st.caption(f"Risk score: {score:.4f}")
    st.info(result["message"])
