import tempfile
import os
import streamlit as st
import pandas as pd

from pdf_ingestion import extract_text_from_pdf, split_into_clauses
from save_clauses import save_clauses, load_clauses
from compliance_engine import load_rules, run_compliance_check
from predict_risk import predict_risk

st.set_page_config(page_title="AIS-175 Compliance Engine", layout="wide")
st.title("AIS-175 Regulatory Compliance Engine")

# --- Session state initialisation ---
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "processing" not in st.session_state:
    st.session_state.processing = False
if "results" not in st.session_state:
    st.session_state.results = None


def request_stop() -> None:
    """Callback: signal that the user wants to stop processing."""
    st.session_state.stop_requested = True


def is_stopped() -> bool:
    """Return True if the user has requested a stop."""
    return st.session_state.stop_requested


# --- Sidebar controls ---
st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])

run_btn = st.sidebar.button(
    "Run Compliance Check",
    disabled=st.session_state.processing,
)
stop_btn = st.sidebar.button(
    "⏹ Stop",
    on_click=request_stop,
    disabled=not st.session_state.processing,
    type="primary",
)

# --- Main processing logic ---
if run_btn and uploaded_file is not None:
    st.session_state.stop_requested = False
    st.session_state.processing = True
    st.session_state.results = None
    st.rerun()

if st.session_state.processing and uploaded_file is not None:
    status = st.status("Processing document…", expanded=True)

    with status:
        # Step 1: Extract text
        st.write("📄 Extracting text from PDF…")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        uploaded_file.seek(0)

        try:
            raw_text = extract_text_from_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)

        if is_stopped():
            status.update(label="⏹ Processing stopped by user.", state="error")
            st.session_state.processing = False
            st.stop()

        clauses = split_into_clauses(raw_text)
        st.write(f"✅ Extracted **{len(clauses)}** clauses.")

        # Step 2: Save clauses
        st.write("💾 Saving clauses…")
        save_clauses(clauses)

        if is_stopped():
            status.update(label="⏹ Processing stopped by user.", state="error")
            st.session_state.processing = False
            st.stop()

        # Step 3: Compliance check
        st.write("🔍 Running compliance check…")
        rules = load_rules()
        violations = run_compliance_check(clauses, rules, stop_flag=is_stopped)

        if is_stopped():
            status.update(label="⏹ Processing stopped by user.", state="error")
            st.session_state.processing = False
            st.stop()

        st.write(f"✅ Found **{len(violations)}** potential violations.")

        # Step 4: Risk prediction
        st.write("🤖 Predicting risk levels…")
        risk_results = predict_risk(clauses, stop_flag=is_stopped)

        if is_stopped():
            status.update(label="⏹ Processing stopped by user.", state="error")
            st.session_state.processing = False
            st.stop()

        st.write(f"✅ Risk prediction complete for **{len(risk_results)}** clauses.")
        status.update(label="✅ Processing complete!", state="complete")

    st.session_state.results = {
        "clauses": clauses,
        "violations": violations,
        "risk_results": risk_results,
    }
    st.session_state.processing = False

# --- Results display ---
if st.session_state.results:
    results = st.session_state.results
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Clauses", len(results["clauses"]))
    col2.metric("Violations Found", len(results["violations"]))
    high_risk = sum(1 for r in results["risk_results"] if r["risk"] == 1)
    col3.metric("High-Risk Clauses", high_risk)

    tab1, tab2, tab3 = st.tabs(["Compliance Violations", "Risk Predictions", "All Clauses"])

    with tab1:
        if results["violations"]:
            st.dataframe(pd.DataFrame(results["violations"]), use_container_width=True)
        else:
            st.info("No violations found.")

    with tab2:
        if results["risk_results"]:
            df_risk = pd.DataFrame(results["risk_results"])
            st.dataframe(df_risk, use_container_width=True)
        else:
            st.info("No risk prediction results.")

    with tab3:
        for i, clause in enumerate(results["clauses"], 1):
            st.text(f"{i}. {clause}")

elif not st.session_state.processing:
    st.info("Upload a PDF and click **Run Compliance Check** to begin.")
