# AIS-175 Regulatory Compliance Engine

An AI-powered tool for querying and assessing compliance with India's AIS-175
automotive safety standard.

---

## Project Structure

```
├── AIS 175_Final Draft_MARCH_2025.pdf  # Source regulation document
├── ais175_rules.json                   # Extracted clauses (generated)
├── risk_model.pkl                      # Trained risk model (generated)
├── pdf_ingestion.py                    # Extract clauses from the PDF
├── save_clauses.py                     # Shortcut: ingest PDF → JSON
├── compliance_engine.py                # Query engine (keyword search)
├── predict_risk.py                     # ML risk prediction
├── train_model.py                      # Train the risk classifier
├── query.py                            # Interactive terminal query tool
├── app.py                              # Streamlit web dashboard
└── requirements.txt
```

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

### 2 · Ingest the AIS-175 PDF

Extracts clauses from the PDF and saves them to `ais175_rules.json`:

```bash
python save_clauses.py
```

### 3 · Train the risk model *(optional)*

```bash
python train_model.py
```

---

## Querying in the Terminal

### Interactive session

```bash
python query.py
```

You will see a prompt where you can type any compliance question:

```
╔══════════════════════════════════════════════════════╗
║   AIS-175 Regulatory Compliance Query Engine         ║
║   Type your question and press Enter.                ║
║   Type 'exit' or press Ctrl+C to quit.               ║
╚══════════════════════════════════════════════════════╝

  Enter query: What are the lighting requirements?
```

### Single query (non-interactive)

Pass your question as a command-line argument:

```bash
python query.py "What are the lighting requirements for vehicles?"
```

---

## Sending a Query from Python

```python
from compliance_engine import ComplianceEngine

# Load the engine (reads ais175_rules.json automatically)
engine = ComplianceEngine()

# Send a query – returns a list of matching clause dicts
results = engine.query("What are the lighting requirements?")

for clause in results:
    print(f"Clause {clause['id']}: {clause['text']}")
```

### Using the `send_query` helper in `query.py`

```python
from query import send_query
from compliance_engine import ComplianceEngine

engine = ComplianceEngine()
results = send_query("braking distance standards", engine)

for clause in results:
    print(f"[{clause['id']}] {clause['text'][:200]}")
```

### Compliance check (match or no-match)

```python
from compliance_engine import ComplianceEngine

engine = ComplianceEngine()
result = engine.check_compliance("Vehicle shall be fitted with rear reflectors.")

print(result["compliant"])       # True / False
print(result["message"])         # Human-readable verdict
print(result["matched_rule"])    # The matching clause dict, or None
```

### Risk prediction

```python
from predict_risk import predict_risk

result = predict_risk("Failure to comply shall attract a penalty under the Act.")

print(result["risk_level"])   # "High" | "Medium" | "Low" | "Unknown"
print(result["risk_score"])   # float 0–1
print(result["message"])
```

---

## Streamlit Web Dashboard

```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

The dashboard lets you:
- Type a compliance question and browse matching AIS-175 clauses.
- Paste a clause and get an instant risk assessment.
