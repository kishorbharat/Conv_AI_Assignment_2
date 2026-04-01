# AIS-175 Regulatory Compliance Engine

## Project Structure

```
.
├── dashboard/
│   └── app.py
├── data/
│   ├── ais175_rules.json
│   └── raw/
│       └── AIS 175_Final Draft_MARCH_2025.pdf
├── models/
├── src/
│   └── compliance_engine/
│       ├── __init__.py
│       ├── engine.py
│       ├── pdf_ingestion.py
│       ├── predict_risk.py
│       ├── save_clauses.py
│       └── train_model.py
└── requirements.txt
```

## Run

```bash
streamlit run dashboard/app.py
```

## Run API Server

```bash
uvicorn src.compliance_engine.api:app --host 0.0.0.0 --port 8001 --reload
```

## API Quick Test

Health:

```bash
curl http://localhost:8001/health
```

Ingest PDF:

```bash
curl -X POST "http://localhost:8001/ingest" \
	-F "file=@data/raw/AIS 175_Final Draft_MARCH_2025.pdf"
```

Ask query:

```bash
curl -X POST "http://localhost:8001/ask" \
	-H "Content-Type: application/json" \
	-d '{"question":"What does AIS-175 say about emergency braking?","top_k":3}'
```

OpenAPI docs:

```bash
http://localhost:8001/docs
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
