# AIS-175 Regulatory Compliance Engine

## Terminal Query Tool

Ask compliance questions directly in the terminal:

```bash
pip install -r requirements.txt
python query.py
```

Example session:

```
Query> What are the position reporting requirements?
  AIS175-002 – Position Reporting Interval
  Class A transponders must report position at intervals no greater than 10 seconds …

Query> vessel identification MMSI
  AIS175-001 – Vessel Identification
  Every vessel must broadcast a unique MMSI number and call sign …

Query> list        # show all loaded rules
Query> exit        # quit
```

## Streamlit Dashboard

```bash
streamlit run app.py
```
