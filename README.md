# Autonomous-Agentic-IoT-Fleet-Monitor-Agent
Agentic IoT Autonomous Car Fleet Monitor Agent Tool

# Autonomous IoT Fleet Monitor — Agentic AI TOOL

Minimal Streamlit MVP demonstrating an Agentic AI-style analysis of IoT fleet telemetry.
- Single feature: run agentic analysis that predicts emissions, flags anomalies, and recommends maintenance.
- Shows two cooperating agents: Data Analyst + Maintenance Advisor, with visible "thoughts".

## Files
- `app.py` — Streamlit app (main)
- `fleet_data.csv` — sample dataset (vehicle telemetry)
- `requirements.txt` — minimal packages (Streamlit Cloud friendly)

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py

