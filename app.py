# app.py
# Autonomous IoT Fleet Monitor — Agentic AI demo (MVP)
# Minimal, debugged, no warnings. Free & open-source stack.

import warnings
warnings.filterwarnings("ignore")  # suppress non-critical warnings for a clean UI

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------
# Streamlit config & header
# ---------------------------
st.set_page_config(page_title="Autonomous IoT Fleet Monitor (Agentic AI)", page_icon="🚛", layout="wide")
st.title("🚛 Autonomous IoT Fleet Monitor — Agentic AI demo")
st.write(
    "Two cooperating agents analyze IoT telemetry to predict emissions and recommend maintenance. "
    "All open-source, lightweight, and ready for Streamlit Cloud."
)

# ---------------------------
# Load sample data safely
# ---------------------------
@st.cache_data
def load_sample_data():
    sample_path = "fleet_data.csv"
    if os.path.exists(sample_path):
        df0 = pd.read_csv(sample_path)
    else:
        # fallback inline data
        df0 = pd.DataFrame(
            {
                "vehicle_id": ["TRK001", "TRK002", "TRK003", "TRK004", "TRK005"],
                "speed": [60, 45, 75, 50, 65],
                "engine_temp": [85, 90, 80, 95, 88],
                "fuel_rate": [7.2, 8.0, 6.5, 8.5, 7.0],
                "co2_emissions": [180, 200, 160, 210, 175],
            }
        )
    # ensure correct dtypes and handle missing
    expected_cols = ["vehicle_id", "speed", "engine_temp", "fuel_rate", "co2_emissions"]
    for c in expected_cols:
        if c not in df0.columns:
            # create safe defaults if missing
            if c == "vehicle_id":
                df0[c] = [f"V{i+1:03d}" for i in range(len(df0))]
            else:
                df0[c] = 0
    # coerce numeric columns safely
    for col in ["speed", "engine_temp", "fuel_rate", "co2_emissions"]:
        df0[col] = pd.to_numeric(df0[col], errors="coerce").fillna(0.0)
    df0["vehicle_id"] = df0["vehicle_id"].astype(str)
    return df0

# ---------------------------
# UI: Data source controls
# ---------------------------
st.sidebar.header("Data")
use_sample = st.sidebar.button("Use sample dataset")
uploaded_file = st.sidebar.file_uploader("Or upload CSV (cols: vehicle_id,speed,engine_temp,fuel_rate,co2_emissions)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # normalize types
        for col in ["speed", "engine_temp", "fuel_rate", "co2_emissions"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                df[col] = 0.0
        if "vehicle_id" not in df.columns:
            df["vehicle_id"] = [f"V{i+1:03d}" for i in range(len(df))]
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        df = load_sample_data()
elif use_sample:
    df = load_sample_data()
else:
    # default view: show sample but don't run analysis until user presses the action button
    df = load_sample_data()
    st.info("Upload your CSV in the left panel or click 'Use sample dataset' in the sidebar to load sample data.")

# ---------------------------
# Preview data
# ---------------------------
st.subheader("Fleet data preview")
st.dataframe(df.reset_index(drop=True))

# ---------------------------
# Agent behaviors (deterministic, explainable)
# ---------------------------

def data_analyst_agent(data: pd.DataFrame):
    """
    Agent 1: summarizes telemetry, detects anomalies based on statistical thresholds.
    Returns: (thought_lines, anomaly_df)
    """
    thoughts = []
    thoughts.append("Agent 1 (Data Analyst): computing fleet-level statistics and searching for anomalies.")
    # statistics
    avg_speed = float(data["speed"].mean())
    avg_temp = float(data["engine_temp"].mean())
    avg_fuel = float(data["fuel_rate"].mean())
    thoughts.append(f"Average speed = {avg_speed:.1f} km/h; average engine temp = {avg_temp:.1f}°C; average fuel rate = {avg_fuel:.2f} L/h.")
    # anomaly rule: engine_temp > avg_temp + 5 OR fuel_rate > avg_fuel + 1
    anomaly_mask = (data["engine_temp"] > (avg_temp + 5)) | (data["fuel_rate"] > (avg_fuel + 1))
    anomalies = data[anomaly_mask].copy()
    thoughts.append(f"Detected {len(anomalies)} potential anomaly(ies) using threshold rules.")
    if len(anomalies) > 0:
        for _, r in anomalies.iterrows():
            thoughts.append(f"- Vehicle {r['vehicle_id']}: temp={r['engine_temp']}, fuel_rate={r['fuel_rate']}.")
    else:
        thoughts.append("No strong anomalies detected by thresholds.")
    return thoughts, anomalies

def maintenance_advisor_agent(anomalies: pd.DataFrame):
    """
    Agent 2: maps anomalies to recommended maintenance actions.
    Returns: list of thought lines (recommendations).
    """
    thoughts = []
    thoughts.append("Agent 2 (Maintenance Advisor): mapping anomalies to prioritized maintenance actions.")
    if anomalies.empty:
        thoughts.append("No immediate maintenance actions required based on current data.")
    else:
        # simple prioritization: higher engine_temp or higher fuel_rate -> higher priority
        anomalies = anomalies.sort_values(by=["engine_temp", "fuel_rate"], ascending=False)
        for _, row in anomalies.iterrows():
            priority = "HIGH" if (row["engine_temp"] > 92 or row["fuel_rate"] > 8.0) else "MEDIUM"
            thoughts.append(f"Recommend {priority} priority maintenance for {row['vehicle_id']}: check cooling system and fuel injector.")
    return thoughts

# ---------------------------
# Analysis action
# ---------------------------
if st.button("Run Agentic Analysis"):
    try:
        # Predict emissions using simple regression
        numeric_cols = ["speed", "engine_temp", "fuel_rate"]
        if df[numeric_cols].shape[0] < 1:
            st.warning("Not enough numeric rows to run model.")
        else:
            X = df[numeric_cols].astype(float).to_numpy()
            y = df["co2_emissions"].astype(float).to_numpy()
            # simple regression (no warnings)
            model = LinearRegression()
            model.fit(X, y)
            preds = model.predict(X)
            df = df.copy()
            df["predicted_emissions"] = np.round(preds, 2)
            # risk bands
            bins = [0, 170, 190, float("inf")]
            labels = ["Low", "Medium", "High"]
            df["emission_risk"] = pd.cut(df["predicted_emissions"], bins=bins, labels=labels, include_lowest=True)

            # Run agents
            analyst_thoughts, anomalies = data_analyst_agent(df)
            advisor_thoughts = maintenance_advisor_agent(anomalies)

            # Present agent thoughts then final table
            st.subheader("Agent 1 — Data Analyst (thoughts)")
            for t in analyst_thoughts:
                st.markdown(f"- {t}")

            st.subheader("Agent 2 — Maintenance Advisor (thoughts)")
            for t in advisor_thoughts:
                st.markdown(f"- {t}")

            st.subheader("Final AI Insights — predicted emissions & risk")
            st.dataframe(df[["vehicle_id", "predicted_emissions", "emission_risk"]].reset_index(drop=True))

            # summary
            counts = df["emission_risk"].value_counts().to_dict()
            high = int(counts.get("High", 0))
            medium = int(counts.get("Medium", 0))
            low = int(counts.get("Low", 0))
            st.markdown(
                f"**Summary:** {high} HIGH risk, {medium} MEDIUM risk, {low} LOW risk vehicles. "
                "High-risk vehicles should be prioritized for maintenance."
            )

            # allow download of results
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download analysis (CSV)", csv_bytes, "fleet_analysis.csv", "text/csv")
            st.success("Agentic analysis complete.")
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")

# footer
st.markdown("---")
st.caption("Demo by Amit — Autonomous IoT Fleet Monitor (Agentic AI). All open-source. For demo purposes only.")
