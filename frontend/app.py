import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="EV Logistics & Scenario Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ EV Logistics & Fleet Scenario Forecaster")
st.caption("End-to-end MLOps pipeline tracking Indian EV adoption, grid stress, and component demand.")

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Scenario Parameters")

scenario = st.sidebar.selectbox(
    "Policy / Market Condition",
    [
        "Baseline (Current Trend)",
        "Aggressive Subsidy Boost (+15%)",
        "Supply Chain Shortage (-10%)"
    ]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Window (Months)",
    min_value=3,
    max_value=12,
    value=6,
    step=1
)

api_url = st.sidebar.text_input(
    "Backend API URL",
    value="http://127.0.0.1:8000/predict"
)

# ----------------- DATA LOADING -----------------
DATA_PATH = "data/cleaned_monthly_ev.csv"

if not os.path.exists(DATA_PATH):
    st.error("Historical dataset missing. Please run `scripts/process_data.py` first.")
    st.stop()

history_df = pd.read_csv(DATA_PATH)
history_df["Date"] = pd.to_datetime(history_df["Date"])

# ----------------- FETCH PREDICTIONS -----------------
with st.spinner("Querying FastAPI forecasting engine..."):
    try:
        response = requests.post(
            api_url,
            json={
                "scenario_type": scenario,
                "months_to_forecast": forecast_horizon
            },
            timeout=5
        )
        if response.status_code == 200:
            pred_data = response.json()
            forecast_values = pred_data.get("forecast", [])
        else:
            st.error(f"API returned status code {response.status_code}")
            forecast_values = []
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to FastAPI. Ensure Uvicorn is running in terminal 1 on port 8000.")
        forecast_values = []

# ----------------- METRICS & LOGISTICS IMPACT -----------------
if forecast_values:
    last_historical_val = history_df["Registrations"].iloc[-1]
    projected_total = sum(forecast_values)
    avg_monthly_demand = int(sum(forecast_values) / len(forecast_values))
    pct_change = ((avg_monthly_demand - last_historical_val) / last_historical_val) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recent Actual Volume", f"{int(last_historical_val):,} units")
    col2.metric("Projected Total Demand", f"{projected_total:,} units", delta=f"{pct_change:.1f}%")
    col3.metric("Est. Battery Pack Need", f"{int(projected_total * 0.03):,} MWh")
    col4.metric("Charging Hub Expansion", f"{int(projected_total / 250):,} points")

    # ----------------- VISUALIZATION -----------------
    last_date = history_df["Date"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_horizon, freq="MS")

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Registrations": forecast_values,
        "Type": "Forecast"
    })

    fig = go.Figure()

    # Historical trace
    fig.add_trace(go.Scatter(
        x=history_df["Date"],
        y=history_df["Registrations"],
        mode="lines+markers",
        name="Historical Actuals",
        line=dict(color="#1f77b4", width=3)
    ))

    # Forecast trace
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Registrations"],
        mode="lines+markers",
        name=f"Forecast ({scenario})",
        line=dict(color="#ff7f0e", width=3, dash="dash")
    ))

    fig.update_layout(
        title="EV Registrations Trajectory vs. Policy Scenarios",
        xaxis_title="Timeline",
        yaxis_title="Monthly Vehicle Units",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Raw Data Table Toggle
    with st.expander("View Numerical Forecast Breakdown"):
        combined_export = pd.concat([
            history_df.assign(Status="Historical"),
            forecast_df.assign(Status="Projected")
        ], ignore_index=True)
        st.dataframe(combined_export, use_container_width=True)