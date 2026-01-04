import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Forecast Impact Analysis",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed_reviews.parquet")

df = load_data()

# --------------------------------------------------
# PAGE TITLE & CONTEXT
# --------------------------------------------------
st.title("📈 Forecast Impact Analysis")

st.markdown(
    """
    This section demonstrates how **rating–sentiment mismatch impacts demand forecasting**.
    Raw star ratings are often used as a proxy for demand, but when ratings are inflated,
    forecasts become **overestimated**, leading to inventory imbalance and financial risk.
    """
)

# --------------------------------------------------
# MONTHLY AGGREGATION (FORECAST-READY)
# --------------------------------------------------
monthly = (
    df.groupby("month")
    .agg(
        raw_demand=("original_rating", "mean"),
        corrected_demand=("corrected_rating", "mean")
    )
    .reset_index()
)

monthly["forecast_gap"] = monthly["raw_demand"] - monthly["corrected_demand"]

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------
avg_overestimation = monthly["forecast_gap"].mean()

overestimation_pct = (
    (monthly["forecast_gap"] > 0).mean() * 100
)

max_gap = monthly["forecast_gap"].max()

best_model = "Bagged LightGBM (Lowest MAE)"

# --------------------------------------------------
# KPI DISPLAY
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Forecast Overestimation", round(avg_overestimation, 3))
col2.metric("Overestimated Periods (%)", f"{overestimation_pct:.1f}%")
col3.metric("Max Forecast Gap", round(max_gap, 3))
col4.metric("Best Performing Model", best_model)

st.caption(
    "📌 *Corrected demand is derived using sentiment-aware rating correction "
    "powered by the final Bagged LightGBM model.*"
)

st.divider()

# --------------------------------------------------
# RAW VS CORRECTED DEMAND TREND
# --------------------------------------------------
st.subheader("📉 Raw vs Sentiment-Adjusted Demand Trend")

fig_forecast = px.line(
    monthly,
    x="month",
    y=["raw_demand", "corrected_demand"],
    title="Impact of Sentiment Correction on Demand Estimation",
    labels={
        "value": "Average Rating (Demand Proxy)",
        "month": "Month",
        "variable": "Demand Type"
    }
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.info(
    "The gap between raw and corrected demand highlights how inflated ratings "
    "can mislead forecasting models when sentiment is ignored."
)

st.divider()

# --------------------------------------------------
# FORECAST GAP OVER TIME
# --------------------------------------------------
st.subheader("📊 Forecast Overestimation Gap Over Time")

fig_gap = px.bar(
    monthly,
    x="month",
    y="forecast_gap",
    title="Monthly Forecast Overestimation Gap",
    labels={
        "forecast_gap": "Overestimation Gap",
        "month": "Month"
    }
)

st.plotly_chart(fig_gap, use_container_width=True)

st.warning(
    "Persistent positive gaps indicate systematic overestimation, "
    "which can lead to excess inventory and higher holding costs."
)

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON (FINAL RESULTS)
# --------------------------------------------------
st.subheader("🏆 Forecast Model Performance Comparison")

model_metrics = pd.DataFrame({
    "Model": [
        "Prophet",
        "SARIMAX",
        "XGBoost",
        "Bagged LightGBM",
        "CatBoost",
        "5-Model Ensemble"
    ],
    "MAE": [
        0.131,
        0.106,
        0.191,
        0.015,
        0.040,
        0.112
    ],
    "RMSE": [
        0.171,
        0.148,
        0.244,
        0.038,
        0.060,
        0.143
    ]
})

fig_mae = px.bar(
    model_metrics,
    x="Model",
    y="MAE",
    title="Model Comparison – MAE (Lower is Better)",
    labels={"MAE": "Mean Absolute Error"}
)

st.plotly_chart(fig_mae, use_container_width=True)

fig_rmse = px.bar(
    model_metrics,
    x="Model",
    y="RMSE",
    title="Model Comparison – RMSE (Lower is Better)",
    labels={"RMSE": "Root Mean Squared Error"}
)

st.plotly_chart(fig_rmse, use_container_width=True)

# --------------------------------------------------
# BUSINESS TAKEAWAY
# --------------------------------------------------
st.success(
    "🔍 **Business Takeaway:**\n\n"
    "- Forecasts based only on star ratings tend to overestimate demand.\n"
    "- Sentiment-aware correction significantly reduces this bias.\n"
    "- Bagged LightGBM provides the most accurate and stable performance, "
    "making it suitable for production-level forecasting systems.\n"
    "- This approach helps organizations optimize inventory, reduce waste, "
    "and improve customer satisfaction."
)
