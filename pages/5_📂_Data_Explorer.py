import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Data Explorer",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA (CACHED)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed_reviews.parquet")

df = load_data()

# --------------------------------------------------
# PAGE TITLE & DESCRIPTION
# --------------------------------------------------
st.title("📂 Data Explorer")

st.markdown(
    """
    Explore the processed review data interactively.
    Use filters to analyze specific categories, products, sentiment ranges,
    and risk levels for deeper business insights.
    """
)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("🔎 Filters")

# Category filter
categories = sorted(df["category"].dropna().unique())
selected_categories = st.sidebar.multiselect(
    "Select Category",
    categories,
    default=categories
)

# Product filter (depends on category)
product_df = df[df["category"].isin(selected_categories)]
products = sorted(product_df["product_id"].unique())

selected_products = st.sidebar.multiselect(
    "Select Product ID",
    products,
    default=products[:50] if len(products) > 50 else products
)

# Rating range
rating_range = st.sidebar.slider(
    "Original Rating Range",
    min_value=1.0,
    max_value=5.0,
    value=(1.0, 5.0),
    step=0.5
)

# Sentiment range
sentiment_range = st.sidebar.slider(
    "Sentiment Score Range",
    min_value=-1.0,
    max_value=1.0,
    value=(-1.0, 1.0),
    step=0.1
)

# Risk level
risk_levels = sorted(df["risk_level"].unique())
selected_risk = st.sidebar.multiselect(
    "Risk Level",
    risk_levels,
    default=risk_levels
)

# Date range
min_date = df["month"].min().date()
max_date = df["month"].max().date()

date_range = st.sidebar.date_input(
    "Select Month Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# --------------------------------------------------
# APPLY FILTERS (SAFE)
# --------------------------------------------------
filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["product_id"].isin(selected_products)) &
    (df["original_rating"].between(*rating_range)) &
    (df["sentiment_score"].between(*sentiment_range)) &
    (df["risk_level"].isin(selected_risk)) &
    (df["month"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
]

# --------------------------------------------------
# HANDLE EMPTY FILTER RESULT
# --------------------------------------------------
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# --------------------------------------------------
# FILTERED KPI SUMMARY
# --------------------------------------------------
st.subheader("📊 Filtered Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Filtered Reviews", f"{len(filtered_df):,}")
col2.metric("Avg Original Rating", round(filtered_df["original_rating"].mean(), 2))
col3.metric("Avg Corrected Rating", round(filtered_df["corrected_rating"].mean(), 2))
col4.metric("Avg Overrated Index", round(filtered_df["overrated_index"].mean(), 2))

st.divider()

# --------------------------------------------------
# RATING VS SENTIMENT SCATTER
# --------------------------------------------------
st.subheader("📈 Rating vs Sentiment")

sample_df = filtered_df.sample(
    min(len(filtered_df), 15000),
    random_state=42
)

fig_scatter = px.scatter(
    sample_df,
    x="original_rating",
    y="sentiment_score",
    color="risk_level",
    labels={
        "original_rating": "Original Rating",
        "sentiment_score": "Sentiment Score"
    }
)

st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------
# RISK LEVEL DISTRIBUTION (FIXED)
# --------------------------------------------------
st.subheader("⚠️ Risk Level Distribution")

risk_counts = filtered_df["risk_level"].value_counts().reset_index()
risk_counts.columns = ["risk_level", "count"]

fig_risk = px.bar(
    risk_counts,
    x="risk_level",
    y="count",
    labels={
        "risk_level": "Risk Level",
        "count": "Number of Reviews"
    }
)

st.plotly_chart(fig_risk, use_container_width=True)

# --------------------------------------------------
# DATA TABLE
# --------------------------------------------------
st.subheader("📋 Filtered Data Table")

st.dataframe(
    filtered_df
    .sort_values("overrated_index", ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)

# --------------------------------------------------
# BUSINESS INTERPRETATION
# --------------------------------------------------
st.info(
    "This Data Explorer enables stakeholders to drill down into specific products, "
    "categories, and time periods. It supports quality audits, root-cause analysis, "
    "and data-driven business decisions."
)
