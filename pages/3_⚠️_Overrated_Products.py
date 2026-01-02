import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Overrated Products Analysis",
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
st.title("⚠️ Overrated Products Analysis")

st.markdown(
    """
    This section identifies **products whose star ratings are inflated**
    compared to customer sentiment. Such products pose **business risk**
    by misleading demand forecasting, increasing return rates,
    and damaging long-term customer trust.
    """
)

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------
avg_overrated_index = df["overrated_index"].mean()

high_risk_reviews = (df["risk_level"] == "High").sum()

products_at_risk = df[df["risk_level"] == "High"]["product_id"].nunique()

# --------------------------------------------------
# KPI DISPLAY
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Avg Overrated Index", round(avg_overrated_index, 2))
col2.metric("High-Risk Reviews", f"{high_risk_reviews:,}")
col3.metric("Products at Risk", f"{products_at_risk:,}")

st.caption(
    "📌 *High-risk reviews indicate cases where sentiment contradicts the rating.*"
)

st.divider()

# --------------------------------------------------
# PRODUCT-LEVEL RISK AGGREGATION
# --------------------------------------------------
product_risk = (
    df.groupby("product_id")
    .agg(
        review_count=("product_id", "count"),
        avg_original_rating=("original_rating", "mean"),
        avg_corrected_rating=("corrected_rating", "mean"),
        avg_overrated_index=("overrated_index", "mean"),
        high_risk_count=("risk_level", lambda x: (x == "High").sum())
    )
    .reset_index()
)

# Filter to avoid noisy products with very few reviews
product_risk = product_risk[product_risk["review_count"] >= 20]

# --------------------------------------------------
# TOP OVERRATED PRODUCTS
# --------------------------------------------------
st.subheader("🚨 Top Overrated Products")

top_overrated = (
    product_risk
    .sort_values("avg_overrated_index", ascending=False)
    .head(10)
)

fig_top = px.bar(
    top_overrated,
    x="avg_overrated_index",
    y="product_id",
    orientation="h",
    title="Top 10 Products with Highest Overrated Index",
    labels={
        "avg_overrated_index": "Average Overrated Index",
        "product_id": "Product ID"
    }
)

st.plotly_chart(fig_top, use_container_width=True)

st.warning(
    "These products show the largest gap between customer ratings "
    "and sentiment, indicating potential quality or expectation issues."
)

st.divider()

# --------------------------------------------------
# CATEGORY-LEVEL ANALYSIS
# --------------------------------------------------
st.subheader("📦 Overrated Risk by Product Category")

category_risk = (
    df.groupby("category")["overrated_index"]
    .mean()
    .reset_index()
    .sort_values("overrated_index", ascending=False)
)

fig_category = px.bar(
    category_risk,
    x="category",
    y="overrated_index",
    title="Average Overrated Index by Category",
    labels={
        "overrated_index": "Average Overrated Index",
        "category": "Product Category"
    }
)

st.plotly_chart(fig_category, use_container_width=True)

st.info(
    "Categories with consistently higher overrated index values "
    "may require stricter quality control or customer expectation management."
)

st.divider()

# --------------------------------------------------
# RISK LEVEL DISTRIBUTION
# --------------------------------------------------
st.subheader("⚖️ Risk Level Distribution")

fig_risk_dist = px.pie(
    df,
    names="risk_level",
    title="Distribution of Review Risk Levels",
    color="risk_level",
    color_discrete_map={
        "High": "#e74c3c",
        "Medium": "#f1c40f",
        "Low": "#2ecc71"
    }
)

st.plotly_chart(fig_risk_dist, use_container_width=True)

st.divider()

# --------------------------------------------------
# ACTIONABLE PRODUCT TABLE
# --------------------------------------------------
st.subheader("📋 High-Risk Product Action List")

# action_table = (
#     product_risk
#     .sort_values("avg_overrated_index", ascending=False)
#     .head(20)
# )

action_table = (
    product_risk[product_risk["avg_overrated_index"] > 0.5]
    .sort_values("avg_overrated_index", ascending=False)
)


st.dataframe(
    action_table,
    use_container_width=True
)




# --------------------------------------------------
# BUSINESS TAKEAWAY
# --------------------------------------------------
st.success(
    "🔍 **Business Takeaway:**\n\n"
    "- Overrated products distort demand signals and inventory planning.\n"
    "- Early identification allows businesses to address quality gaps, "
    "update product descriptions, or intervene before negative impact grows.\n"
    "- Sentiment-aware analysis provides a more reliable foundation for "
    "sales forecasting and product strategy."
)
