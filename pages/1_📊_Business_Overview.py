import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Business Overview",
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
# KPI CALCULATIONS
# --------------------------------------------------
total_reviews = len(df)

avg_original_rating = df["original_rating"].mean()
avg_corrected_rating = df["corrected_rating"].mean()

avg_rating_gap = (df["original_rating"] - df["corrected_rating"]).mean()

overrated_pct = (
    (df["overrated_index"] > 0.8).sum() / total_reviews
) * 100

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------
st.title("📊 Business Overview")

st.markdown(
    """
    This dashboard provides a **high-level business view** of customer rating behavior.
    It highlights how rating–review mismatch leads to **overrated products**, which can
    negatively impact **customer trust, return rates, and demand forecasting accuracy**.
    """
)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Reviews", f"{total_reviews:,}")
col2.metric("Avg Original Rating", round(avg_original_rating, 2))
col3.metric("Avg Corrected Rating", round(avg_corrected_rating, 2))
col4.metric("Avg Rating Gap", round(avg_rating_gap, 2))
col5.metric("Overrated Reviews (%)", f"{overrated_pct:.1f}%")

st.caption(
    "📌 *Corrected ratings are generated using sentiment-aware modeling "
    "(finalized with Bagged LightGBM for best accuracy).*"
)

st.divider()

# --------------------------------------------------
# RATING DISTRIBUTION
# --------------------------------------------------
st.subheader("⭐ Original vs Corrected Rating Distribution")

rating_dist = df.melt(
    value_vars=["original_rating", "corrected_rating"],
    var_name="Rating Type",
    value_name="Rating"
)

fig_dist = px.histogram(
    rating_dist,
    x="Rating",
    color="Rating Type",
    barmode="overlay",
    nbins=20,
    opacity=0.7,
    title="Shift in Ratings After Sentiment Correction"
)

st.plotly_chart(fig_dist, use_container_width=True)

st.info(
    "The leftward shift in corrected ratings indicates that some highly-rated "
    "products receive negative or mixed feedback in textual reviews."
)

st.divider()

# --------------------------------------------------
# OVERRATED DISTRIBUTION
# --------------------------------------------------
st.subheader("⚠️ Overrated vs Normal Reviews")

df["overrated_flag"] = df["overrated_index"].apply(
    lambda x: "Overrated" if x > 0.8 else "Normal"
)

fig_pie = px.pie(
    df,
    names="overrated_flag",
    title="Proportion of Overrated Reviews",
    color="overrated_flag",
    color_discrete_map={
        "Overrated": "#ff4b4b",
        "Normal": "#2ecc71"
    }
)

st.plotly_chart(fig_pie, use_container_width=True)

st.warning(
    "Reviews marked as **Overrated** represent cases where customer sentiment "
    "does not support the given star rating. These cases are high-risk for "
    "returns, complaints, and inaccurate sales forecasting."
)

# --------------------------------------------------
# BUSINESS TAKEAWAY
# --------------------------------------------------
st.success(
    "🔍 **Business Insight:**\n\n"
    "- A noticeable gap exists between star ratings and true customer sentiment.\n"
    "- Overrated products can distort demand signals and inventory planning.\n"
    "- Applying sentiment-aware correction helps businesses make more reliable "
    "forecasting and product-quality decisions."
)
