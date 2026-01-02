import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sentiment Insights",
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
st.title("🧠 Sentiment Insights")

st.markdown(
    """
    This section focuses on **Natural Language Processing (NLP)** applied to customer
    review text. It reveals how **expressed sentiment often contradicts star ratings**,
    leading to **overrated products** and unreliable demand signals.
    """
)

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------
avg_sentiment = df["sentiment_score"].mean()

negative_pct = (df["sentiment_label"] == "Negative").mean() * 100
neutral_pct = (df["sentiment_label"] == "Neutral").mean() * 100
positive_pct = (df["sentiment_label"] == "Positive").mean() * 100

high_rating_negative_pct = (
    df[(df["original_rating"] >= 4) & (df["sentiment_label"] == "Negative")]
    .shape[0] / len(df)
) * 100

# --------------------------------------------------
# KPI DISPLAY
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Avg Sentiment Score", round(avg_sentiment, 3))
col2.metric("Negative Sentiment (%)", f"{negative_pct:.1f}%")
col3.metric("Neutral Sentiment (%)", f"{neutral_pct:.1f}%")
col4.metric("High Rating + Negative (%)", f"{high_rating_negative_pct:.1f}%")

st.caption(
    "📌 *High Rating + Negative Sentiment indicates potential rating bias "
    "and is a strong signal for overrated products.*"
)

st.divider()

# --------------------------------------------------
# SENTIMENT SCORE DISTRIBUTION
# --------------------------------------------------
st.subheader("📊 Sentiment Score Distribution")

fig_dist = px.histogram(
    df,
    x="sentiment_score",
    nbins=50,
    title="Distribution of Sentiment Scores Across Reviews",
    labels={"sentiment_score": "Sentiment Score"},
    color_discrete_sequence=["#1f77b4"]
)

st.plotly_chart(fig_dist, use_container_width=True)

st.info(
    "Sentiment scores range from -1 (very negative) to +1 (very positive). "
    "A large concentration around neutral indicates mixed or ambiguous feedback."
)

st.divider()

# --------------------------------------------------
# RATING VS SENTIMENT (MISMATCH)
# --------------------------------------------------
st.subheader("⚖️ Rating vs Sentiment (Mismatch Detection)")

sample_df = df.sample(
    min(len(df), 20000),
    random_state=42
)

fig_scatter = px.scatter(
    sample_df,
    x="original_rating",
    y="sentiment_score",
    color="risk_level",
    title="User Rating vs Expressed Sentiment",
    labels={
        "original_rating": "User Rating",
        "sentiment_score": "Sentiment Score"
    }
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.warning(
    "Points in the upper-left region (high rating, low sentiment) "
    "represent **overrated reviews** where textual feedback does not support the rating."
)

st.divider()

# --------------------------------------------------
# SENTIMENT LABEL BREAKDOWN
# --------------------------------------------------
st.subheader("🧩 Sentiment Category Breakdown")

fig_pie = px.pie(
    df,
    names="sentiment_label",
    title="Proportion of Sentiment Labels",
    color="sentiment_label",
    color_discrete_map={
        "Positive": "#2ecc71",
        "Neutral": "#f1c40f",
        "Negative": "#e74c3c"
    }
)

st.plotly_chart(fig_pie, use_container_width=True)

# --------------------------------------------------
# HIGH-RISK SENTIMENT MISMATCH SUMMARY
# --------------------------------------------------
st.subheader("🚩 High-Risk Sentiment Mismatch Summary")

high_risk_df = df[
    (df["original_rating"] >= 4) &
    (df["sentiment_label"] == "Negative")
]

col1, col2, col3 = st.columns(3)

col1.metric(
    "High Rating + Negative Reviews",
    f"{len(high_risk_df):,}"
)

col2.metric(
    "Avg Rating (These Reviews)",
    round(high_risk_df["original_rating"].mean(), 2)
)

col3.metric(
    "Avg Sentiment Score",
    round(high_risk_df["sentiment_score"].mean(), 3)
)

st.info(
    "These reviews represent **critical risk cases** where customers rate products highly "
    "but express dissatisfaction in text. Such mismatches strongly contribute to "
    "**overrated products and forecasting errors**."
)


# --------------------------------------------------
# BUSINESS INTERPRETATION
# --------------------------------------------------
st.success(
    "🔍 **Key Insight:** NLP uncovers hidden dissatisfaction that star ratings alone miss. "
    "By integrating sentiment analysis, businesses can detect overrated products early, "
    "reduce return rates, and improve sales forecasting accuracy."
)
