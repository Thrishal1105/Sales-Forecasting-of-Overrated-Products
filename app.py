import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting of Overrated Products",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
.card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}
.main-title {
    font-size: 38px;
    font-weight: 800;
}
.subtitle {
    color: #d1d5db;
}
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff7a18);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
.glow {
    animation: glow 1.6s infinite alternate;
}
@keyframes glow {
    from { box-shadow: 0 0 10px #ff4b4b; }
    to { box-shadow: 0 0 28px #ff4b4b; }
}
.metric-container .metric-label {
    font-size: 30px !important; /* Increase font size of metric headings */
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="card">
  <div class="main-title">📊 Sales Forecasting of Overrated Products</div>
  <p class="subtitle">
    Detecting rating–review mismatch using NLP and machine learning
    to correct biased customer ratings.
  </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------- 
# # SECTION 2: PROBLEM STATEMENT 
# # -------------------------------------------------- 
st.markdown(""" <div class="card"> <h3>❓ The Problem</h3> 
            <p> Customers often give high ratings while expressing dissatisfaction in text reviews. 
            This leads to <b>overrated products</b> and inaccurate demand forecasting. </p> </div> 
            """, unsafe_allow_html=True)

# --------------------------------------------------
# LOAD FINAL MODEL (Bagged LightGBM)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/bagged_lgbm_final.pkl")

bagged_lgbm = load_model()

# --------------------------------------------------
# NLP + FEATURE ENGINEERING
# --------------------------------------------------
sia = SentimentIntensityAnalyzer()

def extract_sentiment(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    return sia.polarity_scores(text)["compound"]

def build_features(review_text, user_rating):
    sentiment = extract_sentiment(review_text)

    corrected_rating = (
        0.6 * user_rating +
        0.4 * ((sentiment + 1) * 2.5)
    )

    overrated_index = user_rating - corrected_rating

    features = pd.DataFrame([{
        "sentiment_score": sentiment,
        "corrected_rating": corrected_rating,
        "overrated_index": overrated_index
    }])

    return features, sentiment

def predict_rating(review_text, user_rating):
    features, sentiment = build_features(review_text, user_rating)

    model_pred = bagged_lgbm.predict(features)[0]

    final_rating = (
        0.7 * model_pred +
        0.3 * ((sentiment + 1) * 2.5)
    )

    final_rating = round(max(1, min(5, final_rating)), 2)
    overrated = final_rating < user_rating

    return final_rating, sentiment, overrated

# --------------------------------------------------
# REVIEW SIMULATOR
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Review Simulator")

review_text = st.text_area(
    "Enter customer review",
    placeholder="Example: The product looks good, but battery failed quickly."
)

user_rating = st.slider(
    "Customer Given Rating",
    1.0, 5.0, 5.0, 0.5
)

analyze = st.button("Analyze Review")
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
if analyze and review_text.strip():
    final_rating, sentiment, overrated = predict_rating(review_text, user_rating)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⭐ Rating Comparison")

    

    c1, c2 = st.columns(2)
    c1.metric("Without Model", user_rating)
    c2.metric("✅ With Model", final_rating)

    if overrated:
        st.markdown("<div class='glow'>⚠️ Overrated Product Detected</div>", unsafe_allow_html=True)
    else:
        st.success("Rating aligns with sentiment")

    st.write("**Sentiment Score:**", round(sentiment, 3))
    st.markdown('</div>', unsafe_allow_html=True)



    # -------------------------------------------------- # 
    # SECTION 5: EXPLANATION 
    # # -------------------------------------------------- 

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Explanation") 
    if sentiment < -0.2:
         st.write("Negative sentiment detected despite high rating.") 
    elif sentiment > 0.2:
         st.write("Positive sentiment supports the given rating.")
    else: st.write("Mixed sentiment caused rating adjustment.")  # noqa: E701
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# MODEL PERFORMANCE – 6 MODEL BAR CHART
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Model Accuracy Comparison (All Models)")

accuracy_df = pd.DataFrame({
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
        0.012,
        0.037,
        0.112
    ]
})

fig = px.bar(
    accuracy_df,
    x="Model",
    y="MAE",
    color="Model",
    title="MAE Comparison Across All Tested Models (Lower is Better)"
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION HINT
# --------------------------------------------------
st.markdown("""
<div class="card">
<h4>📊 Analytics Pages</h4>
<p>
Use the sidebar to explore:
<ul>
<li>Business Overview</li>
<li>Sentiment Insights</li>
<li>Overrated Products</li>
<li>Forecast Impact</li>
<li>Data Explorer</li>
</ul>
</p>
</div>
""", unsafe_allow_html=True)
