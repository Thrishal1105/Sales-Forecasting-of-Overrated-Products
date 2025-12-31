
import streamlit as st
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting of Overrated Products",
    layout="centered"
)

# --------------------------------------------------
# CUSTOM CSS (THEME + ANIMATIONS)
# --------------------------------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Card layout */
.card {
    background: rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
}

/* Title */
.main-title {
    font-size: 40px;
    font-weight: 800;
}

/* Subtitle */
.subtitle {
    color: #d1d5db;
    font-size: 16px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff7a18);
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
    font-weight: bold;
}

/* Glow animation */
.glow {
    animation: glow 1.6s infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 0 10px #ff4b4b; }
    to { box-shadow: 0 0 28px #ff4b4b; }
}

/* Success animation */
.success {
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
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
    This dashboard demonstrates how rating–review mismatch leads to overrated products,
    and how NLP + forecasting models correct this bias.
  </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model_data = joblib.load("model/ensemble_rating_model.pkl")

prophet_model = model_data["prophet_model"]
sarimax_model = model_data["sarimax_model"]
xgb_model = model_data["xgb_model"]
weights = model_data["weights"]

# --------------------------------------------------
# NLP
# --------------------------------------------------
sia = SentimentIntensityAnalyzer()

def extract_sentiment(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return sia.polarity_scores(text)['compound']

def review_level_rating(text, rating):
    sentiment = extract_sentiment(text)
    corrected = 0.6 * rating + 0.4 * ((sentiment + 1) * 2.5)
    return round(max(1, min(5, corrected)), 2), sentiment

def forecast_adjustment():
    future = prophet_model.make_future_dataframe(periods=1, freq="M")
    prophet_pred = prophet_model.predict(future)["yhat"].iloc[-1]
    sarimax_pred = sarimax_model.forecast(steps=1).iloc[0]
    xgb_pred = xgb_model.predict(np.array([[prophet_pred, sarimax_pred, prophet_pred]]))[0]

    return (
        weights["prophet"] * prophet_pred +
        weights["sarimax"] * sarimax_pred +
        weights["xgboost"] * xgb_pred
    )

def final_rating(text, rating):
    review_part, sentiment = review_level_rating(text, rating)
    forecast_part = forecast_adjustment()
    final = 0.7 * review_part + 0.3 * forecast_part
    return round(max(1, min(5, final)), 2), sentiment

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Simulate a Customer Review")

review_text = st.text_area(
    "Customer Review Text",
    placeholder="Example: It's good, but the battery died after two days."
)

user_rating = st.slider(
    "Customer Given Rating",
    1.0, 5.0, 5.0, 0.5
)

analyze = st.button("Analyze Rating")
st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# OUTPUT SECTION
# --------------------------------------------------
if analyze and review_text.strip() != "":
    corrected, sentiment = final_rating(review_text, user_rating)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⭐ Rating Comparison")

    col1, col2 = st.columns(2)
    col1.metric("❌ Without Model", user_rating)
    col2.metric("✅ With Model", corrected)

    if corrected < user_rating:
        st.markdown(
            "<div class='glow'>⚠️ <b>Overrated Product Detected</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='success'>✅ Rating aligns with sentiment</div>",
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------
    # EXPLANATION
    # --------------------------------------------------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 Explanation")

    if sentiment < -0.2:
        st.write("The review expresses **negative sentiment**, reducing the corrected rating.")
    elif sentiment > 0.2:
        st.write("The review expresses **positive sentiment**, so the rating remains high.")
    else:
        st.write("The review expresses **mixed sentiment**, leading to moderate correction.")

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# MODEL PERFORMANCE CHARTS
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Model Performance Comparison")

models = ["Prophet", "SARIMAX", "XGBoost", "Ensemble"]
mae = [0.131, 0.117, 0.191, 0.145]
rmse = [0.171, 0.159, 0.244, 0.187]

fig, ax = plt.subplots()
ax.bar(models, mae)
ax.set_title("MAE Comparison (Lower is Better)")
ax.set_ylabel("MAE")
st.pyplot(fig)

fig2, ax2 = plt.subplots()
ax2.bar(models, rmse)
ax2.set_title("RMSE Comparison (Lower is Better)")
ax2.set_ylabel("RMSE")
st.pyplot(fig2)

st.markdown('</div>', unsafe_allow_html=True)
