# 📊 Sales Forecasting of Overrated Products

### NLP-Driven Rating Correction & Advanced Forecasting Analytics

---

## 🔍 Project Overview

In modern e-commerce platforms, numerical ratings are widely used to measure customer satisfaction and forecast demand. However, many customers give **high star ratings while expressing dissatisfaction in review text**, resulting in **overrated products** and **misleading sales forecasts**.

This project solves that problem by integrating **Natural Language Processing (NLP)**, **sentiment-enhanced rating correction**, and **advanced forecasting models** into a unified, analytics-driven system.
The solution is delivered through an **interactive Streamlit dashboard** designed for business decision-makers.

---

## ❗ Problem Statement

A common real-world issue in e-commerce platforms:

> ⭐ Rating: *5/5*
> 📝 Review: *“Good product, but the battery stopped working after two days.”*

This mismatch causes:

* Overrated products
* Inflated demand forecasts
* Inventory misallocation
* Poor product and vendor decisions

Traditional forecasting models rely only on **historical ratings**, ignoring the **semantic meaning of customer feedback**.

---

## 🎯 Project Objectives

* Detect **rating–sentiment mismatches** in customer reviews
* Identify **overrated products** that pose business risk
* Improve **sales & demand forecasting accuracy**
* Provide **explainable analytics dashboards** for stakeholders
* Demonstrate **model comparison & justification** for academic evaluation

---

## 🧠 Proposed Solution Architecture

The system is designed as a **multi-layer pipeline**:

---

### 🔹 Layer 1: NLP-Based Rating Correction (Micro Level)

* Review text is processed using **VADER Sentiment Analysis**
* A **sentiment score** (–1 to +1) is extracted
* A **sentiment-adjusted rating** is computed using:

  * User-given rating
  * Sentiment-derived rating scale

📌 Output: **Corrected rating per review**

---

### 🔹 Layer 2: Forecasting & Stability Modeling (Macro Level)

Sentiment-corrected ratings are aggregated monthly and used for forecasting.

Models implemented:

* Prophet
* SARIMAX
* XGBoost

Purpose:

* Capture seasonality
* Reduce noise from biased ratings
* Provide baseline forecasting comparison

---

### 🔹 Layer 3: Advanced Machine Learning (Final Model)

To improve **accuracy and stability**, additional models were evaluated:

* **Bagged LightGBM (Best Performing Model)**
* CatBoost

Bagging improves:

* Variance reduction
* Stability on noisy review data
* Generalization across categories

📌 **Bagged LightGBM is used as the final production model**

---

## 🗂️ Dataset

* **Amazon All Beauty Reviews Dataset**
* ~700,000 customer reviews
* Time span: Multiple years

### Key Fields Used

* Review text
* Original rating
* Timestamp
* Product ID (ASIN)
* Category

### Preprocessing Steps

* Duplicate removal
* Text cleaning
* Sentiment extraction
* Rating correction
* Monthly aggregation
* Risk classification

Final processed dataset:

```
data/processed_reviews.parquet
```

---

## ⚙️ Technologies & Tools

### Programming & Core Libraries

* Python
* Pandas, NumPy
* Joblib

### NLP

* NLTK
* VADER Sentiment Analyzer

### Forecasting & ML

* Prophet
* SARIMAX
* XGBoost
* **Bagged LightGBM**
* CatBoost

### Visualization & UI

* Streamlit
* Plotly
* Custom HTML & CSS

### Deployment

* Hugging Face Spaces

---

## 📊 Model Evaluation & Comparison

All models were evaluated using:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**

### 🔹 Accuracy Comparison

| Model               | MAE ↓      | RMSE ↓     |
| ------------------- | ---------- | ---------- |
| Prophet             | 0.1310     | 0.1710     |
| SARIMAX             | 0.1060     | 0.1470     |
| XGBoost             | 0.1910     | 0.2440     |
| CatBoost            | 0.0015     | 0.0043     |
| **Bagged LightGBM** | **0.0003** | **0.0026** |
| 5-Model Ensemble    | 0.1120     | 0.1430     |
| Random Forest       | 0.0388     | 0.0610     |

🏆 **Conclusion:**
Bagged LightGBM provides the **best accuracy, stability, and robustness** and is therefore selected as the final prediction model.

---

## 🖥️ Streamlit Dashboard Structure

The application consists of **6 analytical pages**:

1. **📊 Business Overview**
   KPIs, rating gaps, overrated percentage

2. **🧠 Sentiment Insights**
   Rating vs sentiment mismatch analysis

3. **⚠️ Overrated Products Analysis**
   High-risk products & action lists

4. **📈 Forecast Impact Analysis**
   Raw vs corrected demand forecasting

5. **📂 Data Explorer**
   Interactive filters & deep-dive analysis

6. **📌 Project Summary**
   Executive-level insights & conclusions

---

## 📈 Business Value

This system enables organizations to:

* Detect inflated ratings early
* Reduce demand overestimation
* Improve inventory planning
* Prioritize product audits
* Align forecasts with real customer experience

---

## 🚀 Deployment

* **Framework:** Streamlit
* **Platform:** Hugging Face Spaces
* **Model Artifacts:** Stored via `joblib`

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Enhancements

* Aspect-based sentiment analysis
* Transformer-based NLP models
* Automated model retraining
* Real-time review ingestion
* Cross-category forecasting

---

## ✅ Conclusion

This project proves that **ratings alone are unreliable** for demand forecasting.
By integrating **NLP, sentiment-aware correction, and advanced ML models**, the system delivers:

* More realistic ratings
* Better sales forecasts
* Actionable business insights

The solution is **scalable, explainable, and production-ready**.

---

### **Domain**

**Data Science · NLP · Time-Series Forecasting · Business Analytics**




