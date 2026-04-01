# 💳 ML-Based Credit Risk Prediction System

## 📌 Overview
This project predicts the probability of credit default using machine learning. It analyzes customer financial behavior such as credit limit, payment history, and billing patterns to classify users into risk categories.

The system is built as a full-stack ML application with a Streamlit dashboard, FastAPI backend, and an XGBoost model for prediction.

---

## 🎯 Objectives
- Predict customer credit risk (default probability)
- Apply ML techniques to financial data
- Provide real-time predictions via API
- Enable interactive risk analysis through a dashboard

---

## 🏗️ System Architecture
- **Frontend:** Streamlit dashboard for user input and visualization  
- **Backend:** FastAPI for handling requests and model inference  
- **Model:** XGBoost (final model after comparison with Logistic Regression)  

---

## 🔄 Workflow
1. User enters financial details in the dashboard  
2. Data is sent to FastAPI as a JSON request  
3. Backend preprocesses and validates input  
4. XGBoost model predicts default probability  
5. Result is returned and displayed with visual insights  

---

## 🧠 Model Details
- Baseline: Logistic Regression  
- Final Model: XGBoost (better accuracy and performance)  
- Output: Probability of default + risk classification  

---

## 📊 Features Used
- Credit Limit (LIMIT_BAL)  
- Age, Education  
- Payment History (PAY_0 to PAY_6)  
- Bill Amounts (BILL_AMT1 to BILL_AMT6)  
- Payment Amounts (PAY_AMT1 to PAY_AMT6)  

---

## ⚙️ Tech Stack
- Python  
- Pandas, NumPy  
- XGBoost  
- FastAPI  
- Streamlit  
- Plotly  
- SHAP  
- Joblib  

---

## 📈 Risk Scoring
- **0–30:** Low Risk → Approve  
- **30–60:** Medium Risk → Manual Review  
- **60–100:** High Risk → Reject  

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/ml-based-credit-risk-prediction-system.git
cd ml-based-credit-risk-prediction-system
pip install -r requirements.txt
