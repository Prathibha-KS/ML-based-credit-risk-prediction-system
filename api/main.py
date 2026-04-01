from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

from src.feature_engineering import add_engineered_features

app = FastAPI(title="Credit Risk Prediction API")

# Load trained model
model = joblib.load("model.pkl")
explainer = shap.TreeExplainer(model)

# Define input schema
class CustomerData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/")
def home():
    return {"message": "Credit Risk Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):

    df = pd.DataFrame([data.dict()])
    df = add_engineered_features(df)

    probability = float(model.predict_proba(df)[:, 1][0])
    risk_score = int(probability * 100)

    # ----------------------------
    # Risk Level Classification
    # ----------------------------

    # Convert risk score into human readable category
    if risk_score < 30:
        risk_level = "Low Risk"
    elif risk_score < 60:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"

    # Decision Engine
    if risk_score < 30:
        decision = "Approve"
        reason = "Low credit utilization and strong repayment history."
    elif risk_score < 60:
        decision = "Manual Review Required"
        reason = "Moderate credit utilization and repayment behavior."
    else:
        decision = "Reject"
        reason = "High utilization and/or repayment delays detected."



    # ----------------------------
    # AI Financial Advisor
    # ----------------------------

    # advice = []

    # # Credit utilization check
    # credit_util = df["credit_utilization"].iloc[0]
    # if credit_util > 0.6:
    #     advice.append("Reduce credit utilization below 40% to lower default risk.")

    # # Payment delay check
    # if df["PAY_0"].iloc[0] > 0:
    #     advice.append("Avoid payment delays to maintain a healthy repayment history.")

    # # Payment-to-bill ratio
    # payment_ratio = df["avg_payment_ratio"].iloc[0]

    # if payment_ratio < 0.9:
    #     advice.append("Increase monthly payments closer to bill amount to improve repayment strength.")

    # # If customer is already strong
    # if len(advice) == 0:
    #     advice.append("Customer demonstrates strong credit behavior. Maintain current repayment discipline.")



    # SHAP explanation
    shap_values = explainer(df)
    values = shap_values.values[0]

    feature_importance = {
        feature: float(value)
        for feature, value in sorted(
            zip(df.columns, values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
    }

    return {
        "default_probability": round(probability, 4),

        # Score between 0–100
        "risk_score": risk_score,

        # Human readable risk level
        "risk_level": risk_level,

        # Final lending decision
        "decision": decision,

        # Explanation for decision
        "reason": reason,

        # Top 3 model drivers from SHAP
        "top_risk_drivers": feature_importance,

        #"financial_advice": advice
    }

@app.post("/simulate")
def simulate(data: CustomerData):

    df = pd.DataFrame([data.dict()])
    df = add_engineered_features(df)

    scenarios = {}

    # Scenario 1 — Payment delay increases
    df_delay = df.copy()
    df_delay["PAY_0"] = 2
    prob_delay = float(model.predict_proba(df_delay)[:,1][0])
    scenarios["If payment delay becomes 2 months"] = round(prob_delay*100,2)

    # Scenario 2 — Higher payment behavior
    df_payment = df.copy()
    df_payment["PAY_AMT1"] = df_payment["PAY_AMT1"] * 1.2
    prob_payment = float(model.predict_proba(df_payment)[:,1][0])
    scenarios["If payment increases by 20%"] = round(prob_payment*100,2)

    # Scenario 3 — Lower credit utilization
    df_util = df.copy()
    df_util["BILL_AMT1"] = df_util["BILL_AMT1"] * 0.6
    prob_util = float(model.predict_proba(df_util)[:,1][0])
    scenarios["If bill amount reduces by 40%"] = round(prob_util*100,2)

    return {"scenarios": scenarios}
