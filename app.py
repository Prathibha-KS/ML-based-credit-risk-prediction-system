import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.title("🏦 Machine Learning-Based Credit Risk Prediction System")
st.markdown("Data-driven credit risk prediction using machine learning and financial behavior analysis.")

# ----------------------------
# Education mapping dictionary
# ----------------------------
# API expects numeric values but UI shows readable text
education_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}

# ----------------------------
# Customer Profile Section
# ----------------------------

st.header("Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input(
        "Credit Limit",
        value=50000,
        step=1000
    )  # step added for easier increment

with col2:
    age = st.slider(
        "Age",
        18,
        75,
        30
    )

with col3:
    education = st.selectbox(
        "Education Level",
        list(education_map.keys())  # show readable options
    )

# ----------------------------
# Financial Behavior Section
# ----------------------------

st.header("Financial Behavior (Last 6 Months)")

col4, col5, col6 = st.columns(3)

with col4:
    avg_bill = st.number_input(
        "Average Monthly Bill Amount",
        value=10000,
        step=500
    )

with col5:
    avg_payment = st.number_input(
        "Average Monthly Payment Amount",
        value=10000,
        step=500
    )

with col6:
    max_delay = st.slider(
        "Maximum Payment Delay (Months)",
        -1,   # -1 means paid early
        6,
        0
    )

# ----------------------------
# Predict Button
# ----------------------------

if st.button("🔍 Assess Credit Risk"):

    # Convert education text -> numeric
    education_value = education_map[education]

    # Data sent to FastAPI model
    data = {
        "LIMIT_BAL": limit_bal,
        "SEX": 1,                 # default value (dataset requirement)
        "EDUCATION": education_value,  # mapped numeric value
        "MARRIAGE": 1,            # default (can be added later in UI)
        "AGE": age,

        # Payment delay values replicated for 6 months
        "PAY_0": max_delay,
        "PAY_2": max_delay,
        "PAY_3": max_delay,
        "PAY_4": max_delay,
        "PAY_5": max_delay,
        "PAY_6": max_delay,

        # Bill amounts for 6 months
        "BILL_AMT1": avg_bill,
        "BILL_AMT2": avg_bill,
        "BILL_AMT3": avg_bill,
        "BILL_AMT4": avg_bill,
        "BILL_AMT5": avg_bill,
        "BILL_AMT6": avg_bill,

        # Payment amounts for 6 months
        "PAY_AMT1": avg_payment,
        "PAY_AMT2": avg_payment,
        "PAY_AMT3": avg_payment,
        "PAY_AMT4": avg_payment,
        "PAY_AMT5": avg_payment,
        "PAY_AMT6": avg_payment
    }

    # ----------------------------
    # Call FastAPI endpoint
    # ----------------------------

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = response.json()

        st.divider()

        # ----------------------------
        # Risk Result Section
        # ----------------------------

        st.subheader("📊 Risk Assessment Result")

        col7, col8 = st.columns(2)

        with col7:
            risk_score = result["risk_score"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={"text": "Credit Risk Score"},
                gauge={
                    "axis": {"range": [0,100]},
                    "bar": {"color": "black"},
                    "steps":[
                        {"range":[0,30],"color":"green"},
                        {"range":[30,60],"color":"orange"},
                        {"range":[60,100],"color":"red"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        with col8:
            st.metric(
                "Probability of Default",
                f"{result['default_probability']*100:.2f}%"
            )

            st.metric(
                "Risk Level",
                result["risk_level"]
            )

        # ----------------------------
        # Decision Display
        # ----------------------------


        st.subheader("🏦 Credit Decision")

        decision = result["decision"]

        if decision == "Approve":
            st.success("✅ APPROVED")
        elif decision == "Manual Review Required":
            st.warning("⚠️ MANUAL REVIEW REQUIRED")
        else:
            st.error("❌ REJECTED")

        st.markdown(f"**Explanation:** {result['reason']}")

        # ----------------------------
        # Risk Drivers Section
        # ----------------------------

        st.subheader("🔎 Key Risk Drivers")

        drivers = result["top_risk_drivers"]


        df_drivers = pd.DataFrame({
            "Feature": drivers.keys(),
            "Impact": drivers.values()
        })

        st.bar_chart(df_drivers.set_index("Feature"))


        st.subheader("🧠 Risk Simulation (What-If Analysis)")

        try:

            sim_response = requests.post(
                "http://127.0.0.1:8000/simulate",
                json=data
            )

            sim_result = sim_response.json()

            for scenario, value in sim_result["scenarios"].items():

                if value < 30:
                    st.success(f"{scenario} → Risk Score: {value}")
                elif value < 60:
                    st.warning(f"{scenario} → Risk Score: {value}")
                else:
                    st.error(f"{scenario} → Risk Score: {value}")

        except:
            st.write("Simulation not available.")



        # ----------------------------
        # AI Financial Advisor
        # ----------------------------

        # st.subheader("💡 AI Financial Advisor")

        # for tip in result["financial_advice"]:
        #     st.info(tip)




    # ----------------------------
    # Error Handling
    # ----------------------------

    except Exception as e:
        st.error("⚠️ Could not connect to the prediction API.")
        st.write("Make sure FastAPI server is running.")
        st.write(e)


    