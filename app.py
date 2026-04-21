import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import matplotlib.pyplot as plt
import matplotlib

# Set chart font
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Segoe UI']
matplotlib.rcParams['font.size'] = 11

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# BACKGROUND IMAGE
# ---------------------------

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("assets/churn_bg.png")

st.markdown(f"""
<style>

/* BACKGROUND IMAGE */
.stApp {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

/* DARK OVERLAY */
.stApp:before {{
content:"";
position:fixed;
top:0;
left:0;
width:100%;
height:100%;
background:rgba(0,0,0,0.30);
z-index:-1;
}}

/* TRANSPARENT GLASS CONTENT PANEL */
.block-container {{
background:rgba(255,255,255,0.35);
backdrop-filter: blur(6px);
padding:35px;
border-radius:12px;
border:1px solid rgba(255,255,255,0.4);
}}

/* TEXT COLOR */
h1, h2, h3, h4, h5, h6, label, p {{
color:black !important;
font-weight:500;
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("models/churn_model.pkl")

# ---------------------------
# PAGE STYLE
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------

st.markdown(
"""
<h1 style='text-align:center;'>📊 Customer Churn Prediction System</h1>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div style="
text-align:center;
font-size:18px;
padding:12px;
border-radius:8px;
background-color:#eef2f7;
border:1px solid #d9e2ef;
max-width:800px;
margin:auto;
">
<b>This application predicts whether a telecom customer will churn or stay using a trained Machine Learning model.</b>
</div>
""",
unsafe_allow_html=True
)

st.divider()


# ---------------------------
# INPUT FORM
# ---------------------------
with st.form("prediction_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["Yes","No"])
        tenure = st.slider("Tenure (months)",0,72,12)

    with col2:
        PhoneService = st.selectbox("Phone Service",["Yes","No"])
        MultipleLines = st.selectbox("Multiple Lines",["Yes","No","No phone service"])
        InternetService = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
        OnlineSecurity = st.selectbox("Online Security",["Yes","No","No internet service"])
        OnlineBackup = st.selectbox("Online Backup",["Yes","No","No internet service"])

    with col3:
        DeviceProtection = st.selectbox("Device Protection",["Yes","No","No internet service"])
        TechSupport = st.selectbox("Tech Support",["Yes","No","No internet service"])
        StreamingTV = st.selectbox("Streaming TV",["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])
        Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])

    col4, col5 = st.columns(2)

    with col4:
        PaperlessBilling = st.selectbox("Paperless Billing",["Yes","No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
        )

    with col5:
        MonthlyCharges = st.number_input("Monthly Charges",20.0,150.0,70.0)
        TotalCharges = st.number_input("Total Charges",20.0,9000.0,1000.0)

    predict_button = st.form_submit_button("🔍 Predict Customer Churn")

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
CustomerLifetimeValue = tenure * MonthlyCharges
AvgMonthlySpend = TotalCharges / (tenure + 1)

# ---------------------------
# CREATE INPUT DATAFRAME
# ---------------------------
input_data = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "CustomerLifetimeValue": CustomerLifetimeValue,
    "AvgMonthlySpend": AvgMonthlySpend
}])

# ---------------------------
# MANUAL ENCODING
# ---------------------------
gender_map = {"Female":0, "Male":1}
yes_no_map = {"No":0, "Yes":1}
internet_map = {"DSL":0, "Fiber optic":1, "No":2}
contract_map = {"Month-to-month":0, "One year":1, "Two year":2}

payment_map = {
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer (automatic)":2,
    "Credit card (automatic)":3
}

multi_map = {"No":0,"Yes":1,"No phone service":2}
internet_service_map = {"No":0,"Yes":1,"No internet service":2}

input_data["gender"] = input_data["gender"].map(gender_map)

for col in ["Partner","Dependents","PhoneService","PaperlessBilling"]:
    input_data[col] = input_data[col].map(yes_no_map)

input_data["MultipleLines"] = input_data["MultipleLines"].map(multi_map)
input_data["InternetService"] = input_data["InternetService"].map(internet_map)

for col in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
    input_data[col] = input_data[col].map(internet_service_map)

input_data["Contract"] = input_data["Contract"].map(contract_map)
input_data["PaymentMethod"] = input_data["PaymentMethod"].map(payment_map)

# ---------------------------
# PREDICTION
# ---------------------------
if predict_button:

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    prob_percent = round(probability * 100,2)

    # ---------------------------
    # RISK BADGE LOGIC 
    # ---------------------------
    if prob_percent >= 70:
        risk_label = "🔴 High Churn Risk"
        color = "red"
    elif prob_percent >= 40:
        risk_label = "🟡 Medium Churn Risk"
        color = "orange"
    else:
        risk_label = "🟢 Low Churn Risk"
        color = "green"

    # Show badge
    st.markdown(
        f"<h2 style='color:{color}; text-align:center'>{risk_label}</h2>",
        unsafe_allow_html=True
    )

    # ---------------------------
    # ORIGINAL RESULT DISPLAY
    # ---------------------------

    if prediction == 1:
        st.error("⚠️ High Risk Customer (Likely to Churn)")
    else:
        st.success("✅ Customer Likely to Stay")

    st.metric("Churn Probability", f"{prob_percent}%")
    st.progress(float(probability))

    # ---------------------------
    # CHART DASHBOARD
    # ---------------------------
    colA, colB = st.columns(2)

    # CHURN RISK CHART
    with colA:
        st.subheader("Churn Risk")

        fig, ax = plt.subplots(figsize=(7,3))
        

        # Make chart transparent
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        ax.barh(["Risk"], [prob_percent])
        ax.set_xlim(0,100)
        ax.set_xlabel("Probability %")
        ax.spines[['top','right']].set_visible(False)

        st.pyplot(fig)

    # FEATURE IMPORTANCE
    with colB:
        st.subheader("Top Feature Importance")

        try:
            importance = model.named_steps["model"].feature_importances_
        except:
            importance = model.feature_importances_

        feature_names = [
            "gender","SeniorCitizen","Partner","Dependents","tenure",
            "PhoneService","MultipleLines","InternetService","OnlineSecurity",
            "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
            "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
            "MonthlyCharges","TotalCharges","CustomerLifetimeValue","AvgMonthlySpend"
        ]

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False).head(8)

        fig2, ax2 = plt.subplots(figsize=(7,4))

        # Make chart transparent
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("none")

        ax2.barh(importance_df["Feature"], importance_df["Importance"])
        ax2.invert_yaxis()
        ax2.spines[['top','right']].set_visible(False)

        st.pyplot(fig2)

    # ---------------------------
    # INPUT SUMMARY
    # ---------------------------
    st.subheader("Customer Input Summary")
    st.dataframe(input_data)
