import streamlit as st
import pandas as pd
import joblib
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="MdTech AI",
    page_icon="❤️",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
}

.block-container {
    max-width: 800px;
    padding-top: 2rem;
}

.main-title {
    text-align: center;
    color: white;
    font-size: 3rem;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border: none;
    border-radius: 12px;
    background: linear-gradient(90deg, #ef4444, #dc2626);
    color: white;
    font-size: 20px;
    font-weight: bold;
}

.stButton > button:hover {
    transform: scale(1.02);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- HEADER ----------------

st.markdown("""
<div class="main-title">❤️ MdTech AI</div>
<div class="subtitle">
Advanced Heart Disease Risk Prediction System
</div>
""", unsafe_allow_html=True)

# ---------------- INPUT FORM ----------------

age = st.slider("👤 Age", 18, 100, 40)

sex = st.selectbox(
    "🚻 Sex",
    ["Male", "Female"]
)

chest_pain = st.selectbox(
    "💔 Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "🩺 Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    value=120
)

cholesterol = st.number_input(
    "🩸 Cholesterol (mg/dL)",
    min_value=100,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "🍬 Fasting Blood Sugar > 120 mg/dL",
    [0, 1]
)

resting_ecg = st.selectbox(
    "📈 Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "❤️ Max Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "🏃 Exercise-Induced Angina",
    ["Yes", "No"]
)

oldpeak = st.slider(
    "📉 Oldpeak (ST Depression)",
    0.0,
    6.0,
    1.0,
    0.1
)

st_slope = st.selectbox(
    "📊 ST Slope",
    ["Up", "Flat", "Down"]
)

# ---------------- PREDICT ----------------

if st.button("🔍 Predict Heart Risk"):

    with st.spinner("Analyzing patient data..."):
        time.sleep(1.5)

        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1,
            "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        # Probability
        try:
            probability = model.predict_proba(scaled_input)[0][1]
            risk_percent = round(probability * 100, 2)
        except:
            risk_percent = 50

    st.markdown("---")

    st.subheader("📊 Heart Disease Risk Score")

    st.metric(
        "Risk Percentage",
        f"{risk_percent}%"
    )

    st.progress(int(risk_percent))

    st.markdown("---")

    st.info(f"👤 Age: {age}")
    st.info(f"❤️ Max Heart Rate: {max_hr}")
    st.info(f"🩸 Cholesterol: {cholesterol}")

    st.markdown("---")

    if prediction == 1:

        st.error(
            f"⚠️ High Risk of Heart Disease ({risk_percent}%)"
        )

        if risk_percent >= 80:

            st.warning("""
### 🚨 Doctor Advice

• Consult a Cardiologist Immediately

• ECG Test Recommended

• Echocardiography Recommended

• Monitor Blood Pressure Daily

• Reduce Cholesterol Intake

• Avoid Smoking

• Avoid Alcohol

• Follow Doctor's Treatment Plan
""")

        elif risk_percent >= 60:

            st.warning("""
### ⚠️ Doctor Advice

• Schedule Doctor Appointment Soon

• Exercise Regularly

• Reduce Sugar Intake

• Monitor Blood Pressure

• Maintain Healthy Weight
""")

        else:

            st.info("""
### ℹ️ Doctor Advice

• Maintain Healthy Lifestyle

• Annual Heart Checkup Recommended

• Continue Physical Activity
""")

    else:

        st.success(
            f"✅ Low Risk of Heart Disease ({risk_percent}%)"
        )

        st.info("""
### 💚 Health Recommendations

• Continue Regular Exercise

• Eat More Fruits & Vegetables

• Stay Hydrated

• Sleep 7–8 Hours Daily

• Maintain Healthy Weight

• Annual Health Checkup
""")

    st.markdown("---")

    st.caption(
        "⚕️ This prediction is generated using a Machine Learning model and should not be considered a medical diagnosis. Please consult a qualified healthcare professional for medical advice."
    )