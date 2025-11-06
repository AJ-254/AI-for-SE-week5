import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("maternal_risk_model.pkl")

# Page configuration
st.set_page_config(page_title="Maternal Health Risk Predictor", page_icon="👩‍🍼", layout="centered")

# App title and description
st.title("👩‍⚕️ Maternal Health Risk Prediction App")
st.markdown("""
This AI-powered app predicts a **maternal health risk level** (Low, Medium, or High) 
based on clinical data.  
Fill in the fields below and click **Predict Risk** to get your result.
""")

# Input form
with st.form("risk_form"):
    st.subheader("🔢 Enter Patient Details")

    age = st.number_input("Age (years)", min_value=15, max_value=50, value=25)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=130, value=80)
    bs = st.number_input("Blood Sugar (mmol/L)", min_value=2.0, max_value=30.0, value=5.0, step=0.1)
    body_temp = st.number_input("Body Temperature (°C)", min_value=34.0, max_value=42.0, value=36.5, step=0.1)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=160, value=80)

    submitted = st.form_submit_button("Predict Risk")

# Prediction logic
if submitted:
    # Prepare input data as DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp],
        'HeartRate': [heart_rate]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Interpret prediction
    risk_label = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}.get(prediction, "Unknown")

    # Display result
    st.success(f"🩺 **Predicted Maternal Health Risk: {risk_label}**")

    # Add extra feedback
    if risk_label == "High Risk":
        st.warning("⚠️ High Risk detected! Immediate medical attention is advised.")
    elif risk_label == "Mid Risk":
        st.info("🧡 Medium Risk — regular monitoring and lifestyle adjustments are recommended.")
    else:
        st.balloons()
        st.success("✅ Low Risk — maintain a healthy lifestyle!")

# Footer
st.markdown("---")
st.caption("Developed as part of AI Week 5 Assignment • Powered by Streamlit & scikit-learn")
