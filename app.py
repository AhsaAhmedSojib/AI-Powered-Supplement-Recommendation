
import streamlit as st
import pandas as pd
import joblib

st.title("AI-Powered Supplement Recommendation")

st.markdown("Fill in patient biomarker data below:")
age = st.slider("Age", 18, 90, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
vit_d = st.number_input("Vitamin D (nmol/L)", 10.0, 150.0, 50.0)
ferritin = st.number_input("Ferritin (µg/L)", 10.0, 500.0, 100.0)
ldl = st.number_input("LDL Cholesterol (mmol/L)", 1.0, 7.0, 3.5)
testosterone = st.number_input("Testosterone (nmol/L)", 0.5, 30.0, 15.0)
hscrp = st.number_input("hsCRP (mg/L)", 0.1, 10.0, 1.0)
cortisol = st.number_input("Cortisol (nmol/L)", 50.0, 700.0, 300.0)
mch = st.number_input("MCH (pg)", 20.0, 40.0, 28.0)

features = ['Age', 'Sex', 'Vitamin_D', 'Ferritin', 'LDL', 'Testosterone', 'hsCRP', 'Cortisol', 'MCH']
if st.button("Predict Supplements"):
    input_data = pd.DataFrame([[age, 1 if sex == "Male" else 0, vit_d, ferritin, ldl, testosterone, hscrp, cortisol, mch]], columns=features)
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("supplement_model.joblib")
    label_cols = joblib.load("supplement_labels.joblib")

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_df = pd.DataFrame(prediction, columns=label_cols)

    st.subheader("Recommended Supplement Dose Classes:")
    for col in prediction_df.columns:
        st.write(f"{col}: Class {int(prediction_df[col].values[0])}")
