import streamlit as st
import pandas as pd
import pickle

# Load the trained Random Forest model
model = pickle.load(open("best_model.pkl", "rb"))

st.title("❤️👨‍⚕️ Heart Disease Prediction (AI Doctor)")
st.write("Enter patient details to predict the risk of heart disease.")

# All inputs as number_input for consistent numeric types
age = st.number_input("Age", min_value=1, max_value=100, step=1)
sex = st.number_input("Gender (Male: 1, Female: 0)", min_value=0, max_value=1, step=1)
# cp = st.number_input("Chest Pain Type (0–3)", min_value=0, max_value=3, step=1)
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, step=1)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=120, max_value=564, step=1)
fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (True: 1, False: 0)", min_value=0, max_value=1, step=1)
restecg = st.number_input("Resting ECG Result (0–2)", min_value=0, max_value=2, step=1)
thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=202, step=1)
exang = st.number_input("Exercise-Induced Angina (Yes: 1, No: 0)", min_value=0, max_value=1, step=1)
oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.2, step=0.1)
slope = st.number_input("Slope of ST Segment (0–2)", min_value=0, max_value=2, step=1)
ca = st.number_input("Number of Major Vessels (0–4)", min_value=0, max_value=4, step=1)
thal = st.number_input("Thalassemia Type (Normal: 0, Fixed: 1, Reversible: 2)", min_value=0, max_value=2, step=1)

# Prediction
if st.button("Predict Disease"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    pred = model.predict(input_df)[0]
    st.success(f"❤️👨‍⚕️ Estimated Heart Disease Risk: ⚕️ {pred:,.0f}")