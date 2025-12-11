import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models and encoders
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('label_encoder_gender.pkl')
le_diabetic = joblib.load('label_encoder_diabetic.pkl')
le_smoker = joblib.load('label_encoder_smoker.pkl')
model = joblib.load('best_insurance_model.pkl')

# Streamlit UI
st.set_page_config(page_title="Insurance Premium Prediction", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to predict the insurance payment amount.")



with st.form("insurance_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    with col2:
        blood_pressure = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
        gender = st.selectbox("Gender", options=le_gender.classes_)
        diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_)
        smoker = st.selectbox("Smoker", options=le_smoker.classes_)

    submitted = st.form_submit_button("Predict Insurance Payment")

if submitted:
    # Prepare input
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'bmi': [bmi],
        'children': [children],
        'bloodpressure': [blood_pressure],   # Ensure column name matches training
        'diabetic': [diabetic],
        'smoker': [smoker]
    })

    # Encode categoricals
    input_data['gender'] = le_gender.transform(input_data['gender'])
    input_data['diabetic'] = le_diabetic.transform(input_data['diabetic'])
    input_data['smoker'] = le_smoker.transform(input_data['smoker'])

    # Scale numeric columns
    num_col = ['age', 'bmi', 'children', 'bloodpressure']
    input_data[num_col] = scaler.transform(input_data[num_col])
    
    correct_order = ['age', 'gender', 'bmi', 'children', 'smoker', 'diabetic', 'bloodpressure']
    input_data = input_data[correct_order]


    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"The predicted insurance payment amount is: ${np.round(prediction, 2)}")
