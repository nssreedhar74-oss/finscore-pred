import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('financial_health_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("💰 Financial Health Score Predictor")

st.write("Enter customer details to predict Financial Health Score")

# USER INPUTS
age = st.number_input("Age", 18, 100)
income = st.number_input("Monthly Income", 0, 200000)
debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0)
credit_lines = st.number_input("Number of Credit Lines", 0, 50)
late_payments = st.number_input("Number of Late Payments (90 days)", 0, 20)
dependents = st.number_input("Number of Dependents", 0, 10)

# FEATURE ENGINEERING (same as model)
income_per_dep = income / (dependents + 1)
late_payment_score = late_payments

# CREATE INPUT DATAFRAME
input_data = pd.DataFrame({
    'age': [age],
    'MonthlyIncome': [income],
    'DebtRatio': [debt_ratio],
    'NumberOfOpenCreditLinesAndLoans': [credit_lines],
    'NumberOfTimes90DaysLate': [late_payments],
    'NumberOfDependents': [dependents],
    'Income_per_Dependent': [income_per_dep],
    'Late_Payment_Score': [late_payment_score]
})

# SCALE INPUT
input_scaled = scaler.transform(input_data)

# PREDICT
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Financial Health Score: {round(prediction, 2)} / 100")
