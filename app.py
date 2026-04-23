import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open('financial_health_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("💰 Financial Health Score Predictor")

st.write("Enter customer details:")

# =========================
# USER INPUTS
# =========================
age = st.number_input("Age", 18, 100, value=30)
income = st.number_input("Monthly Income", 0, 200000, value=15000)
debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0, value=1.0)
credit_lines = st.number_input("Number of Credit Lines", 0, 50, value=5)
late_payments = st.number_input("Late Payments (90 days)", 0, 50, value=0)
dependents = st.number_input("Number of Dependents", 0, 10, value=2)

# =========================
# FEATURE ENGINEERING
# =========================
income_per_dep = income / (dependents + 1)
late_payment_score = late_payments

# =========================
# CREATE INPUT DATAFRAME
# (MATCH TRAINING FEATURES EXACTLY)
# =========================
input_data = pd.DataFrame({
    'RevolvingUtilizationOfUnsecuredLines': [0],
    'age': [age],
    'NumberOfTime30-59DaysPastDueNotWorse': [0],
    'DebtRatio': [debt_ratio],
    'MonthlyIncome': [income],
    'NumberOfOpenCreditLinesAndLoans': [credit_lines],
    'NumberOfTimes90DaysLate': [late_payments],
    'NumberRealEstateLoansOrLines': [0],
    'NumberOfTime60-89DaysPastDueNotWorse': [0],
    'NumberOfDependents': [dependents],
    'Income_per_Dependent': [income_per_dep],
    'Late_Payment_Score': [late_payment_score]
})

# =========================
# SCALE INPUT
# =========================
input_scaled = scaler.transform(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💡 Financial Health Score: {round(prediction, 2)} / 100")

    # Optional interpretation
    if prediction > 75:
        st.success("🟢 Excellent Financial Health")
    elif prediction > 50:
        st.warning("🟡 Moderate Financial Health")
    else:
        st.error("🔴 Poor Financial Health")
