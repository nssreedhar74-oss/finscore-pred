import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# LOAD FILES
# =========================
model = pickle.load(open('financial_health_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))  # ⭐ KEY FIX

st.title("💰 Financial Health Score Predictor")
st.write("Enter customer details:")

# =========================
# INPUTS
# =========================
age = st.number_input("Age", 18, 100, value=30)
income = st.number_input("Monthly Income", 0, 200000, value=15000)
debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0, value=1.0)
credit_lines = st.number_input("Credit Lines", 0, 50, value=5)
late_payments = st.number_input("Late Payments (90 days)", 0, 50, value=0)
dependents = st.number_input("Dependents", 0, 10, value=2)

# =========================
# FEATURE ENGINEERING
# =========================
income_per_dep = income / (dependents + 1)
late_payment_score = late_payments

# =========================
# CREATE EMPTY INPUT WITH CORRECT STRUCTURE
# =========================
input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill only relevant fields
input_data['age'] = age
input_data['MonthlyIncome'] = income
input_data['DebtRatio'] = debt_ratio
input_data['NumberOfOpenCreditLinesAndLoans'] = credit_lines
input_data['NumberOfTimes90DaysLate'] = late_payments
input_data['NumberOfDependents'] = dependents
input_data['Income_per_Dependent'] = income_per_dep
input_data['Late_Payment_Score'] = late_payment_score

# =========================
# SCALE + PREDICT
# =========================
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]

    st.success(f"💡 Financial Health Score: {round(prediction, 2)} / 100")

    if prediction > 75:
        st.success("🟢 Excellent Financial Health")
    elif prediction > 50:
        st.warning("🟡 Moderate Financial Health")
    else:
        st.error("🔴 Poor Financial Health")
