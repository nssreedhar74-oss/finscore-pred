import streamlit as st
import pandas as pd
import pickle

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open('financial_health_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

# =========================
# UI
# =========================
st.title("💰 Financial Health Score Predictor")
st.write("Enter customer details:")

# =========================
# USER INPUTS
# =========================
age = st.number_input("Age", 18, 100, value=30)
income = st.number_input("Monthly Income", 0, 200000, value=15000)
debt_ratio = st.number_input("Debt Ratio", 0.0, 5.0, value=1.0)
credit_lines = st.number_input("Credit Lines", 0, 50, value=5)
late_payments = st.number_input("Late Payments (90 days)", 0, 50, value=0)
dependents = st.number_input("Dependents", 0, 10, value=2)

# =========================
# CREATE INPUT DATA
# =========================
input_data = pd.DataFrame({
    'age': [age],
    'MonthlyIncome': [income],
    'DebtRatio': [debt_ratio],
    'NumberOfOpenCreditLinesAndLoans': [credit_lines],
    'NumberOfTimes90DaysLate': [late_payments],
    'NumberOfDependents': [dependents]
})

# Ensure correct order
input_data = input_data[columns]

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    st.success(f"💡 Financial Health Score: {round(prediction, 2)} / 100")

    if prediction > 75:
        st.success("🟢 Excellent Financial Health")
    elif prediction > 50:
        st.warning("🟡 Moderate Financial Health")
st.write("Model Loaded Successfully")
st.write(model)
    else:
        st.error("🔴 Poor Financial Health")
