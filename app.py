import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('financial_health_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("💰 Financial Health Score Predictor")
st.write("Enter customer details:")

age = st.number_input("Age", 18, 100, value=35)
income = st.number_input("Monthly Income ($)", 0, 200000, value=5000, step=500)

# ✅ Fix: realistic range 0.0 to 1.0, not 5.0
debt_ratio = st.number_input(
    "Debt Ratio (0 = no debt, 1 = all income goes to debt)",
    min_value=0.0, max_value=1.0, value=0.3, step=0.01
)

credit_lines = st.number_input("Open Credit Lines & Loans", 0, 50, value=5)
late_payments = st.number_input("Times 90+ Days Late", 0, 50, value=0)
dependents = st.number_input("Number of Dependents", 0, 10, value=0)

input_data = pd.DataFrame({
    'age': [age],
    'MonthlyIncome': [income],
    'DebtRatio': [debt_ratio],
    'NumberOfOpenCreditLinesAndLoans': [credit_lines],
    'NumberOfTimes90DaysLate': [late_payments],
    'NumberOfDependents': [dependents]
})

input_data = input_data[columns]

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction = float(max(0, min(100, prediction)))  # ✅ safety clip

    st.success(f"💡 Financial Health Score: {round(prediction, 2)} / 100")
    st.progress(prediction / 100)

    if prediction > 75:
        st.success("🟢 Excellent Financial Health")
    elif prediction > 50:
        st.warning("🟡 Moderate Financial Health")
    else:
        st.error("🔴 Poor Financial Health")
