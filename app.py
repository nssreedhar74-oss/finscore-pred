import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Health Score Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .main .block-container { background: white; border-radius: 15px; padding: 2rem; margin: 1rem; }
    h1 { color: #2c3e50 !important; font-size: 2.2rem !important; }
    h2 { color: #34495e !important; }
    h3 { color: #7f8c8d !important; }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        margin: 5px;
    }
    .score-excellent { background: linear-gradient(135deg, #11998e, #38ef7d) !important; }
    .score-good      { background: linear-gradient(135deg, #f7971e, #ffd200) !important; }
    .score-poor      { background: linear-gradient(135deg, #cb2d3e, #ef473a) !important; }
    .stSlider > div > div { color: #667eea; }
</style>
""", unsafe_allow_html=True)

# ─── Model Training (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔧 Training model on 150,000 records...")
def train_model():
    df = pd.read_csv("cleaned_financial_data__2_.csv")

    # Drop multicollinear columns
    df.drop(['NumberOfTime30-59DaysPastDueNotWorse',
             'NumberOfTime60-89DaysPastDueNotWorse'], axis=1, inplace=True)

    # Feature engineering
    df['Income_per_Dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['Late_Payment_Score']   = df['NumberOfTimes90DaysLate']

    # Target: Financial Health Score
    df['Financial_Health_Score'] = (
        0.35 * (df['MonthlyIncome'] / df['MonthlyIncome'].max()) +
        0.25 * (1 - df['DebtRatio'].clip(0, 1)) +
        0.20 * (df['NumberOfOpenCreditLinesAndLoans'] / df['NumberOfOpenCreditLinesAndLoans'].max()) +
        0.20 * (1 - df['Late_Payment_Score'] / (df['Late_Payment_Score'].max() + 1))
    ) * 100
    df['Financial_Health_Score'] += np.random.normal(0, 2, len(df))
    df['Financial_Health_Score']  = df['Financial_Health_Score'].clip(0, 100)

    X = df.drop(['Financial_Health_Score', 'SeriousDlqin2yrs'], axis=1)
    y = df['Financial_Health_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train_s, y_train)

    feature_names = X.columns.tolist()
    return model, scaler, feature_names, df

model, scaler, feature_names, df_full = train_model()

# ─── Helper: Score Category ────────────────────────────────────────────────────
def score_category(score):
    if score >= 70:
        return "🟢 Excellent", "score-excellent", "Your financial health is strong. Keep maintaining low debt and consistent income."
    elif score >= 45:
        return "🟡 Moderate", "score-good", "You have average financial health. Focus on reducing debt ratio and late payments."
    else:
        return "🔴 At Risk", "score-poor", "Financial health needs immediate attention. Seek to reduce debt and improve income stability."

# ─── Gauge Chart ──────────────────────────────────────────────────────────────
def draw_gauge(score):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('white')

    # Colored arc segments
    for start, end, color in [(0, 0.45, '#e74c3c'), (0.45, 0.70, '#f39c12'), (0.70, 1.0, '#2ecc71')]:
        theta = np.linspace(np.pi * (1 - end), np.pi * (1 - start), 100)
        ax.fill_between(theta, 0.7, 1.0, color=color, alpha=0.85)

    # Needle
    needle_angle = np.pi * (1 - score / 100)
    ax.annotate('', xy=(needle_angle, 0.85), xytext=(needle_angle + np.pi, 0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2.5))

    ax.set_ylim(0, 1.1)
    ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)
    ax.axis('off')
    ax.text(np.pi / 2, -0.15, f"{score:.1f}", ha='center', va='center',
            fontsize=26, fontweight='bold', color='#2c3e50', transform=ax.transData)
    ax.text(np.pi / 2, -0.38, "Financial Health Score", ha='center', va='center',
            fontsize=9, color='#7f8c8d', transform=ax.transData)

    patches = [
        mpatches.Patch(color='#e74c3c', label='At Risk (0-45)'),
        mpatches.Patch(color='#f39c12', label='Moderate (45-70)'),
        mpatches.Patch(color='#2ecc71', label='Excellent (70-100)')
    ]
    ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.25),
              ncol=3, fontsize=7, frameon=False)
    plt.tight_layout()
    return fig

# ─── Feature Importance Chart ─────────────────────────────────────────────────
def draw_feature_importance():
    importances = model.feature_importances_
    labels = [f.replace('NumberOf', '# ').replace('RevolvingUtilization', 'Credit Util.')
                .replace('OfUnsecuredLines', '').replace('OpenCreditLinesAndLoans', 'Open Credits')
                .replace('RealEstateLoansOrLines', 'Real Estate')
                .replace('Times90DaysLate', '90-Day Late')
                .replace('Income_per_Dependent', 'Income/Dependent')
                .replace('Late_Payment_Score', 'Late Payment')
                .replace('MonthlyIncome', 'Monthly Income')
                .replace('DebtRatio', 'Debt Ratio')
                .replace('age', 'Age') for f in feature_names]

    sorted_idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
    ax.barh([labels[i] for i in sorted_idx], importances[sorted_idx], color=colors)
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title("Feature Importance", fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    return fig

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 Enter Your Details")
    st.markdown("---")

    monthly_income       = st.slider("💵 Monthly Income (₹)", 0, 50000, 5000, 500)
    debt_ratio           = st.slider("📊 Debt Ratio", 0.0, 2.0, 0.3, 0.01)
    revolving_util       = st.slider("💳 Credit Utilization (%)", 0.0, 1.0, 0.3, 0.01)
    age                  = st.slider("🎂 Age", 18, 100, 35)
    open_credits         = st.slider("🏦 Open Credit Lines", 0, 30, 5)
    real_estate_loans    = st.slider("🏠 Real Estate Loans", 0, 10, 1)
    times_90_days_late   = st.slider("⚠️ Times 90+ Days Late", 0, 20, 0)
    num_dependents       = st.slider("👨‍👩‍👧 Dependents", 0, 10, 1)

    st.markdown("---")
    predict_btn = st.button("🔮 Predict My Score", use_container_width=True, type="primary")

# ─── Main Page ────────────────────────────────────────────────────────────────
st.markdown("# 💰 Financial Health Score Predictor")
st.markdown("### *India-Focused Predictive Analytics Engine*")
st.markdown("---")

# KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📦 Dataset Size", "1,50,000", "records")
with col2:
    st.metric("🤖 Model", "Random Forest", "Regressor")
with col3:
    st.metric("📈 Features Used", "9", "variables")
with col4:
    st.metric("🎯 Target", "Score 0–100", "composite")

st.markdown("---")

if predict_btn:
    # Build input
    income_per_dep   = monthly_income / (num_dependents + 1)
    late_pay_score   = times_90_days_late

    input_data = pd.DataFrame([[
        revolving_util, age, debt_ratio, monthly_income,
        open_credits, times_90_days_late, real_estate_loans,
        num_dependents, income_per_dep, late_pay_score
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_data)
    score        = float(model.predict(input_scaled)[0])
    score        = np.clip(score, 0, 100)

    label, css_class, advice = score_category(score)

    # Result Section
    st.markdown("## 📊 Your Prediction Results")
    col_gauge, col_info = st.columns([1, 1])

    with col_gauge:
        fig_gauge = draw_gauge(score)
        st.pyplot(fig_gauge, use_container_width=True)

    with col_info:
        st.markdown(f"### Status: {label}")
        st.markdown(f"""
        <div class="metric-card {css_class}">
            <h1 style='font-size:3rem; margin:0;'>{score:.1f}</h1>
            <p style='margin:0; font-size:1rem;'>out of 100</p>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"**💡 Advice:** {advice}")

        # Input summary
        st.markdown("#### 📋 Your Input Summary")
        summary_df = pd.DataFrame({
            "Parameter": ["Monthly Income", "Debt Ratio", "Credit Utilization",
                          "Age", "Open Credits", "Late Payments (90d)", "Dependents"],
            "Value": [f"₹{monthly_income:,}", f"{debt_ratio:.2f}", f"{revolving_util*100:.0f}%",
                      str(age), str(open_credits), str(times_90_days_late), str(num_dependents)]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Feature importance + score breakdown
    col_fi, col_bd = st.columns([1, 1])
    with col_fi:
        st.markdown("#### 🔍 What Drives the Score?")
        st.pyplot(draw_feature_importance(), use_container_width=True)

    with col_bd:
        st.markdown("#### 📐 Score Formula Breakdown")
        income_contrib  = 0.35 * (monthly_income / 50000) * 100
        debt_contrib    = 0.25 * max(0, 1 - debt_ratio) * 100
        credit_contrib  = 0.20 * (open_credits / 30) * 100
        late_contrib    = 0.20 * max(0, 1 - times_90_days_late / 21) * 100

        breakdown_df = pd.DataFrame({
            "Component": ["Income (35%)", "Low Debt (25%)", "Credit Lines (20%)", "Payment History (20%)"],
            "Score Contribution": [f"{income_contrib:.1f}", f"{debt_contrib:.1f}",
                                   f"{credit_contrib:.1f}", f"{late_contrib:.1f}"]
        })
        st.dataframe(breakdown_df, hide_index=True, use_container_width=True)

        st.markdown("##### 💡 Key Recommendations")
        tips = []
        if debt_ratio > 0.5:
            tips.append("📉 Reduce your debt ratio below 0.5")
        if times_90_days_late > 0:
            tips.append("📅 Avoid late payments — biggest risk factor")
        if monthly_income < 3000:
            tips.append("💰 Increasing income will boost score significantly")
        if revolving_util > 0.5:
            tips.append("💳 Keep credit utilization below 30%")
        if not tips:
            tips.append("✅ Great profile! Keep maintaining your habits.")
        for t in tips:
            st.markdown(f"- {t}")

else:
    # Landing state
    st.markdown("## 👈 Enter your details in the sidebar and click **Predict My Score**")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 🎯 About This Model
        This tool predicts a **Financial Health Score (0–100)** based on:
        - Monthly income & debt ratio
        - Credit utilization & open credit lines
        - Payment history & late payments
        - Age & number of dependents

        **Higher score = Better financial health**
        """)
    with col_b:
        st.markdown("""
        ### 📊 Score Interpretation
        | Score Range | Status |
        |---|---|
        | 🟢 70 – 100 | Excellent |
        | 🟡 45 – 70  | Moderate  |
        | 🔴  0 – 45  | At Risk   |

        Built on **150,000 real records** using
        **Random Forest Regression** with feature engineering.
        """)

    st.markdown("---")
    st.pyplot(draw_feature_importance(), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    "Financial Health Score Predictor | Applied Business Analytics Project | "
    "Data: Kaggle Give Me Some Credit Dataset</p>",
    unsafe_allow_html=True
)
