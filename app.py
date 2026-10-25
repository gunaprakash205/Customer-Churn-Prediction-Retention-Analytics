
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="AttritionIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "customer_attrition_gb_pipeline.joblib"


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.name}. "
            "Place the model file in the same folder as app.py."
        )
    return joblib.load(MODEL_PATH)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #172554 0%, transparent 35%),
        radial-gradient(ellipse at 95% 5%, #312e81 0%, transparent 30%),
        #080d1a;
    color: #f1f5f9;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #0c1324;
    border-right: 1px solid #26314a;
}

[data-testid="stSidebar"] * {
    color: #dbe4f0;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    padding: 30px 32px;
    border-radius: 22px;
    border: 1px solid #33456b;
    background: linear-gradient(120deg, #172554 0%, #312e81 55%, #4338ca 100%);
    margin-bottom: 25px;
}

.hero-tag {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 20px;
    background: #ffffff18;
    border: 1px solid #ffffff30;
    color: #ddd6fe;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    color: white;
    font-size: clamp(28px, 4vw, 43px);
    font-weight: 800;
    margin: 15px 0 8px 0;
}

.hero p {
    color: #dbeafe;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 0;
}

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #f8fafc;
    margin: 12px 0 5px 0;
}

.section-subtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 18px;
}

.panel {
    padding: 22px;
    border: 1px solid #263650;
    border-radius: 18px;
    background: #111a2b;
    margin-bottom: 16px;
}

.metric-card {
    padding: 20px;
    background: linear-gradient(145deg, #131f35, #101728);
    border: 1px solid #2a3a56;
    border-radius: 16px;
    min-height: 112px;
}

.metric-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .8px;
}

.metric-value {
    font-size: 25px;
    font-weight: 800;
    color: #f8fafc;
    margin-top: 10px;
}

.metric-note {
    color: #a5b4fc;
    font-size: 12px;
    margin-top: 5px;
}

.result-box {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(135deg, #172554, #1e1b4b);
    border: 1px solid #4f46e5;
    margin-top: 15px;
}

.result-heading {
    color: #c4b5fd;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}

.result-value {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin: 8px 0;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 700;
    min-height: 46px;
    transition: .2s ease;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    border: 1px solid #c4b5fd;
    color: white;
    filter: brightness(1.12);
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #dbe4f0 !important;
    font-size: 13px;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background-color: #101a2c;
    color: white;
    border-color: #34445f;
    border-radius: 9px;
}

div[data-testid="stForm"] {
    background: #111a2b;
    border: 1px solid #263650;
    padding: 24px;
    border-radius: 18px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.footer {
    color: #64748b;
    text-align: center;
    padding: 30px 0 5px 0;
    font-size: 12px;
}

hr {
    border-color: #263650;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 📊 AttritionIQ")
    st.caption("Customer Intelligence Platform")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠  Prediction Dashboard")
    st.markdown("👤  Customer Profile")
    st.markdown("📈  Risk Analysis")

    st.divider()
    st.markdown("### Model Information")
    st.markdown("**Algorithm:** Gradient Boosting")
    st.markdown("**Task:** Binary Classification")
    st.markdown("**Target:** Customer Attrition")

    st.divider()
    st.caption(
        "Built with Python, Scikit-learn and Streamlit."
    )


# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <span class="hero-tag">MACHINE LEARNING • CUSTOMER ANALYTICS</span>
    <h1>Customer Attrition Intelligence</h1>
    <p>
        Predict customer attrition risk using your trained Gradient Boosting model.
        Explore customer profiles, estimate risk and support data-informed
        retention decisions.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL WITH FRIENDLY ERROR HANDLING
# =========================================================
try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Ensure customer_attrition_gb_pipeline.joblib is uploaded "
        "beside app.py and that the required packages are installed."
    )
    st.stop()


# =========================================================
# DASHBOARD METRICS
# =========================================================
m1, m2, m3, m4 = st.columns(4)

metrics = [
    (m1, "MODEL", "Gradient Boosting", "Tuned classifier"),
    (m2, "TEST ACCURACY", "96.79%", "Your reported test result"),
    (m3, "TEST RECALL", "82.77%", "Attrited customer class"),
    (m4, "TEST ROC-AUC", "99.05%", "Your reported test result"),
]

for column, label, value, note in metrics:
    with column:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """, unsafe_allow_html=True)

st.caption(
    "Metrics above are from your notebook's test evaluation; "
    "they are not calculated from the customer entered below."
)

st.markdown("")
st.markdown('<div class="section-title">👤 Customer Profile</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Enter the customer details below '
    'to generate a prediction.</div>',
    unsafe_allow_html=True
)


# =========================================================
# CUSTOMER INPUT FORM
# Keep the feature names exactly as used during training.
# =========================================================
with st.form("customer_prediction_form"):

    st.markdown("#### 01 — Personal Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Customer age", 18, 100, 45
        )
        gender = st.selectbox("Gender", ["M", "F"])
        dependents = st.number_input(
            "Number of dependents", 0, 10, 2
        )

    with c2:
        education = st.selectbox(
            "Education level",
            [
                "Unknown", "Uneducated", "High School", "College",
                "Graduate", "Post-Graduate", "Doctorate"
            ]
        )
        marital = st.selectbox(
            "Marital status",
            ["Unknown", "Single", "Married", "Divorced"]
        )

    with c3:
        income = st.selectbox(
            "Income category",
            [
                "Unknown", "Less than $40K", "$40K - $60K",
                "$60K - $80K", "$80K - $120K", "$120K +"
            ]
        )
        card = st.selectbox(
            "Card category", ["Blue", "Silver", "Gold", "Platinum"]
        )

    st.divider()
    st.markdown("#### 02 — Account Relationship")

    a1, a2, a3 = st.columns(3)

    with a1:
        months_on_book = st.number_input(
            "Months on book", 0, 100, 36
        )
        relationships = st.number_input(
            "Total relationship count", 0, 10, 4
        )

    with a2:
        inactive = st.number_input(
            "Months inactive in last 12 months", 0, 12, 2
        )
        contacts = st.number_input(
            "Contacts in last 12 months", 0, 20, 2
        )

    with a3:
        credit_limit = st.number_input(
            "Credit limit ($)", min_value=0.0,
            value=8000.0, step=500.0
        )
        revolving = st.number_input(
            "Total revolving balance ($)", min_value=0.0,
            value=1200.0, step=100.0
        )
        open_to_buy = st.number_input(
            "Average open to buy ($)", min_value=0.0,
            value=6800.0, step=100.0
        )

    st.divider()
    st.markdown("#### 03 — Transaction Behaviour")

    t1, t2, t3 = st.columns(3)

    with t1:
        amount_change = st.number_input(
            "Transaction amount change (Q4/Q1)",
            min_value=0.0, value=0.7, step=0.1,
            format="%.4f"
        )
        transaction_amount = st.number_input(
            "Total transaction amount ($)",
            min_value=0.0, value=3000.0, step=100.0
        )

    with t2:
        transaction_count = st.number_input(
            "Total transaction count",
            min_value=0, value=45, step=1
        )
        count_change = st.number_input(
            "Transaction count change (Q4/Q1)",
            min_value=0.0, value=0.6, step=0.1,
            format="%.4f"
        )

    with t3:
        utilization = st.number_input(
            "Average utilization ratio",
            min_value=0.0, max_value=1.0,
            value=0.2, step=0.05, format="%.2f"
        )
        st.caption(
            "Utilization ratio ranges from 0 to 1."
        )

    st.divider()

    submitted = st.form_submit_button(
        "✨  Analyze Customer Attrition Risk",
        use_container_width=True
    )


# =========================================================
# PREDICTION RESULTS
# =========================================================
if submitted:

    customer = pd.DataFrame([{
        "Customer_Age": age,
        "Gender": gender,
        "Dependent_count": dependents,
        "Education_Level": education,
        "Marital_Status": marital,
        "Income_Category": income,
        "Card_Category": card,
        "Months_on_book": months_on_book,
        "Total_Relationship_Count": relationships,
        "Months_Inactive_12_mon": inactive,
        "Contacts_Count_12_mon": contacts,
        "Credit_Limit": credit_limit,
        "Total_Revolving_Bal": revolving,
        "Avg_Open_To_Buy": open_to_buy,
        "Total_Amt_Chng_Q4_Q1": amount_change,
        "Total_Trans_Amt": transaction_amount,
        "Total_Trans_Ct": transaction_count,
        "Total_Ct_Chng_Q4_Q1": count_change,
        "Avg_Utilization_Ratio": utilization
    }])

    try:
        prediction = int(model.predict(customer)[0])
        probabilities = model.predict_proba(customer)[0]

        # Assumes class 1 represents Attrited Customer.
        classes = list(model.classes_)
        positive_index = classes.index(1)
        probability = float(probabilities[positive_index])

        st.divider()
        st.markdown(
            '<div class="section-title">📈 Prediction Results</div>',
            unsafe_allow_html=True
        )

        r1, r2 = st.columns([1.15, 1])

        with r1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Predicted Customer Status**")

            if prediction == 1:
                st.error("⚠️ Attrited Customer")
                st.markdown(
                    "The model classifies this customer as belonging "
                    "to the attrited class."
                )
            else:
                st.success("✓ Existing Customer")
                st.markdown(
                    "The model classifies this customer as belonging "
                    "to the existing customer class."
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="result-box">
                <div class="result-heading">Estimated Attrition Probability</div>
                <div class="result-value">{probability:.1%}</div>
                <div style="color:#cbd5e1;font-size:13px;">
                    Model probability for class 1
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(probability)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Customer Summary**")

        summary1, summary2, summary3 = st.columns(3)
        summary1.metric("Age", f"{age} years")
        summary2.metric("Months on book", str(months_on_book))
        summary3.metric("Transactions", str(transaction_count))

        summary4, summary5, summary6 = st.columns(3)
        summary4.metric("Credit limit", f"${credit_limit:,.0f}")
        summary5.metric("Inactive months", str(inactive))
        summary6.metric("Utilization", f"{utilization:.0%}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Download the entered profile and its model output.
        result_download = customer.copy()
        result_download["Predicted_Class"] = prediction
        result_download["Attrition_Probability"] = probability
        result_download["Predicted_Status"] = (
            "Attrited Customer" if prediction == 1
            else "Existing Customer"
        )

        st.download_button(
            label="⬇️ Download Prediction Report (CSV)",
            data=result_download.to_csv(index=False).encode("utf-8"),
            file_name="customer_attrition_prediction.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.caption(
            "The prediction uses the model's default classification threshold. "
            "Probabilities may not be calibrated and should not be treated "
            "as guarantees. Validate the application before making real "
            "customer or financial decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that your saved model was trained with the same 19 "
            "feature columns and that the app uses compatible package versions."
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    AttritionIQ · Customer Attrition Prediction ·
    Powered by Gradient Boosting and Streamlit
</div>
""", unsafe_allow_html=True)
