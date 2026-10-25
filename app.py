from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "customer_chrun.joblib"


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(99, 102, 241, 0.17), transparent 28%),
            radial-gradient(circle at 95% 10%, rgba(168, 85, 247, 0.12), transparent 24%),
            #0b1020;
        color: #eef2ff;
    }
    [data-testid="stHeader"] { background: rgba(11, 16, 32, 0.85); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11182b 0%, #0c1222 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: #f8fafc !important;
    }
    p, label, span, div { }
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 22px;
        background: linear-gradient(120deg, rgba(79, 70, 229, .22), rgba(30, 41, 59, .55));
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        color: #c4b5fd;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        font-weight: 800;
        line-height: 1.15;
        margin: .45rem 0 .6rem;
        color: #ffffff;
    }
    .hero-copy { color: #cbd5e1; font-size: 1rem; max-width: 800px; }
    .metric-card {
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 16px;
        background: rgba(15, 23, 42, .76);
        padding: 1rem 1.1rem;
        min-height: 100px;
    }
    .metric-label { color: #94a3b8; font-size: .82rem; }
    .metric-value { color: #f8fafc; font-size: 1.55rem; font-weight: 800; margin-top: .25rem; }
    .section-card {
        border: 1px solid rgba(148, 163, 184, .16);
        background: rgba(15, 23, 42, .58);
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        margin: .4rem 0 1rem;
    }
    .muted { color: #a5b4fc; font-size: .9rem; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%;
        border: 0;
        border-radius: 12px;
        min-height: 3rem;
        color: white;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        border: 1px solid #c4b5fd;
        color: white;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
    }
    div[data-baseweb="input"] input, div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] textarea {
        background-color: #111827;
        color: #f8fafc;
        border-color: #334155;
    }
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, .7);
        border: 1px solid rgba(148, 163, 184, .18);
        border-radius: 14px;
        padding: 1rem;
    }
    .result-box {
        border-radius: 18px;
        padding: 1.25rem;
        border: 1px solid rgba(167, 139, 250, .3);
        background: rgba(30, 41, 59, .72);
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load trained preprocessing + model pipeline
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Customer analytics · Predictive intelligence</div>
        <div class="hero-title">Customer Churn Intelligence</div>
        <div class="hero-copy">
            Estimate the likelihood that a bank customer may leave. Enter the customer's
            profile and relationship activity to receive a model prediction and a
            probability-based risk estimate.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_cols = st.columns(3)
metric_data = [
    ("MODEL", "Gradient Boosting"),
    ("TASK", "Binary classification"),
    ("OUTPUT", "Churn risk estimate"),
]
for col, (label, value) in zip(metric_cols, metric_data):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 Churn dashboard")
    st.caption("Gradient Boosting model")
    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        "1. Enter customer information.\n"
        "2. Click **Predict churn risk**.\n"
        "3. Review the predicted class and estimated probability."
    )
    st.markdown("---")
    st.markdown("### About this app")
    st.caption(
        "This app expects a saved scikit-learn pipeline that includes both "
        "preprocessing and the trained classifier."
    )
    st.caption("Model file: `customer_chrun.joblib`")


if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. "
        "Upload `customer_chrun.joblib` to the same GitHub folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The model could not be loaded. Make sure `customer_chrun.joblib` is the "
        "trained scikit-learn pipeline and that your deployed scikit-learn version "
        "matches the version used during training."
    )
    st.exception(exc)
    st.stop()


# ---------------------------------------------------------
# Customer input form
# Feature names must match the columns used when training.
# CLIENTNUM and target/leakage columns are intentionally omitted.
# ---------------------------------------------------------
st.markdown("## Customer profile")
st.markdown(
    '<div class="muted">Complete the fields below. Values are used only to generate '
    'a prediction from your saved model.</div>',
    unsafe_allow_html=True,
)

with st.form("customer_churn_form"):
    st.markdown("### 01 · Personal details")
    p1, p2, p3 = st.columns(3)
    with p1:
        customer_age = st.number_input("Customer age", min_value=18, max_value=100, value=45, step=1)
        gender = st.selectbox("Gender", ["M", "F"])
    with p2:
        dependent_count = st.number_input("Dependent count", min_value=0, max_value=10, value=2, step=1)
        education_level = st.selectbox(
            "Education level",
            ["Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate", "Unknown"],
        )
    with p3:
        marital_status = st.selectbox("Marital status", ["Married", "Single", "Divorced", "Unknown"])
        income_category = st.selectbox(
            "Income category",
            ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"],
        )

    card_category = st.selectbox("Card category", ["Blue", "Silver", "Gold", "Platinum"])

    st.markdown("### 02 · Account relationship")
    a1, a2, a3 = st.columns(3)
    with a1:
        months_on_book = st.number_input("Months on book", min_value=0, max_value=100, value=36, step=1)
        total_relationship_count = st.number_input(
            "Total relationship products", min_value=1, max_value=10, value=4, step=1
        )
    with a2:
        months_inactive_12_mon = st.number_input(
            "Inactive months (last 12)", min_value=0, max_value=12, value=2, step=1
        )
        contacts_count_12_mon = st.number_input(
            "Contacts (last 12 months)", min_value=0, max_value=20, value=2, step=1
        )
    with a3:
        credit_limit = st.number_input("Credit limit", min_value=0.0, max_value=1_000_000.0, value=8000.0, step=500.0)
        total_revolving_bal = st.number_input(
            "Total revolving balance", min_value=0.0, max_value=1_000_000.0, value=1200.0, step=100.0
        )

    st.markdown("### 03 · Transaction activity")
    t1, t2, t3 = st.columns(3)
    with t1:
        avg_open_to_buy = st.number_input(
            "Average open-to-buy", min_value=0.0, max_value=1_000_000.0, value=6800.0, step=500.0
        )
        total_amt_chng = st.number_input(
            "Amount change (Q4/Q1)", min_value=0.0, max_value=10.0, value=0.7, step=0.05, format="%.3f"
        )
    with t2:
        total_trans_amt = st.number_input(
            "Total transaction amount", min_value=0.0, max_value=1_000_000.0, value=4000.0, step=250.0
        )
        total_trans_ct = st.number_input(
            "Total transaction count", min_value=0, max_value=1000, value=65, step=1
        )
    with t3:
        total_ct_chng = st.number_input(
            "Transaction count change (Q4/Q1)", min_value=0.0, max_value=10.0, value=0.7, step=0.05, format="%.3f"
        )
        avg_utilization = st.number_input(
            "Average utilization ratio", min_value=0.0, max_value=1.0, value=0.25, step=0.05, format="%.3f"
        )

    submitted = st.form_submit_button("✨ Predict churn risk")


if submitted:
    # Keep the original training column names and order.
    input_data = pd.DataFrame(
        [
            {
                "Customer_Age": customer_age,
                "Gender": gender,
                "Dependent_count": dependent_count,
                "Education_Level": education_level,
                "Marital_Status": marital_status,
                "Income_Category": income_category,
                "Card_Category": card_category,
                "Months_on_book": months_on_book,
                "Total_Relationship_Count": total_relationship_count,
                "Months_Inactive_12_mon": months_inactive_12_mon,
                "Contacts_Count_12_mon": contacts_count_12_mon,
                "Credit_Limit": credit_limit,
                "Total_Revolving_Bal": total_revolving_bal,
                "Avg_Open_To_Buy": avg_open_to_buy,
                "Total_Amt_Chng_Q4_Q1": total_amt_chng,
                "Total_Trans_Amt": total_trans_amt,
                "Total_Trans_Ct": total_trans_ct,
                "Total_Ct_Chng_Q4_Q1": total_ct_chng,
                "Avg_Utilization_Ratio": avg_utilization,
            }
        ]
    )

    try:
        prediction = int(model.predict(input_data)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            # The target was encoded as 0 = Existing Customer, 1 = Attrited Customer.
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])
            else:
                probability = float(probabilities[-1])

        st.markdown("---")
        st.markdown("## Prediction result")

        result_col, probability_col = st.columns([1.15, 1])
        with result_col:
            if prediction == 1:
                st.error("⚠️ **Predicted class: Attrited Customer**")
                st.write("The model predicts that this customer belongs to the attrited-customer class.")
            else:
                st.success("✅ **Predicted class: Existing Customer**")
                st.write("The model predicts that this customer belongs to the existing-customer class.")

        with probability_col:
            if probability is not None:
                st.metric("Estimated churn probability", f"{probability:.1%}")
                st.progress(max(0.0, min(1.0, probability)))
                if probability >= 0.70:
                    st.warning("Higher estimated churn probability")
                elif probability >= 0.40:
                    st.info("Moderate estimated churn probability")
                else:
                    st.success("Lower estimated churn probability")
            else:
                st.info("This saved model does not provide `predict_proba`; showing the predicted class only.")

        st.caption(
            "This is a machine-learning estimate, not a guarantee that a customer will leave. "
            "Use it alongside business context and appropriate customer-service practices."
        )

        download_data = input_data.copy()
        download_data["Predicted_Attrition_Flag"] = prediction
        download_data["Predicted_Class"] = (
            "Attrited Customer" if prediction == 1 else "Existing Customer"
        )
        if probability is not None:
            download_data["Estimated_Churn_Probability"] = probability

        st.download_button(
            label="⬇️ Download prediction as CSV",
            data=download_data.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_prediction.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(
            "Prediction failed. The input column names and categories must match the "
            "data used to train the model. Check the feature list and the saved pipeline."
        )
        st.exception(exc)


st.markdown("---")
st.caption("Customer Churn Intelligence · Built with Streamlit and scikit-learn")
