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
