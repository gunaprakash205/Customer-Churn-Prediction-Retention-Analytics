
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

MODEL_PATH = Path(__file__).parent / "customer_chrun.joblib"


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
