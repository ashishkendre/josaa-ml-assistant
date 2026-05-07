"""
JoSAA ML-Powered Choice Filling Assistant
Premium Edition — Sophisticated UI with refined typography and color palette
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="JoSAA Intelligence Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM DESIGN SYSTEM
# Color Palette: Deep navy (#0A1929) primary, Champagne gold (#C9A961) accent,
# Off-white (#FAFAF7) background, Slate (#475569) supporting
# Typography: Playfair Display (display), Inter (body), JetBrains Mono (code)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    /* ═══════════ ROOT VARIABLES ═══════════ */
    :root {
        --navy-deep: #0A1929;
        --navy-medium: #142B47;
        --navy-light: #1E3A5F;
        --gold: #C9A961;
        --gold-light: #E8D5A8;
        --gold-dark: #9C7E3F;
        --cream: #FAFAF7;
        --paper: #FFFFFF;
        --slate: #475569;
        --slate-light: #64748B;
        --slate-dark: #1E293B;
        --success: #2D6A4F;
        --warning: #B7791F;
        --danger: #9B2C2C;
        --hairline: #D1D5DB;
    }

    /* ═══════════ GLOBAL ═══════════ */
    .stApp {
        background: linear-gradient(180deg, #0A1929 0%, #142B47 100%);
        background-attachment: fixed;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        letter-spacing: -0.011em;
    }

    /* ═══════════ MAIN CONTAINER ═══════════ */
    .main .block-container {
        background: var(--cream);
        padding: 3rem 3.5rem 4rem;
        border-radius: 4px;
        box-shadow:
            0 1px 2px rgba(0,0,0,0.04),
            0 8px 24px rgba(10,25,41,0.12),
            0 32px 64px rgba(10,25,41,0.18);
        max-width: 1400px;
        margin-top: 2rem;
        margin-bottom: 2rem;
        position: relative;
    }

    .main .block-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg,
            transparent 0%,
            var(--gold) 25%,
            var(--gold-light) 50%,
            var(--gold) 75%,
            transparent 100%);
    }

    /* ═══════════ TYPOGRAPHY ═══════════ */
    h1 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        font-size: 3.25rem !important;
        line-height: 1.1 !important;
        color: var(--navy-deep) !important;
        letter-spacing: -0.02em !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 600 !important;
        color: var(--navy-deep) !important;
        font-size: 2rem !important;
        letter-spacing: -0.015em !important;
    }

    h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: var(--navy-medium) !important;
        font-size: 1.25rem !important;
        letter-spacing: -0.01em !important;
    }

    h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: var(--slate) !important;
        font-size: 1rem !important;
    }

    p, .stMarkdown {
        font-family: 'Inter', sans-serif !important;
        color: var(--slate-dark) !important;
        line-height: 1.65 !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 0.875rem !important;
    }

    /* ═══════════ HERO HEADER ═══════════ */
    .hero-container {
        margin: -1rem -1rem 3rem -1rem;
        padding: 2.5rem 0 2rem;
        border-bottom: 1px solid var(--hairline);
        position: relative;
    }

    .hero-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.25em;
        color: var(--gold-dark);
        text-transform: uppercase;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .hero-eyebrow::before {
        content: '';
        width: 32px;
        height: 1px;
        background: var(--gold);
    }

    .hero-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1.05;
        color: var(--navy-deep);
        letter-spacing: -0.025em;
        margin-bottom: 0.75rem;
    }

    .hero-title .accent {
        font-style: italic;
        color: var(--gold-dark);
        font-weight: 500;
    }

    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.0625rem;
        font-weight: 400;
        color: var(--slate);
        line-height: 1.55;
        max-width: 720px;
        letter-spacing: -0.005em;
    }

    .hero-meta {
        display: flex;
        gap: 2.5rem;
        margin-top: 1.75rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--hairline);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--slate-light);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        flex-wrap: wrap;
    }

    .hero-meta strong {
        color: var(--navy-deep);
        font-weight: 600;
    }

    /* ═══════════ SIDEBAR ═══════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1929 0%, #142B47 100%) !important;
        border-right: 1px solid rgba(201,169,97,0.15);
    }

    [data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--cream) !important;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: var(--cream) !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(201,169,97,0.2);
        letter-spacing: -0.01em;
    }

    [data-testid="stSidebar"] label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: var(--gold-light) !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div,
    [data-testid="stSidebar"] input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(201,169,97,0.25) !important;
        border-radius: 2px !important;
        color: var(--cream) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] .stSelectbox > div > div:hover,
    [data-testid="stSidebar"] input:hover {
        border-color: var(--gold) !important;
        background: rgba(255,255,255,0.09) !important;
    }

    /* Selectbox dropdown popover - light background, so dark text needed */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] ul {
        background: var(--paper) !important;
        border: 1px solid var(--hairline) !important;
    }

    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] li {
        color: var(--navy-deep) !important;
        background: var(--paper) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] li:hover {
        background: rgba(201,169,97,0.1) !important;
        color: var(--navy-deep) !important;
    }

    [data-baseweb="popover"] [aria-selected="true"] {
        background: var(--navy-deep) !important;
        color: var(--cream) !important;
    }

    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--gold) !important;
        border: 2px solid var(--cream) !important;
        box-shadow: 0 2px 8px rgba(201,169,97,0.4) !important;
    }

    .sidebar-info-card {
        background: rgba(201,169,97,0.08);
        border: 1px solid rgba(201,169,97,0.2);
        border-left: 3px solid var(--gold);
        padding: 1.25rem;
        margin-top: 1.5rem;
        border-radius: 2px;
    }

    .sidebar-info-card-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--gold) !important;
        margin-bottom: 0.75rem;
    }

    .sidebar-info-card-list {
        font-size: 0.8125rem;
        line-height: 1.7;
        color: rgba(250,250,247,0.85) !important;
    }

    /* ═══════════ BUTTON ═══════════ */
    .stButton > button {
        background: var(--navy-deep) !important;
        color: var(--cream) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        border-radius: 2px !important;
        padding: 0.875rem 2rem !important;
        border: 1px solid var(--gold) !important;
        width: 100% !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stButton > button:hover {
        background: var(--gold) !important;
        color: var(--navy-deep) !important;
        border-color: var(--gold) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(201,169,97,0.3) !important;
    }

    /* ═══════════ TABS ═══════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 2px solid var(--hairline);
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--slate) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 3px solid transparent !important;
        padding: 1rem 2rem !important;
        margin: 0 !important;
        margin-bottom: -2px !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [data-baseweb="tab"] p {
        color: var(--slate) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--navy-deep) !important;
        background: rgba(201,169,97,0.05) !important;
    }

    .stTabs [data-baseweb="tab"]:hover p {
        color: var(--navy-deep) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--navy-deep) !important;
        border-bottom-color: var(--gold-dark) !important;
        font-weight: 700 !important;
    }

    .stTabs [aria-selected="true"] p {
        color: var(--navy-deep) !important;
        font-weight: 700 !important;
    }

    /* ═══════════ METRICS ═══════════ */
    [data-testid="stMetric"] {
        background: var(--paper);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 1.5rem 1.5rem 1.25rem;
        position: relative;
        transition: all 0.25s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--gold);
        box-shadow: 0 4px 16px rgba(10,25,41,0.06);
        transform: translateY(-1px);
    }

    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 32px;
        height: 2px;
        background: var(--gold);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.6875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.18em !important;
        color: var(--slate) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: var(--navy-deep) !important;
        line-height: 1.1 !important;
        margin-top: 0.5rem !important;
    }

    /* ═══════════ RECOMMENDATION CARDS ═══════════ */
    .premium-card {
        background: var(--paper);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        position: relative;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }

    .premium-card::before {
        content: '';
        position: absolute;
        top: 0; bottom: 0;
        left: 0;
        width: 3px;
        background: var(--gold);
        transform: scaleY(0);
        transform-origin: top;
        transition: transform 0.3s ease;
    }

    .premium-card:hover {
        border-color: rgba(201,169,97,0.4);
        box-shadow: 0 12px 40px rgba(10,25,41,0.08);
        transform: translateY(-2px);
    }

    .premium-card:hover::before {
        transform: scaleY(1);
    }

    .card-rank {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 0.875rem;
        color: var(--gold-dark);
        font-weight: 500;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        font-style: italic;
    }

    .card-institute {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--navy-deep);
        line-height: 1.25;
        margin-bottom: 0.5rem;
        letter-spacing: -0.015em;
    }

    .card-program {
        font-family: 'Inter', sans-serif;
        font-size: 0.9375rem;
        font-weight: 500;
        color: var(--slate);
        margin-bottom: 1.25rem;
    }

    .card-divider {
        height: 1px;
        background: var(--hairline);
        margin: 1.25rem 0;
    }

    .card-stats {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
    }

    .card-stat-label {
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.65rem;
        color: var(--slate-light);
        margin-bottom: 0.25rem;
        display: block;
        font-family: 'JetBrains Mono', monospace;
    }

    .card-stat-value {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--navy-deep);
        font-size: 0.95rem;
    }

    .card-badges {
        display: flex;
        gap: 0.625rem;
        flex-wrap: wrap;
        margin-top: 1.25rem;
        align-items: center;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.4rem 0.875rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        border-radius: 2px;
        border: 1px solid;
    }

    .pill-nirf {
        background: var(--navy-deep);
        color: var(--cream);
        border-color: var(--navy-deep);
    }

    .pill-high {
        background: rgba(45,106,79,0.08);
        color: var(--success);
        border-color: rgba(45,106,79,0.3);
    }

    .pill-moderate {
        background: rgba(183,121,31,0.08);
        color: var(--warning);
        border-color: rgba(183,121,31,0.3);
    }

    .pill-low {
        background: rgba(155,44,44,0.08);
        color: var(--danger);
        border-color: rgba(155,44,44,0.3);
    }

    .card-prediction {
        background: linear-gradient(135deg, rgba(10,25,41,0.03), rgba(201,169,97,0.05));
        border-left: 2px solid var(--gold);
        padding: 1rem 1.25rem;
        margin-top: 1.25rem;
        font-size: 0.875rem;
    }

    .card-prediction-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: var(--gold-dark);
        margin-bottom: 0.375rem;
    }

    .card-prediction-value {
        font-family: 'Inter', sans-serif;
        color: var(--navy-deep);
        font-weight: 500;
    }

    /* ═══════════ SECTION HEADER ═══════════ */
    .section-header {
        margin: 2.5rem 0 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--hairline);
    }

    .section-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--gold-dark);
        margin-bottom: 0.5rem;
    }

    .section-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.875rem;
        font-weight: 600;
        color: var(--navy-deep);
        letter-spacing: -0.015em;
    }

    /* ═══════════ EXPANDER ═══════════ */
    .streamlit-expanderHeader {
        background: var(--paper) !important;
        border: 1px solid var(--hairline) !important;
        border-radius: 2px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: var(--navy-deep) !important;
        padding: 1rem 1.25rem !important;
    }

    /* ═══════════ DATAFRAME ═══════════ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--hairline);
        border-radius: 2px;
        overflow: hidden;
    }

    /* ═══════════ ALERTS ═══════════ */
    .stAlert {
        border-radius: 2px !important;
        border-left-width: 3px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ═══════════ FOOTER ═══════════ */
    .premium-footer {
        margin-top: 4rem;
        padding-top: 2.5rem;
        border-top: 1px solid var(--hairline);
        text-align: center;
    }

    .footer-mark {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--navy-deep);
        margin-bottom: 0.5rem;
    }

    .footer-mark .accent {
        color: var(--gold-dark);
        font-style: italic;
    }

    .footer-tagline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 400;
        color: var(--slate-light);
        text-transform: uppercase;
        letter-spacing: 0.25em;
        margin-bottom: 1.5rem;
    }

    .footer-meta {
        font-family: 'Inter', sans-serif;
        font-size: 0.8125rem;
        color: var(--slate);
        line-height: 1.7;
    }

    .footer-meta strong {
        color: var(--navy-deep);
        font-weight: 600;
    }

    /* ═══════════ SETUP SCREEN ═══════════ */
    .setup-screen {
        background: var(--paper);
        border: 1px solid var(--hairline);
        border-radius: 2px;
        padding: 3rem;
        margin-top: 2rem;
        position: relative;
    }

    .setup-screen::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        height: 3px;
        width: 100%;
        background: linear-gradient(90deg, var(--gold), var(--gold-light), var(--gold));
    }

    /* ═══════════ EMPTY STATE ═══════════ */
    .empty-state {
        padding: 4rem 2rem;
        text-align: center;
        background: var(--paper);
        border: 1px solid var(--hairline);
        border-radius: 2px;
    }

    .empty-state-title {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 1.5rem;
        color: var(--navy-deep);
        font-weight: 600;
        margin-bottom: 0.75rem;
        letter-spacing: -0.01em;
    }

    .empty-state-text {
        font-family: 'Inter', sans-serif;
        color: var(--slate);
        font-size: 0.95rem;
        line-height: 1.65;
        max-width: 480px;
        margin: 0 auto;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--cream); }
    ::-webkit-scrollbar-thumb { background: var(--slate-light); border-radius: 0; }
    ::-webkit-scrollbar-thumb:hover { background: var(--slate); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    paths_to_try = [
        ('models/cutoff_prediction_model.pkl',  'models/admission_probability_model.pkl'),
        ('./models/cutoff_prediction_model.pkl', './models/admission_probability_model.pkl'),
        ('cutoff_prediction_model.pkl',          'admission_probability_model.pkl'),
        ('./cutoff_prediction_model.pkl',        './admission_probability_model.pkl'),
    ]
    for cutoff_path, admission_path in paths_to_try:
        if Path(cutoff_path).exists() and Path(admission_path).exists():
            try:
                cutoff_model = joblib.load(cutoff_path)
                admission_model = joblib.load(admission_path)
                return cutoff_model, admission_model
            except Exception as e:
                st.warning(f"Found files at {cutoff_path} but failed to load: {e}")
    return None, None


@st.cache_data
def load_data():
    paths_to_try = [
        'processed_data/josaa_ml_ready.csv',
        'processed_data/josaa_ml_ready_small.csv',
        './processed_data/josaa_ml_ready.csv',
        './processed_data/josaa_ml_ready_small.csv',
        'josaa_ml_ready.csv',
        'josaa_ml_ready_small.csv',
        './josaa_ml_ready.csv',
        './josaa_ml_ready_small.csv',
    ]
    for path in paths_to_try:
        if Path(path).exists():
            try:
                return pd.read_csv(path)
            except:
                continue
    return None


@st.cache_data
def load_predictions_2025():
    paths_to_try = [
        'results/predictions_2025.csv',
        './results/predictions_2025.csv',
        'predictions_2025.csv',
        './predictions_2025.csv',
    ]
    for path in paths_to_try:
        if Path(path).exists():
            try:
                return pd.read_csv(path)
            except:
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ML INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def get_admission_probability(student_rank, college_data, admission_model_data):
    model = admission_model_data['model']
    feature_cols = admission_model_data['feature_columns']

    features = {
        'student_rank': student_rank,
        'prev_year_closing_rank': college_data.get('Closing Rank', 0),
        'prev_year_opening_rank': college_data.get('Opening Rank', 0),
        'rank_vs_prev_closing': student_rank - college_data.get('Closing Rank', 0),
        'rank_ratio_prev': student_rank / college_data.get('Closing Rank', 1) if college_data.get('Closing Rank', 0) > 0 else 1.0,
        'prev_closing_opening_ratio': college_data.get('Closing Rank', 1) / college_data.get('Opening Rank', 1) if college_data.get('Opening Rank', 0) > 0 else 1.0,
        'year': 2025,
        'round': college_data.get('Round', 1),
        'institute_code': college_data.get('Institute_Code', 0),
        'institute_type_code': college_data.get('Institute_Type_Code', 0),
        'branch_code': college_data.get('Academic Program Name_Code', 0),
        'branch_category_code': college_data.get('Branch_Category_Code', 0),
        'quota_code': college_data.get('Quota_Code', 0),
        'seat_type_code': college_data.get('Seat Type_Code', 0),
        'gender_code': college_data.get('Gender_Code', 0),
        'nirf_rank': college_data.get('NIRF_Rank', 999),
        'round_progression': college_data.get('Round_Progression', 0.5)
    }

    X = pd.DataFrame([features])[feature_cols]
    return model.predict_proba(X)[0][1]


def get_recommendations_ml(df, predictions_df, user_prefs, admission_model_data):
    exam_type = user_prefs['exam_type']

    if exam_type == 'advanced':
        filtered = df[df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)].copy()
    else:
        filtered = df[~df['Institute'].str.contains('Indian Institute of Technology', case=False, na=False)].copy()

    filtered = filtered[filtered['Year'] == 2024]
    filtered = filtered[filtered['Seat Type'].str.contains(user_prefs['category'], case=False, na=False)]
    filtered = filtered[
        (filtered['Gender'].str.contains(user_prefs['gender'], case=False, na=False)) |
        (filtered['Gender'].str.contains('Neutral', case=False, na=False))
    ]

    if len(filtered) == 0:
        return None

    user_rank = user_prefs['rank']
    probabilities = [get_admission_probability(user_rank, row.to_dict(), admission_model_data)
                     for _, row in filtered.iterrows()]
    filtered['Admission_Probability'] = probabilities

    if predictions_df is not None:
        merge_cols = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Round']
        pred_subset = predictions_df[merge_cols + ['Predicted_Closing_Rank_2025', 'Change_Percent']]
        filtered = filtered.merge(pred_subset, on=merge_cols, how='left')

    filtered = filtered[filtered['Admission_Probability'] > 0.05]
    filtered['Quality_Score'] = filtered['NIRF_Rank'].apply(lambda x: max(0, 40 - x * 0.5) if x != 999 else 5)
    filtered['Branch_Quality'] = filtered['Branch_Category'].map({
        'CS/IT': 30, 'ECE/EE': 25, 'Mechanical': 20,
        'Aerospace': 22, 'Chemical': 18, 'Civil': 15,
        'Material': 17, 'Bio': 12, 'Other': 10
    }).fillna(10)
    filtered['Composite_Score'] = (
        filtered['Quality_Score'] +
        filtered['Branch_Quality'] +
        filtered['Admission_Probability'] * 30
    )

    filtered = filtered.sort_values('Composite_Score', ascending=False)
    return filtered.head(user_prefs.get('max_choices', 50))


def categorize_probability(prob):
    if prob >= 0.7:
        return "High Confidence", "pill-high"
    elif prob >= 0.4:
        return "Moderate", "pill-moderate"
    return "Reach", "pill-low"


# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM PLOTLY THEME
# ═══════════════════════════════════════════════════════════════════════════════

PREMIUM_PLOTLY_LAYOUT = dict(
    plot_bgcolor='#FAFAF7',
    paper_bgcolor='#FAFAF7',
    font=dict(family='Inter, sans-serif', color='#1E293B', size=12),
    xaxis=dict(gridcolor='#E5E7EB', linecolor='#94A3B8',
               tickfont=dict(family='JetBrains Mono, monospace', size=10)),
    yaxis=dict(gridcolor='#E5E7EB', linecolor='#94A3B8',
               tickfont=dict(family='JetBrains Mono, monospace', size=10)),
    margin=dict(l=60, r=40, t=60, b=60),
    hoverlabel=dict(bgcolor='#0A1929',
                    font=dict(family='Inter', color='#FAFAF7', size=12),
                    bordercolor='#C9A961'),
)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-container">
    <div class="hero-eyebrow">Bachelor Thesis Project · IIT Delhi · I12</div>
    <div class="hero-title">JoSAA <span class="accent">Intelligence</span> Platform</div>
    <div class="hero-subtitle">
        A machine-learning powered decision support system for engineering admissions counseling.
        Built on five years of historical data, two XGBoost models predict admission probability
        and forecast 2025 closing ranks with rigorous accuracy.
    </div>
    <div class="hero-meta">
        <span><strong>282,415</strong>&nbsp;&nbsp;Records</span>
        <span><strong>130</strong>&nbsp;&nbsp;Institutes</span>
        <span><strong>R²&nbsp;0.85</strong>&nbsp;&nbsp;Cutoff Model</span>
        <span><strong>AUC&nbsp;0.975</strong>&nbsp;&nbsp;Admission Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

with st.spinner("Initializing intelligence platform..."):
    cutoff_model_data, admission_model_data = load_models()
    df = load_data()
    predictions_df = load_predictions_2025()

models_missing = cutoff_model_data is None or admission_model_data is None
data_missing = df is None

if models_missing or data_missing:
    st.markdown('<div class="setup-screen">', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">System Status</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Setup Required</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("Required files are not available in the deployment environment. Please ensure the following artifacts are present in the repository root or designated subfolders.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Required Files**")
        st.code("""cutoff_prediction_model.pkl
admission_probability_model.pkl
josaa_ml_ready_small.csv
predictions_2025.csv""", language="text")
    with col2:
        st.markdown("**Resolution Steps**")
        st.markdown("""
1. Run `python reduce_csv_size.py` locally
2. Upload all four files to GitHub repository
3. Reboot the application from Streamlit Cloud
4. Refresh this page
        """)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# Model info expander
with st.expander("Model Specifications"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Cutoff Prediction Model**
        - Algorithm: `{cutoff_model_data['model_name']}`
        - Features: `{len(cutoff_model_data['feature_columns'])}`
        - Training set: `173,699` records (2020–2023)
        - Test R²: `0.85` · MAE: `2,976`
        """)
    with col2:
        st.markdown(f"""
        **Admission Probability Model**
        - Algorithm: `{admission_model_data['model_name']}`
        - Features: `{len(admission_model_data['feature_columns'])}`
        - Training set: `1,259,174` scenarios
        - Test AUC: `0.975` · Accuracy: `91.3%`
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Configuration")

    exam_type_display = st.selectbox(
        "Examination",
        ["", "JEE Advanced (IITs)", "JEE Main (NITs/IIITs/GFTIs)"]
    )

    rank = st.number_input("Rank", min_value=1, max_value=1000000, value=5000)

    category = st.selectbox(
        "Category",
        ["", "OPEN", "EWS", "OBC-NCL", "SC", "ST"]
    )

    gender = st.selectbox(
        "Gender",
        ["", "Gender-Neutral", "Female"]
    )

    max_choices = st.slider("Recommendations", 10, 100, 30)

    st.markdown('<br>', unsafe_allow_html=True)
    submit_button = st.button("Generate Recommendations")

    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-card-title">Platform Capabilities</div>
        <div class="sidebar-info-card-list">
            • Predicted 2025 closing ranks<br>
            • Quantitative admission probability<br>
            • Multi-criteria recommendation scoring<br>
            • Five-year historical trend analysis
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["Recommendations", "2025 Forecasts", "Historical Analysis"])

# ───────── TAB 1: Recommendations ─────────
with tab1:
    if submit_button:
        if not exam_type_display or not category or not gender:
            st.error("Please complete all required fields in the configuration panel.")
        else:
            exam_type = 'advanced' if 'Advanced' in exam_type_display else 'mains'
            user_prefs = {
                'exam_type': exam_type, 'rank': rank,
                'category': category, 'gender': gender,
                'max_choices': max_choices
            }

            with st.spinner("Computing personalized recommendations..."):
                recommendations = get_recommendations_ml(df, predictions_df, user_prefs, admission_model_data)

            if recommendations is None or len(recommendations) == 0:
                st.warning("No matching institutions found. Please adjust your criteria.")
            else:
                st.markdown("""
                <div class="section-header">
                    <div class="section-eyebrow">Summary</div>
                    <div class="section-title">Recommendation Overview</div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Options", f"{len(recommendations)}")
                with col2:
                    high_prob = (recommendations['Admission_Probability'] >= 0.7).sum()
                    st.metric("High Confidence", f"{high_prob}")
                with col3:
                    moderate = ((recommendations['Admission_Probability'] >= 0.4) &
                                (recommendations['Admission_Probability'] < 0.7)).sum()
                    st.metric("Moderate", f"{moderate}")
                with col4:
                    avg_nirf = recommendations[recommendations['NIRF_Rank'] != 999]['NIRF_Rank'].mean()
                    st.metric("Avg NIRF", f"{avg_nirf:.0f}" if pd.notna(avg_nirf) else "—")

                st.markdown("""
                <div class="section-header">
                    <div class="section-eyebrow">Personalized Selection</div>
                    <div class="section-title">Curated Institutions for You</div>
                </div>
                """, unsafe_allow_html=True)

                for idx, (_, rec) in enumerate(recommendations.iterrows(), 1):
                    prob = rec['Admission_Probability']
                    prob_label, prob_class = categorize_probability(prob)
                    nirf = int(rec['NIRF_Rank']) if rec['NIRF_Rank'] != 999 else None
                    nirf_pill = f'<span class="pill pill-nirf">NIRF #{nirf}</span>' if nirf else ''

                    pred_html = ""
                    if 'Predicted_Closing_Rank_2025' in rec and pd.notna(rec.get('Predicted_Closing_Rank_2025')):
                        change = rec.get('Change_Percent', 0)
                        direction = "↑" if change > 0 else "↓" if change < 0 else "→"
                        pred_html = f"""
                        <div class="card-prediction">
                            <div class="card-prediction-label">2025 Forecast</div>
                            <div class="card-prediction-value">
                                Predicted closing rank: <strong>{int(rec['Predicted_Closing_Rank_2025']):,}</strong>
                                &nbsp;&nbsp;{direction}&nbsp;{change:+.1f}% vs. 2024
                            </div>
                        </div>
                        """

                    st.markdown(f"""
                    <div class="premium-card">
                        <div class="card-rank">No. {idx:02d}</div>
                        <div class="card-institute">{rec['Institute']}</div>
                        <div class="card-program">{rec['Academic Program Name']}</div>
                        <div class="card-divider"></div>
                        <div class="card-stats">
                            <div>
                                <span class="card-stat-label">2024 Closing Rank</span>
                                <span class="card-stat-value">{int(rec['Closing Rank']):,}</span>
                            </div>
                            <div>
                                <span class="card-stat-label">Round</span>
                                <span class="card-stat-value">{int(rec['Round'])}</span>
                            </div>
                            <div>
                                <span class="card-stat-label">Quota</span>
                                <span class="card-stat-value">{rec['Quota']}</span>
                            </div>
                            <div>
                                <span class="card-stat-label">Composite Score</span>
                                <span class="card-stat-value">{rec['Composite_Score']:.1f}</span>
                            </div>
                        </div>
                        <div class="card-badges">
                            {nirf_pill}
                            <span class="pill {prob_class}">{prob_label} · {prob*100:.1f}%</span>
                        </div>
                        {pred_html}
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-title">Begin Your Analysis</div>
            <div class="empty-state-text">
                Configure your parameters in the panel to the left and generate personalized,
                machine-learning powered recommendations across 130 engineering institutions.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ───────── TAB 2: 2025 Forecasts ─────────
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">Predictive Analytics</div>
        <div class="section-title">2025 Closing Rank Forecasts</div>
    </div>
    """, unsafe_allow_html=True)

    if predictions_df is not None:
        col1, col2 = st.columns(2)
        with col1:
            filter_institute = st.selectbox(
                "Institute Filter",
                ["All Institutes"] + sorted(predictions_df['Institute'].unique().tolist())
            )
        with col2:
            filter_seat = st.selectbox(
                "Seat Type Filter",
                ["All Categories"] + sorted(predictions_df['Seat Type'].unique().tolist())
            )

        filtered_pred = predictions_df.copy()
        if filter_institute != "All Institutes":
            filtered_pred = filtered_pred[filtered_pred['Institute'] == filter_institute]
        if filter_seat != "All Categories":
            filtered_pred = filtered_pred[filtered_pred['Seat Type'] == filter_seat]

        if len(filtered_pred) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Forecasts", f"{len(filtered_pred):,}")
            with col2:
                avg_change = filtered_pred['Change_Percent'].mean()
                st.metric("Avg Change", f"{avg_change:+.1f}%")
            with col3:
                increasing = (filtered_pred['Change_Percent'] > 0).sum()
                st.metric("Trending Up", f"{increasing} / {len(filtered_pred)}")

            display_df = filtered_pred[['Institute', 'Academic Program Name', 'Round',
                                       'Closing_Rank_2024', 'Predicted_Closing_Rank_2025',
                                       'Change_Percent']].copy()
            display_df.columns = ['Institute', 'Program', 'Round', '2024 Closing', '2025 Predicted', 'Δ %']
            st.dataframe(display_df.head(50), use_container_width=True, height=420)

            st.markdown("""
            <div class="section-header">
                <div class="section-eyebrow">Visualization</div>
                <div class="section-title">Top 20 Predicted Movements</div>
            </div>
            """, unsafe_allow_html=True)

            top_changes = filtered_pred.nlargest(20, 'Change_Percent')
            fig = px.bar(
                top_changes,
                x='Change_Percent',
                y='Institute',
                color='Change_Percent',
                color_continuous_scale=[[0, '#9C7E3F'], [0.5, '#C9A961'], [1, '#0A1929']],
                orientation='h',
                labels={'Change_Percent': 'Predicted Change (%)', 'Institute': ''}
            )
            fig.update_layout(**PREMIUM_PLOTLY_LAYOUT, height=600, showlegend=False)
            fig.update_traces(marker_line_color='#0A1929', marker_line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Forecast data is currently unavailable.")

# ───────── TAB 3: Historical Analysis ─────────
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-eyebrow">Historical Intelligence</div>
        <div class="section-title">Five-Year Cutoff Trend Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            institutes = sorted(df['Institute'].unique().tolist())
            selected_institute = st.selectbox("Institute", institutes)

        institute_data = df[df['Institute'] == selected_institute]

        with col2:
            if len(institute_data) > 0:
                branches = sorted(institute_data['Academic Program Name'].unique().tolist())
                selected_branch = st.selectbox("Academic Program", branches)
            else:
                selected_branch = None

        if selected_branch:
            branch_data = institute_data[institute_data['Academic Program Name'] == selected_branch]

            if len(branch_data) > 0:
                yearly_data = branch_data.groupby('Year').agg({
                    'Opening Rank': 'min',
                    'Closing Rank': 'max'
                }).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=yearly_data['Year'], y=yearly_data['Opening Rank'],
                    name='Opening Rank', mode='lines+markers',
                    line=dict(color='#C9A961', width=2.5),
                    marker=dict(size=10, line=dict(color='#FAFAF7', width=2), color='#C9A961')
                ))
                fig.add_trace(go.Scatter(
                    x=yearly_data['Year'], y=yearly_data['Closing Rank'],
                    name='Closing Rank', mode='lines+markers',
                    line=dict(color='#0A1929', width=2.5),
                    marker=dict(size=10, line=dict(color='#FAFAF7', width=2), color='#0A1929')
                ))

                fig.update_layout(
                    **PREMIUM_PLOTLY_LAYOUT,
                    title=dict(
                        text=f'<b>{selected_institute}</b><br><span style="font-size:13px; color:#475569; font-weight:400;">{selected_branch}</span>',
                        font=dict(family='Playfair Display, Georgia, serif', size=18, color='#0A1929'),
                        x=0.02
                    ),
                    xaxis_title='Year',
                    yaxis_title='Rank',
                    hovermode='x unified',
                    height=520,
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1,
                        font=dict(family='JetBrains Mono, monospace', size=10),
                        bgcolor='rgba(0,0,0,0)'
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("""
                <div class="section-header">
                    <div class="section-eyebrow">Tabular Data</div>
                    <div class="section-title">Year-on-Year Breakdown</div>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(yearly_data, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="premium-footer">
    <div class="footer-mark">JoSAA <span class="accent">Intelligence</span></div>
    <div class="footer-tagline">— Engineered for Aspirants —</div>
    <div class="footer-meta">
        <strong>Bachelor Thesis Project I12</strong> · Department of Mechanical Engineering · Indian Institute of Technology Delhi<br>
        Powered by XGBoost · Cutoff Prediction & Admission Probability Models<br>
        <span style="color: var(--slate-light); font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.1em;">
            © 2026 · careerjankari.com
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
