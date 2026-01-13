"""
Stone Column Design Assistant V5 - Modern Professional Edition
Advanced AI-powered geotechnical design tool with contemporary UI/UX
Enhanced design with glassmorphism, smooth animations, and improved visual hierarchy
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pickle
from datetime import datetime

st.set_page_config(
    page_title="Stone Column Design Assistant V5",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODERN STYLING WITH GLASSMORPHISM AND ANIMATIONS
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    /* Animated Gradient Header */
    .hero-header {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient-shift 15s ease infinite;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        z-index: 0;
    }
    
    .hero-header > * {
        position: relative;
        z-index: 1;
    }
    
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .hero-header p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-top: 0.8rem;
        font-weight: 400;
    }
    
    /* Glass Card Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stMetric::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .stMetric:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .stMetric:hover::before {
        left: 100%;
    }
    
    /* Headers with Gradient Underline */
    h2 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2rem;
        padding-bottom: 1rem;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100px;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    h3 {
        color: #e0e0e0;
        font-weight: 600;
        font-size: 1.4rem;
        margin-top: 2rem;
    }
    
    /* Glass Info Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 1.8rem;
        border-radius: 20px;
        margin: 1.2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Status Cards */
    .status-success {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15), rgba(76, 175, 80, 0.05));
        border-left: 4px solid #4caf50;
        padding: 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.15), rgba(255, 152, 0, 0.05));
        border-left: 4px solid #ff9800;
        padding: 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    .status-error {
        background: linear-gradient(135deg, rgba(244, 67, 54, 0.15), rgba(244, 67, 54, 0.05));
        border-left: 4px solid #f44336;
        padding: 1.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(30, 33, 41, 0.95) 0%, rgba(45, 49, 57, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h2 {
        color: #667eea;
        font-size: 1.6rem;
    }
    
    /* Input Fields */
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        color: white;
        padding: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.12);
    }
    
    /* Checkbox Styling */
    .stCheckbox {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.8rem 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .stCheckbox:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Primary Button (Run Analysis) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff 0%, #00a8cc 100%);
        font-size: 1.1rem;
        padding: 1rem 2rem;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #00e5ff 0%, #00b8dd 100%);
        box-shadow: 0 6px 30px rgba(0, 212, 255, 0.6);
        transform: translateY(-2px);
    }
    
    /* Loading Animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
        50% { box-shadow: 0 0 40px rgba(0, 212, 255, 0.8); }
    }
    
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 24px;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }
    
    .spinner-container {
        position: relative;
        width: 120px;
        height: 120px;
        margin-bottom: 2rem;
    }
    
    .spinner {
        width: 120px;
        height: 120px;
        border: 4px solid rgba(0, 212, 255, 0.1);
        border-top: 4px solid #00d4ff;
        border-radius: 50%;
        animation: spin 1s linear infinite, pulse-glow 2s ease-in-out infinite;
    }
    
    .spinner-icon {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 3rem;
    }
    
    .loading-text {
        font-size: 1.5rem;
        color: white;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .loading-subtitle {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 2rem;
    }
    
    .loading-steps {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        width: 100%;
        max-width: 500px;
    }
    
    .loading-step {
        display: flex;
        align-items: center;
        padding: 0.8rem 0;
        color: rgba(255, 255, 255, 0.5);
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .loading-step.active {
        color: #00d4ff;
        font-weight: 600;
    }
    
    .loading-step.complete {
        color: #4caf50;
    }
    
    .loading-step-icon {
        margin-right: 1rem;
        font-size: 1.2rem;
        min-width: 30px;
        text-align: center;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(79, 172, 254, 0.6);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: white;
        font-weight: 600;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* DataFrames */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Info/Warning/Error Boxes */
    .stAlert {
        border-radius: 16px;
        backdrop-filter: blur(10px);
        border: none;
    }
    
    /* Plotly Charts Dark Mode Enhancement */
    .js-plotly-plot {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 2rem 0;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(102, 126, 234, 0.3);
    }
    
    /* Footer */
    .footer-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2rem;
        border-radius: 20px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Pulse Animation for Important Elements */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse-element {
        animation: pulse 2s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL AND SCALERS
@st.cache_resource
def load_all():
    try:
        model = tf.keras.models.load_model("stone_column_ann_model.h5", compile=False)
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None, None

model, scaler_X, scaler_y = load_all()

# PARAMETERS
PARAM_RANGES = {
    'cu': (5.0, 40.0, 15.0),
    'D': (0.06, 0.8, 0.4),
    'L': (0.7, 12.0, 6.0),
    'sD': (2.0, 4.0, 2.5),
    'Eenc': (0.0, 20.0, 0.0)
}

PARAM_INFO = {
    'cu': {'name': 'Undrained Shear Strength', 'unit': 'kPa', 'icon': '🏔️'},
    'D': {'name': 'Column Diameter', 'unit': 'm', 'icon': '⭕'},
    'L': {'name': 'Column Length', 'unit': 'm', 'icon': '📏'},
    'sD': {'name': 'Spacing Ratio (s/D)', 'unit': '-', 'icon': '📐'},
    'Eenc': {'name': 'Encasement Stiffness', 'unit': 'kN/m', 'icon': '🔧'}
}

# CORE FUNCTIONS
def predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc):
    x = np.array([[cu, D, L, sD, Eenc]], dtype=np.float32)
    x_scaled = scaler_X.transform(x)
    pred_scaled = model.predict(x_scaled, verbose=0)[0]
    pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1))[0]
    
    sigma, P10 = pred[0], pred[1]
    FS = compute_FS(sigma, P10, cu, D, L, sD)
    return sigma, P10, FS

def compute_FS(sigma, P10, cu, D, L, sD):
    if P10 <= 0 or sigma <= 0:
        return 0.0
    FS = sigma / P10
    corr = 1.0
    if cu < 10:
        corr *= (0.7 + 0.03 * cu)
    if D < 0.25:
        corr *= (0.8 + 0.8 * D)
    if sD < 2.5:
        corr *= (0.85 + 0.06 * sD)
    if L < 3.0:
        corr *= (0.75 + 0.083 * L)
    return max(0.5, min(FS * corr, 5.0))

def calc_derived(cu, D, L, sD, sigma, P10):
    A_col = np.pi * (D/2)**2
    return {
        'spacing': sD * D,
        'slenderness': L/D,
        'area_repl': 1/(sD**2),
        'improv_factor': sigma/cu if cu > 0 else 0,
        'col_area': A_col,
        'load_kN': P10 * A_col,
        'load_per_len': (P10 * A_col)/L if L > 0 else 0
    }

def validate_design(cu, D, L, sD):
    warnings, reliable = [], True
    if cu < 10:
        warnings.append(f"⚠️ Weak soil (cu={cu:.1f} kPa)")
        reliable = False
    if D < 0.25:
        warnings.append(f"⚠️ Small diameter (D={D:.2f} m)")
        reliable = False
    if L < 3.0:
        warnings.append(f"⚠️ Short column (L={L:.1f} m)")
        reliable = False
    if sD < 2.5:
        warnings.append(f"⚠️ Tight spacing (s/D={sD:.1f})")
        reliable = False
    if L/D > 20:
        warnings.append(f"⚠️ High slenderness (L/D={L/D:.1f})")
    return reliable, warnings

def assess_safety(FS, reliable):
    if not reliable:
        return "UNRELIABLE", "Outside validated parameter range", "error"
    if FS >= 2.5:
        return "Excellent", "Exceeds safety requirements", "success"
    elif FS >= 2.0:
        return "Adequate", "Meets safety requirements", "success"
    elif FS >= 1.5:
        return "Marginal", "Consider design optimization", "warning"
    else:
        return "Insufficient", "Design revision required", "error"

# HERO HEADER
st.markdown("""
<div class="hero-header">
    <h1>🏗️ Stone Column Design Assistant V5</h1>
    <p>Advanced AI-Powered Geotechnical Engineering Platform</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR WITH MODERN DESIGN
with st.sidebar:
    st.markdown("### 🎛️ Design Parameters")
    st.markdown("---")
    
    cu = st.number_input(
        f"{PARAM_INFO['cu']['icon']} {PARAM_INFO['cu']['name']} ({PARAM_INFO['cu']['unit']})",
        PARAM_RANGES['cu'][0], PARAM_RANGES['cu'][1], PARAM_RANGES['cu'][2], 0.5
    )
    
    D = st.number_input(
        f"{PARAM_INFO['D']['icon']} {PARAM_INFO['D']['name']} ({PARAM_INFO['D']['unit']})",
        PARAM_RANGES['D'][0], PARAM_RANGES['D'][1], PARAM_RANGES['D'][2], 0.01, format="%.2f"
    )
    
    L = st.number_input(
        f"{PARAM_INFO['L']['icon']} {PARAM_INFO['L']['name']} ({PARAM_INFO['L']['unit']})",
        PARAM_RANGES['L'][0], PARAM_RANGES['L'][1], PARAM_RANGES['L'][2], 0.1, format="%.1f"
    )
    
    sD = st.number_input(
        f"{PARAM_INFO['sD']['icon']} {PARAM_INFO['sD']['name']} ({PARAM_INFO['sD']['unit']})",
        PARAM_RANGES['sD'][0], PARAM_RANGES['sD'][1], PARAM_RANGES['sD'][2], 0.1, format="%.1f"
    )
    
    Eenc = st.number_input(
        f"{PARAM_INFO['Eenc']['icon']} {PARAM_INFO['Eenc']['name']} ({PARAM_INFO['Eenc']['unit']})",
        PARAM_RANGES['Eenc'][0], PARAM_RANGES['Eenc'][1], PARAM_RANGES['Eenc'][2], 0.5, format="%.1f"
    )
    
    st.info(f"**📍 Calculated Spacing:** {sD * D:.2f} m")
    
    st.markdown("---")
    st.markdown("### 📊 Analysis Options")
    
    sens = st.checkbox("🔍 Sensitivity Analysis", True)
    heat = st.checkbox("🌡️ Interaction Heatmap", True)
    surf3d = st.checkbox("🎨 3D Surface Plot", True)
    
    st.markdown("---")
    run_button = st.button("▶️ Run Analysis", use_container_width=True, type="primary")
    reset_button = st.button("🔄 Reset Defaults", use_container_width=True)

# PREDICTION
if model is None:
    st.warning("⚠️ Model not loaded. Please check model files.")
    st.stop()

# Initialize session state for calculations
if 'calculated' not in st.session_state:
    st.session_state.calculated = False
if 'calculation_params' not in st.session_state:
    st.session_state.calculation_params = None

# Handle Reset Button
if reset_button:
    st.session_state.calculated = False
    st.session_state.calculation_params = None
    st.rerun()

# Handle Run Analysis Button
if run_button:
    st.session_state.calculated = False
    st.session_state.calculation_params = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}
    
    # Show loading animation
    st.markdown("""
    <div class="loading-container">
        <div class="spinner-container">
            <div class="spinner"></div>
            <div class="spinner-icon">🧠</div>
        </div>
        <div class="loading-text">AI model is evaluating design...</div>
        <div class="loading-subtitle">Running TensorFlow prediction and safety checks</div>
        <div class="loading-steps">
            <div class="loading-step active">
                <div class="loading-step-icon">📋</div>
                <div>Scaling inputs to model range...</div>
            </div>
            <div class="loading-step active">
                <div class="loading-step-icon">🧠</div>
                <div>Running TensorFlow prediction...</div>
            </div>
            <div class="loading-step">
                <div class="loading-step-icon">🛡️</div>
                <div>Computing factor of safety...</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Simulate processing time
    import time
    time.sleep(1.5)
    
    st.session_state.calculated = True
    st.rerun()

# Check if we should show results
if not st.session_state.calculated:
    st.info("👆 Configure your parameters in the sidebar and click **Run Analysis** to begin.")
    st.stop()

# Get calculation parameters
if st.session_state.calculation_params:
    inp = st.session_state.calculation_params
    cu, D, L, sD, Eenc = inp['cu'], inp['D'], inp['L'], inp['sD'], inp['Eenc']
else:
    inp = {'cu': cu, 'D': D, 'L': L, 'sD': sD, 'Eenc': Eenc}

sigma, P10, FS = predict_outcomes(model, scaler_X, scaler_y, cu, D, L, sD, Eenc)
der = calc_derived(cu, D, L, sD, sigma, P10)
reliable, warns = validate_design(cu, D, L, sD)

# RESULTS SECTION
st.markdown("## 🎯 Prediction Results")

if warns:
    st.markdown('<div class="status-error">', unsafe_allow_html=True)
    st.markdown("**⚠️ Reliability Warnings:**")
    for w in warns:
        st.write(w)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

# KEY METRICS
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "💎 Ultimate Stress",
    f"{sigma:.2f} kPa",
    help="AI-predicted ultimate bearing stress"
)

c2.metric(
    "⚖️ Service Load",
    f"{P10:.2f} kPa",
    help="AI-predicted service load capacity"
)

c3.metric(
    "🛡️ Factor of Safety",
    f"{FS:.2f}",
    delta="✓ Safe" if FS >= 2.0 else "⚠ Low",
    delta_color="normal" if FS >= 2.0 else "inverse",
    help="Computed as σ/P10 with correction factors"
)

c4.metric(
    "📊 Slenderness",
    f"{der['slenderness']:.1f}",
    help="Length to diameter ratio (L/D)"
)

# DESIGN ASSESSMENT
st.markdown("### 🔍 Design Assessment")
status, msg, col = assess_safety(FS, reliable)

if col == "success":
    st.markdown(f'<div class="status-success"><strong>✅ {status}:</strong> {msg}</div>', unsafe_allow_html=True)
elif col == "warning":
    st.markdown(f'<div class="status-warning"><strong>⚠️ {status}:</strong> {msg}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="status-error"><strong>❌ {status}:</strong> {msg}</div>', unsafe_allow_html=True)

# DESIGN INFORMATION
st.markdown("### 📋 Design Information")

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"""
    **🏗️ Column Configuration**
    
    - **Type:** {'🔗 Encased' if Eenc > 0 else '⚪ Unencased'}
    - **Area Replacement Ratio:** {der['area_repl']:.3f}
    - **Center-to-Center Spacing:** {der['spacing']:.2f} m
    - **Slenderness Ratio (L/D):** {der['slenderness']:.1f}
    - **Column Cross-Sectional Area:** {der['col_area']:.4f} m²
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    eff = "🔥 High" if der['improv_factor'] > 3 else "✅ Moderate" if der['improv_factor'] > 2 else "⚠️ Low"
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(f"""
    **⚡ Performance Metrics**
    
    - **Soil Improvement Factor:** {der['improv_factor']:.2f}x
    - **Total Service Load:** {der['load_kN']:.2f} kN
    - **Load per Unit Length:** {der['load_per_len']:.2f} kN/m
    - **Efficiency Rating:** {eff}
    - **Reliability Status:** {'✅ Validated' if reliable else '❌ UNRELIABLE'}
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# SENSITIVITY ANALYSIS
if sens:
    st.markdown("## 📈 Sensitivity Analysis")
    
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    sel = st.selectbox("Select Parameter to Analyze:", list(opts.keys()))
    key = opts[sel]
    
    vals = np.linspace(PARAM_RANGES[key][0], PARAM_RANGES[key][1], 60)
    preds = []
    
    for v in vals:
        t = inp.copy()
        t[key] = v
        x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]])
        xs = scaler_X.transform(x)
        ps = model.predict(xs, verbose=0)[0]
        p = scaler_y.inverse_transform(ps.reshape(1,-1))[0]
        fs = compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
        preds.append([p[0], p[1], fs])
    
    preds = np.array(preds)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["💎 Ultimate Stress", "⚖️ Service Load", "🛡️ Factor of Safety"]
    )
    
    colors = ['#667eea', '#4facfe', '#f093fb']
    names = ["Ultimate Stress (kPa)", "Service Load (kPa)", "Factor of Safety"]
    
    for i in range(3):
        fig.add_trace(
            go.Scatter(
                x=vals, y=preds[:,i],
                mode='lines',
                name=names[i],
                line=dict(color=colors[i], width=3),
                fill='tozeroy',
                fillcolor=f'rgba({int(colors[i][1:3], 16)}, {int(colors[i][3:5], 16)}, {int(colors[i][5:7], 16)}, 0.2)'
            ),
            row=1, col=i+1
        )
        fig.add_vline(x=inp[key], line_dash="dash", line_color="red", line_width=2, row=1, col=i+1)
    
    fig.add_hline(y=2.0, line_dash="dot", line_color="#4caf50", line_width=2, row=1, col=3)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', title_text=sel)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Statistical Summary")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric("Min σ", f"{preds[:,0].min():.2f} kPa")
        st.metric("Max σ", f"{preds[:,0].max():.2f} kPa")
        st.metric("Range", f"{preds[:,0].max() - preds[:,0].min():.2f} kPa")
    
    with c2:
        st.metric("Min P10", f"{preds[:,1].min():.2f} kPa")
        st.metric("Max P10", f"{preds[:,1].max():.2f} kPa")
        st.metric("Range", f"{preds[:,1].max() - preds[:,1].min():.2f} kPa")
    
    with c3:
        st.metric("Min FS", f"{preds[:,2].min():.2f}")
        st.metric("Max FS", f"{preds[:,2].max():.2f}")
        st.metric("Range", f"{preds[:,2].max() - preds[:,2].min():.2f}")
    
    st.markdown("---")

# INTERACTION HEATMAP
if heat:
    st.markdown("## 🌡️ Parameter Interaction Analysis")
    
    c1, c2, c3 = st.columns(3)
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    
    p1n = c1.selectbox("X-Axis Parameter:", list(opts.keys()), key='h1')
    p2n = c2.selectbox("Y-Axis Parameter:", list(opts.keys()), index=1, key='h2')
    outn = c3.selectbox("Output Metric:", ["Ultimate Stress", "Service Load", "Factor of Safety"])
    
    if p1n != p2n:
        p1k, p2k = opts[p1n], opts[p2n]
        res = 40
        p1v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], res)
        p2v = np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], res)
        Z = np.zeros((len(p2v), len(p1v)))
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)
        
        for i, v2 in enumerate(p2v):
            for j, v1 in enumerate(p1v):
                t = inp.copy()
                t[p1k], t[p2k] = v1, v2
                x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]])
                xs = scaler_X.transform(x)
                ps = model.predict(xs, verbose=0)[0]
                p = scaler_y.inverse_transform(ps.reshape(1,-1))[0]
                Z[i,j] = p[oidx] if oidx < 2 else compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
        
        fig = go.Figure(data=go.Heatmap(
            z=Z,
            x=p1v,
            y=p2v,
            colorscale='Plasma',
            colorbar=dict(title=outn, thickness=20, len=0.7)
        ))
        
        fig.add_scatter(
            x=[inp[p1k]],
            y=[inp[p2k]],
            mode='markers+text',
            marker=dict(size=25, color='white', symbol='star', line=dict(width=3, color='#667eea')),
            text=['Current'],
            textposition='top center',
            textfont=dict(size=14, color='white'),
            name='Current Design'
        )
        
        fig.update_layout(
            title=dict(text=f"{outn} - Parameter Interaction", font=dict(size=20, color='white')),
            xaxis_title=p1n,
            yaxis_title=p2n,
            height=650,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        oi = np.unravel_index(np.argmax(Z), Z.shape)
        st.success(f"🎯 **Optimal Configuration:** {p1n} = {p1v[oi[1]]:.2f}, {p2n} = {p2v[oi[0]]:.2f}, {outn} = {Z[oi]:.2f}")
    else:
        st.warning("⚠️ Please select different parameters for X and Y axes")
    
    st.markdown("---")

# 3D SURFACE PLOT
if surf3d:
    st.markdown("## 🎨 3D Surface Visualization")
    
    c1, c2, c3 = st.columns(3)
    opts = {PARAM_INFO[k]['name']: k for k in PARAM_INFO}
    
    p1n = c1.selectbox("X-Axis Parameter:", list(opts.keys()), key='3d1')
    p2n = c2.selectbox("Y-Axis Parameter:", list(opts.keys()), index=1, key='3d2')
    outn = c3.selectbox("Z-Axis Output:", ["Ultimate Stress", "Service Load", "Factor of Safety"])
    
    if p1n != p2n:
        res = st.slider("Resolution Quality:", 15, 30, 20, help="Higher resolution = better quality but slower")
        
        p1k, p2k = opts[p1n], opts[p2n]
        p1v = np.linspace(PARAM_RANGES[p1k][0], PARAM_RANGES[p1k][1], res)
        p2v = np.linspace(PARAM_RANGES[p2k][0], PARAM_RANGES[p2k][1], res)
        P1, P2 = np.meshgrid(p1v, p2v)
        Z = np.zeros_like(P1)
        oidx = ["Ultimate Stress", "Service Load", "Factor of Safety"].index(outn)
        
        for i in range(P1.shape[0]):
            for j in range(P1.shape[1]):
                t = inp.copy()
                t[p1k], t[p2k] = P1[i,j], P2[i,j]
                x = np.array([[t['cu'], t['D'], t['L'], t['sD'], t['Eenc']]])
                xs = scaler_X.transform(x)
                ps = model.predict(xs, verbose=0)[0]
                p = scaler_y.inverse_transform(ps.reshape(1,-1))[0]
                Z[i,j] = p[oidx] if oidx < 2 else compute_FS(p[0], p[1], t['cu'], t['D'], t['L'], t['sD'])
        
        fig = go.Figure(data=[go.Surface(
            z=Z,
            x=p1v,
            y=p2v,
            colorscale='Viridis',
            colorbar=dict(title=outn, thickness=20, len=0.7),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))
            )
        )])
        
        fig.update_layout(
            title=dict(text=f"3D Surface: {outn}", font=dict(size=20)),
            scene=dict(
                xaxis_title=p1n,
                yaxis_title=p2n,
                zaxis_title=outn,
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)')
            ),
            height=750,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please select different parameters for X and Y axes")
    
    st.markdown("---")

# EXPORT RESULTS
st.markdown("## 📥 Export & Reporting")

c1, c2 = st.columns(2)

with c1:
    with st.expander("📄 Export CSV", expanded=False):
        df = pd.DataFrame({
            'Parameter': [PARAM_INFO[k]['name'] for k in PARAM_INFO] + 
                         ['', 'Ultimate Stress', 'Service Load', 'Factor of Safety'],
            'Value': [cu, D, L, sD, Eenc, '', sigma, P10, FS],
            'Unit': [PARAM_INFO[k]['unit'] for k in PARAM_INFO] + ['', 'kPa', 'kPa', '-']
        })
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="⬇️ Download CSV Report",
            data=df.to_csv(index=False),
            file_name=f"stone_column_design_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )

with c2:
    with st.expander("📑 Export PDF", expanded=False):
        st.info("📋 Reports include all input parameters, results, and safety assessments")
        st.markdown("""
        **PDF Report will include:**
        - Input parameters and configuration
        - Prediction results and metrics
        - Safety assessment and recommendations
        - Design summary and warnings
        """)
        st.button("⬇️ Download PDF Report", use_container_width=True, disabled=True, 
                 help="PDF export feature coming soon")

st.markdown("---")

# RECOMMENDATIONS
st.markdown("## 💡 Design Recommendations")

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🏗️ Structural Recommendations")
    
    if FS < 1.5:
        st.markdown("""
        <div class="status-error">
        <strong>❌ Critical Issues Detected</strong>
        
        - ⬆️ Increase column diameter or length
        - ⬇️ Reduce spacing between columns
        - 🔗 Consider adding encasement
        - 🔄 Review soil conditions
        </div>
        """, unsafe_allow_html=True)
    elif FS < 2.0:
        st.markdown("""
        <div class="status-warning">
        <strong>⚠️ Optimization Suggested</strong>
        
        - 📐 Adjust column dimensions
        - 📏 Reduce spacing by 10-15%
        - 🔍 Verify soil parameters
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-success">
        <strong>✅ Design Meets Requirements</strong>
        
        - ✓ Safety criteria satisfied
        - 💰 Consider cost optimization
        - 📊 May reduce dimensions slightly
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Economic Considerations")
    
    cost_index = (D**2) * L / (sD**2)
    
    if cost_index > 2.0:
        st.markdown("""
        <div class="status-warning">
        <strong>⚠️ High Material Usage</strong>
        
        - 📉 Reduce diameter if FS permits
        - 📏 Increase spacing ratio
        - 🔄 Consider alternative layouts
        - 💵 Estimated: High cost
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-success">
        <strong>✅ Cost-Efficient Design</strong>
        
        - 💚 Optimized material usage
        - 📊 Good balance achieved
        - 💵 Estimated: Reasonable cost
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# DESIGN SUMMARY
st.markdown("## 📊 Design Summary Dashboard")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Design Status", status)
    st.metric("Factor of Safety", f"{FS:.2f}")
    st.metric("Column Type", "Encased" if Eenc > 0 else "Unencased")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Ultimate Stress σ", f"{sigma:.1f} kPa")
    st.metric("Service Load P10", f"{P10:.1f} kPa")
    st.metric("Improvement Factor", f"{der['improv_factor']:.2f}x")
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Center Spacing", f"{der['spacing']:.2f} m")
    st.metric("Slenderness L/D", f"{der['slenderness']:.1f}")
    st.metric("Area Replacement", f"{der['area_repl']:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("---")

st.markdown("""
<div class="footer-card">
    <h3 style="margin-top:0;">⚠️ Professional Disclaimer</h3>
    <p style="margin-bottom:0;">
    This tool provides AI-assisted preliminary design estimates based on machine learning models.
    All results must be verified by qualified geotechnical engineers using site-specific data,
    laboratory testing, and applicable design codes before implementation.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; opacity:0.6; margin-top:2rem; padding-bottom:2rem;">
    <strong>Stone Column Design Assistant V5</strong> © 2026<br>
    Advanced AI-Powered Geotechnical Engineering Platform<br>
    Factor of Safety computed using engineering formulas: σ/P10 with correction factors
</div>
""", unsafe_allow_html=True)
