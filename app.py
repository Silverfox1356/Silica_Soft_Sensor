import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Silica Soft Sensor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.stApp { background-color: #0f1117; color: #e8eaf0; }

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.2rem; font-weight: 600; }
.metric-label { font-size: 0.75rem; color: #8b8fa8; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

.alert-green  { color: #4ade80; border-color: #4ade80; }
.alert-amber  { color: #fbbf24; border-color: #fbbf24; }
.alert-red    { color: #f87171; border-color: #f87171; }

.warning-box {
    background: #2a1f00;
    border: 1px solid #fbbf24;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-top: 0.5rem;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: #8b8fa8;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load('models/rf_silica_model.pkl')
    features = pd.read_csv('data/feature_list.csv')['feature'].tolist()
    metrics  = pd.read_csv('data/model_metrics.csv', index_col=0)
    return model, features, metrics

model, FEATURES, metrics_df = load_artifacts()

# Operating bounds from training data (from EDA descriptive stats)
BOUNDS = {
    '% Iron Feed':                        (42.74, 65.78),
    '% Silica Feed':                      (1.31,  33.40),
    'Starch Flow':                        (54.6,  6270.2),
    'Amina Flow':                         (242.9, 737.0),
    'Ore Pulp Flow':                      (376.8, 418.1),
    'Ore Pulp pH':                        (8.75,  10.81),
    'Flotation Column 01 Air Flow':       (175.9, 312.3),
    'Flotation Column 02 Air Flow':       (178.2, 309.9),
    'Flotation Column 03 Air Flow':       (177.2, 302.8),
    'Flotation Column 06 Air Flow':       (196.5, 355.0),
    'Flotation Column 07 Air Flow':       (199.7, 351.3),
    'Flotation Column 01 Level':          (181.9, 859.0),
    'Flotation Column 02 Level':          (224.9, 827.8),
    'Flotation Column 03 Level':          (135.2, 884.8),
    'Flotation Column 04 Level':          (165.7, 675.6),
    'Flotation Column 05 Level':          (214.7, 674.1),
    'Flotation Column 06 Level':          (203.7, 698.5),
    'Flotation Column 07 Level':          (185.1, 655.5),
    '% Iron Concentrate':                 (62.05, 68.01),
}

SPEC_LIMIT = 2.0  # % SiO2 upper specification limit

# ── Sidebar inputs ───────────────────────────────────────────────
st.sidebar.markdown("## ⚗️ Silica Soft Sensor")
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">Sensor Readings</p>', unsafe_allow_html=True)

# Default values = training data means
DEFAULTS = {
    '% Iron Feed': 56.3, '% Silica Feed': 14.7, 'Starch Flow': 2869.0,
    'Amina Flow': 488.1, 'Ore Pulp Flow': 397.6, 'Ore Pulp pH': 9.77,
    'Flotation Column 01 Air Flow': 280.2, 'Flotation Column 02 Air Flow': 277.2,
    'Flotation Column 03 Air Flow': 281.1, 'Flotation Column 06 Air Flow': 292.1,
    'Flotation Column 07 Air Flow': 290.8, 'Flotation Column 01 Level': 520.2,
    'Flotation Column 02 Level': 522.6, 'Flotation Column 03 Level': 531.4,
    'Flotation Column 04 Level': 420.3, 'Flotation Column 05 Level': 425.3,
    'Flotation Column 06 Level': 429.9, 'Flotation Column 07 Level': 421.0,
    '% Iron Concentrate': 65.05,
}

# Engineered feature defaults (lag and rolling = same as current reading)
ENG_DEFAULTS = {
    'Silica_lag_1': 2.33, 'Silica_lag_2': 2.33,
    'Amina Flow_lag1': 488.1, 'Starch Flow_lag1': 2869.0,
    'Flotation Column 01 Air Flow_lag1': 280.2,
    'Flotation Column 01 Air Flow_roll_mean3': 280.2,
    'Flotation Column 01 Air Flow_roll_std3': 5.0,
    'Flotation Column 03 Air Flow_roll_mean3': 281.1,
    'Flotation Column 03 Air Flow_roll_std3': 5.0,
    'Amina_x_Col01Air': 488.1 * 280.2,
}

# Group inputs into sections
FEED_COLS    = ['% Iron Feed', '% Silica Feed', 'Ore Pulp Flow', 'Ore Pulp pH']
REAGENT_COLS = ['Starch Flow', 'Amina Flow']
AIR_COLS     = [f for f in FEATURES if 'Air Flow' in f and 'lag' not in f and 'roll' not in f]
LEVEL_COLS   = [f for f in FEATURES if 'Level' in f and 'lag' not in f and 'roll' not in f]
OTHER_COLS   = ['% Iron Concentrate']

inputs = {}
out_of_range = []

def add_inputs(cols, label):
    st.sidebar.markdown(f'<p class="section-header">{label}</p>', unsafe_allow_html=True)
    for col in cols:
        if col not in FEATURES:
            continue
        lo, hi = BOUNDS.get(col, (0.0, 1000.0))
        val = st.sidebar.number_input(col, min_value=float(lo)*0.5,
                                       max_value=float(hi)*1.5,
                                       value=float(DEFAULTS.get(col, (lo+hi)/2)),
                                       format="%.2f", key=col)
        inputs[col] = val
        if val < lo or val > hi:
            out_of_range.append(col)

add_inputs(FEED_COLS,    "Feed & Pulp")
add_inputs(REAGENT_COLS, "Reagents")
add_inputs(AIR_COLS,     "Air Flow")
add_inputs(LEVEL_COLS,   "Column Levels")
add_inputs(OTHER_COLS,   "Concentrate")

# Fill engineered features with defaults
for feat in FEATURES:
    if feat not in inputs:
        inputs[feat] = ENG_DEFAULTS.get(feat, 0.0)

# Build input vector in correct feature order
X_in = np.array([inputs[f] for f in FEATURES]).reshape(1, -1)
pred = float(model.predict(X_in)[0])

# Alert level
if pred > SPEC_LIMIT:
    alert_class = "alert-red"
    alert_text  = "⚠ ABOVE SPECIFICATION"
elif pred > SPEC_LIMIT * 0.9:
    alert_class = "alert-amber"
    alert_text  = "◈ APPROACHING LIMIT"
else:
    alert_class = "alert-green"
    alert_text  = "✓ WITHIN SPECIFICATION"

# Session state for trend
if 'trend' not in st.session_state:
    st.session_state.trend = []
if st.sidebar.button("Log this prediction"):
    st.session_state.trend.append(round(pred, 3))
    if len(st.session_state.trend) > 10:
        st.session_state.trend.pop(0)

# ── Pages ────────────────────────────────────────────────────────
pages = st.tabs(["🔬 Predict", "📊 Model Performance", "🔍 Feature Importance", "📋 About"])

# ════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════
with pages[0]:
    st.markdown("## Silica Concentration Prediction")
    st.markdown("Real-time % SiO₂ inference from flotation process sensor data.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card {'alert-red' if alert_class == 'alert-red' else 'alert-amber' if alert_class == 'alert-amber' else 'alert-green'}" 
             style="border-color: {'#f87171' if alert_class == 'alert-red' else '#fbbf24' if alert_class == 'alert-amber' else '#4ade80'}">
            <div class="metric-value" style="color: {'#f87171' if alert_class == 'alert-red' else '#fbbf24' if alert_class == 'alert-amber' else '#4ade80'}">
                {pred:.3f}%
            </div>
            <div class="metric-label">Predicted % SiO₂</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #8b8fa8">{SPEC_LIMIT:.1f}%</div>
            <div class="metric-label">Specification Limit</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-color: {'#f87171' if alert_class == 'alert-red' else '#fbbf24' if alert_class == 'alert-amber' else '#4ade80'}">
            <div class="metric-value" style="color: {'#f87171' if alert_class == 'alert-red' else '#fbbf24' if alert_class == 'alert-amber' else '#4ade80'}; font-size: 1.2rem; padding-top: 0.5rem;">
                {alert_text}
            </div>
            <div class="metric-label">Status</div>
        </div>
        """, unsafe_allow_html=True)

    # Out of range warning
    if out_of_range:
        st.markdown(f"""
        <div class="warning-box">
            ⚠ The following inputs are outside normal operating range and may reduce prediction reliability:
            {', '.join(out_of_range)}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_shap, col_trend = st.columns([3, 2])

    with col_shap:
        st.markdown('<p class="section-header">SHAP Explanation — What drove this prediction</p>', unsafe_allow_html=True)
        try:
            explainer   = shap.TreeExplainer(model)
            shap_vals   = explainer(X_in)
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
            plt.gcf().patch.set_facecolor('#1a1d27')
            for text in plt.gcf().findobj(plt.Text):
                text.set_color('#e8eaf0')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.info("SHAP explanation unavailable — ensure the model file is loaded correctly.")

    with col_trend:
        st.markdown('<p class="section-header">Prediction Trend (last 10 logged)</p>', unsafe_allow_html=True)
        if len(st.session_state.trend) >= 2:
            trend_df = pd.DataFrame({
                'Reading': range(1, len(st.session_state.trend) + 1),
                '% SiO₂': st.session_state.trend
            })
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            fig2.patch.set_facecolor('#1a1d27')
            ax2.set_facecolor('#1a1d27')
            ax2.plot(trend_df['Reading'], trend_df['% SiO₂'],
                     color='#60a5fa', lw=2, marker='o', markersize=4)
            ax2.axhline(y=SPEC_LIMIT, color='#f87171', lw=1, linestyle='--', label=f'Spec limit ({SPEC_LIMIT}%)')
            ax2.set_xlabel('Reading', color='#8b8fa8', fontsize=9)
            ax2.set_ylabel('% SiO₂', color='#8b8fa8', fontsize=9)
            ax2.tick_params(colors='#8b8fa8')
            for spine in ax2.spines.values():
                spine.set_edgecolor('#2a2d3a')
            ax2.legend(fontsize=8, labelcolor='#e8eaf0', facecolor='#1a1d27')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
        else:
            st.markdown("""
            <div style="color: #8b8fa8; font-size: 0.85rem; padding: 2rem 0; text-align: center;">
                Adjust inputs and click<br><b>Log this prediction</b><br>in the sidebar to build a trend.
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════
with pages[1]:
    st.markdown("## Model Performance")
    st.markdown("All three models evaluated on a held-out chronological test set (last 20% of data, never seen during training).")
    st.markdown("---")

    st.markdown('<p class="section-header">Test Set Metrics</p>', unsafe_allow_html=True)

    # Style the metrics table
    styled = metrics_df.style\
        .format(precision=4)\
        .background_gradient(subset=['R²'], cmap='Greens')\
        .background_gradient(subset=['MAE', 'RMSE'], cmap='Reds_r')
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Key Takeaways</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    best_r2  = metrics_df['R²'].max()
    best_mae = metrics_df['MAE'].min()
    best_mod = metrics_df['R²'].idxmax()

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #4ade80">{best_r2:.3f}</div>
            <div class="metric-label">Best R² ({best_mod})</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #60a5fa">{best_mae:.3f}%</div>
            <div class="metric-label">Best MAE (% SiO₂)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        within_one = 95.1
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #a78bfa">{within_one}%</div>
            <div class="metric-label">Predictions within ±1% SiO₂</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">How this compares to the conventional approach</p>', unsafe_allow_html=True)
    st.markdown("""
    Without a soft sensor, operators estimate silica using recent lab results — effectively a rolling mean 
    of past measurements. That baseline achieves **R² ≈ 0.42** on the same test set. The Random Forest 
    soft sensor achieves **R² ≈ 0.77**, nearly doubling the explained variance. The mean absolute error 
    drops from **0.59%** to **0.41% SiO₂**.
    """)

# ════════════════════════════════════════
# PAGE 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════
with pages[2]:
    st.markdown("## Feature Importance")
    st.markdown("SHAP values computed on the XGBoost model across the full test set.")
    st.markdown("---")

    st.markdown('<p class="section-header">What drives silica variation</p>', unsafe_allow_html=True)

    # Precomputed top features from final run
    top_features = {
        '% Iron Concentrate':                      0.540,
        'Silica_lag_1':                            0.256,
        'Amina Flow':                              0.024,
        'Silica_lag_2':                            0.021,
        'Flotation Column 01 Air Flow_roll_mean3': 0.020,
        '% Silica Feed':                           0.018,
        'Flotation Column 06 Level':               0.016,
        'Ore Pulp Flow':                           0.016,
        'Flotation Column 03 Level':               0.015,
        'Starch Flow_lag1':                        0.015,
    }

    shap_df = pd.Series(top_features).sort_values()

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    fig3.patch.set_facecolor('#1a1d27')
    ax3.set_facecolor('#1a1d27')
    colors = ['#f87171' if v > 0.1 else '#60a5fa' if v > 0.02 else '#475569'
              for v in shap_df.values]
    ax3.barh(shap_df.index, shap_df.values, color=colors, edgecolor='none', height=0.6)
    ax3.set_xlabel('Mean |SHAP Value|', color='#8b8fa8', fontsize=9)
    ax3.tick_params(colors='#e8eaf0', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor('#2a2d3a')
    ax3.set_title('Top 10 Features — XGBoost SHAP Importance', color='#e8eaf0', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")
    st.markdown('<p class="section-header">Engineering Interpretation</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **% Iron Concentrate (SHAP: 0.540)**  
        Dominant predictor — iron and silica are inversely related in the concentrate.
        High iron content reliably signals low silica and vice versa.

        **Silica_lag_1 (SHAP: 0.256)**  
        Silica 2 hours ago is the second most important feature. The flotation circuit 
        has process memory — conditions don't change instantaneously between lab measurements.
        """)
    with col2:
        st.markdown("""
        **Amina Flow (SHAP: 0.024)**  
        The amine collector directly controls silica hydrophobicity and flotation efficiency. 
        Its effect is nonlinear — captured by tree models but missed by Lasso.

        **Col 01 Air Flow rolling mean (SHAP: 0.020)**  
        Average air flow over the last 3 windows captures process stability better 
        than an instantaneous reading alone.
        """)

# ════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════
with pages[3]:
    st.markdown("## About This Project")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Problem
        In iron ore froth flotation, silica (SiO₂) impurity in the final concentrate is measured 
        by lab assay every 2 hours. That delay means operators are reacting to quality problems 
        that occurred hours ago — too late to prevent off-specification product.

        ### Solution
        A data-driven soft sensor that predicts % SiO₂ in real time from continuously available 
        process sensor readings (air flows, reagent dosing, column levels, ore pulp properties), 
        replacing the slow lab feedback loop with immediate inference.

        ### Methodology
        - **Dataset**: 737,453 rows of 20-second sensor data (March–September 2017), resampled to 2-hour intervals
        - **Models**: Lasso Regression (baseline), Random Forest, XGBoost
        - **Feature engineering**: Lag features, rolling statistics, interaction terms
        - **Tuning**: Bayesian hyperparameter optimisation via Optuna
        - **Explainability**: SHAP TreeExplainer for prediction-level interpretation
        - **Validation**: Chronological 80/20 split with TimeSeriesSplit cross-validation

        ### Result
        Random Forest achieved **R² = 0.774**, **MAE = 0.414% SiO₂**, with **95.1%** of predictions 
        falling within ±1% SiO₂ of actual lab measurements.
        """)

    with col2:
        st.markdown("""
        ### Project Details
        | | |
        |---|---|
        | **Author** | Rishav Kumar |
        | **Roll No** | 230107057 |
        | **Course** | CL653 |
        | **Institute** | IIT Guwahati |
        | **Best Model** | Random Forest |
        | **R²** | 0.774 |
        | **MAE** | 0.414% SiO₂ |

        ### Data Source
        [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)  
        Kaggle — edumagalhaes

        ### Tech Stack
        `Python` `scikit-learn` `XGBoost`  
        `SHAP` `Optuna` `Streamlit`  
        `Pandas` `NumPy` `Matplotlib`
        """)
