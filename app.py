import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from config import BOUNDS, DEFAULTS, ENG_DEFAULTS, FEED_COLS, REAGENT_COLS, HISTORY_COLS

matplotlib.use('Agg')

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Silica Soft Sensor",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ──────────────────────────────────────────────────────
# Note: Colors are now handled natively via .streamlit/config.toml
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.metric-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600; }
.metric-label { font-size: 0.72rem; color: #8b8fa8; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.3rem; }

.warning-box {
    background: #2a1f00;
    border: 1px solid #fbbf24;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 0.8rem;
}
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    color: #8b8fa8;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    rf     = joblib.load('models/rf_silica_model.pkl')
    lasso  = joblib.load('models/lasso_silica_model.pkl')
    xgb    = joblib.load('models/xgb_silica_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    features = pd.read_csv('data/feature_list.csv')['feature'].tolist()
    metrics  = pd.read_csv('data/model_metrics.csv', index_col=0)
    return rf, lasso, xgb, scaler, features, metrics

rf_model, lasso_model, xgb_model, scaler, FEATURES, metrics_df = load_artifacts()

AIR_COLS   = [f for f in FEATURES if 'Air Flow' in f and 'lag' not in f and 'roll' not in f]
LEVEL_COLS = [f for f in FEATURES if 'Level' in f and 'lag' not in f and 'roll' not in f]

# ── Session state ────────────────────────────────────────────────
if 'trend' not in st.session_state:
    st.session_state.trend = []
if 'reset' not in st.session_state:
    st.session_state.reset = False

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.markdown("## ⚗️ Silica Soft Sensor")
st.sidebar.markdown("*Iron ore froth flotation quality prediction*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🔬 Predict", "📁 Batch Predict", "📊 Model Performance", "🔍 Feature Importance", "📋 About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size:0.7rem;color:#8b8fa8;letter-spacing:0.1em;text-transform:uppercase;">Settings</p>',
                    unsafe_allow_html=True)
SPEC_LIMIT = st.sidebar.number_input("Spec limit (% SiO₂)", min_value=0.5,
                                      max_value=5.0, value=2.0, step=0.1, format="%.1f")
st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size:0.72rem;color:#475569;">CL653 · IIT Guwahati<br>Rishav Kumar · 230107057</p>',
                    unsafe_allow_html=True)

# ── Helper: gauge chart ──────────────────────────────────────────
def make_gauge(pred, spec_limit):
    max_val = max(spec_limit * 2.5, pred * 1.2, 5.5)
    if pred > spec_limit:
        bar_color = "#f87171"
    elif pred > spec_limit * 0.9:
        bar_color = "#fbbf24"
    else:
        bar_color = "#4ade80"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred,
        number={'suffix': "%", 'font': {'size': 28, 'color': bar_color,
                                         'family': 'IBM Plex Mono'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1,
                     'tickcolor': "#8b8fa8", 'tickfont': {'color': '#8b8fa8', 'size': 10}},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "#1a1d27",
            'borderwidth': 0,
            'steps': [
                {'range': [0, spec_limit * 0.9],          'color': '#1a2e1a'},
                {'range': [spec_limit * 0.9, spec_limit],  'color': '#2a2200'},
                {'range': [spec_limit, max_val],           'color': '#2a1515'},
            ],
            'threshold': {
                'line': {'color': "#f87171", 'width': 2},
                'thickness': 0.75,
                'value': spec_limit
            }
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=20, b=10, l=20, r=20),
        paper_bgcolor='#1a1d27', font={'color': '#e8eaf0'},
    )
    return fig

# ── Helper: RF confidence interval ──────────────────────────────
def rf_confidence(model, X_in):
    tree_preds = np.array([tree.predict(X_in)[0] for tree in model.estimators_])
    return tree_preds.mean(), tree_preds.std()

# ════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════
if page == "🔬 Predict":
    st.markdown("## Silica Concentration Prediction")
    st.markdown("Adjust sensor readings on the left. All outputs update instantly.")
    st.markdown("---")

    col_inputs, col_results = st.columns([1, 1], gap="large")

    # ── LEFT: Sensor inputs with Tabs ───────────────────────────
    with col_inputs:
        inputs = {}
        out_of_range = []

        if st.button("↺ Reset to defaults", use_container_width=True):
            st.session_state.reset = True
            st.rerun()

        def render_sliders(cols):
            for col in cols:
                if col not in FEATURES:
                    continue
                lo, hi    = BOUNDS.get(col, (0.0, 1000.0))
                default   = float(DEFAULTS.get(col, (lo + hi) / 2))
                s_min     = round(float(lo) * 0.8, 2)
                s_max     = round(float(hi) * 1.2, 2)
                step      = round((s_max - s_min) / 200, 3)
                
                # Define keys for the two widgets
                sl_key = f"sl_{col}"
                ni_key = f"ni_{col}"

                # Initialize keys in session state
                if sl_key not in st.session_state or st.session_state.reset:
                    st.session_state[sl_key] = default
                if ni_key not in st.session_state or st.session_state.reset:
                    st.session_state[ni_key] = default

                # Callbacks for synchronization
                def on_slider(slider_k=sl_key, num_k=ni_key):
                    st.session_state[num_k] = st.session_state[slider_k]
                def on_number(slider_k=sl_key, num_k=ni_key):
                    st.session_state[slider_k] = st.session_state[num_k]

                st.slider(col, min_value=s_min, max_value=s_max, key=sl_key, on_change=on_slider)
                st.number_input("Manual entry", min_value=s_min, max_value=s_max, step=step, 
                                format="%.2f", key=ni_key, on_change=on_number, label_visibility="collapsed")

                inputs[col] = float(st.session_state[sl_key])
                if inputs[col] < lo or inputs[col] > hi:
                    out_of_range.append(col)
                st.markdown("<div style='margin-bottom:0.4rem'></div>", unsafe_allow_html=True)

        st.session_state.reset = False # Clear reset trigger

        tab_feed, tab_reag, tab_air, tab_lvl, tab_hist = st.tabs([
            "🪨 Feed", "🧪 Reagents", "💨 Air Flow", "📏 Levels", "🕒 Lab History"
        ])

        with tab_feed:
            st.markdown('<p class="section-header">Feed & Pulp Parameters</p>', unsafe_allow_html=True)
            render_sliders(FEED_COLS)
        with tab_reag:
            st.markdown('<p class="section-header">Chemical Reagents</p>', unsafe_allow_html=True)
            render_sliders(REAGENT_COLS)
        with tab_air:
            st.markdown('<p class="section-header">Flotation Air Flow</p>', unsafe_allow_html=True)
            render_sliders(AIR_COLS)
        with tab_lvl:
            st.markdown('<p class="section-header">Flotation Column Levels</p>', unsafe_allow_html=True)
            render_sliders(LEVEL_COLS)
        with tab_hist:
            st.markdown('<p class="section-header">Last Known Lab Results (2h Ago)</p>', unsafe_allow_html=True)
            st.info("The model uses these historical lab assays as 'anchors' to improve accuracy.")
            render_sliders(HISTORY_COLS)

    # ── RIGHT: Results ───────────────────────────────────────────
    with col_results:
        for feat in FEATURES:
            if feat not in inputs:
                inputs[feat] = ENG_DEFAULTS.get(feat, 0.0)

        X_in_raw = np.array([inputs[f] for f in FEATURES]).reshape(1, -1)
        rf_pred, rf_std = rf_confidence(rf_model, X_in_raw)
        rf_lo, rf_hi = rf_pred - 1.96 * rf_std, rf_pred + 1.96 * rf_std
        X_in_sc = scaler.transform(X_in_raw)
        lasso_pred, xgb_pred = float(lasso_model.predict(X_in_sc)[0]), float(xgb_model.predict(X_in_raw)[0])
        pred = rf_pred

        # Delta Logic
        delta_html = ""
        if st.session_state.trend:
            delta = pred - st.session_state.trend[-1]
            color = "#f87171" if delta > 0.005 else "#4ade80" if delta < -0.005 else "#8b8fa8"
            arrow = "↑" if delta > 0.005 else "↓" if delta < -0.005 else "→"
            delta_html = f"<span style='color:{color};font-size:0.9rem;margin-left:8px;'>{arrow} {abs(delta):.2f}%</span>"

        st.markdown('<p class="section-header">Predicted % SiO₂</p>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(pred, SPEC_LIMIT), use_container_width=True)

        pred_color = "#f87171" if pred > SPEC_LIMIT else "#fbbf24" if pred > SPEC_LIMIT * 0.9 else "#4ade80"
        alert_text = "⚠ ABOVE SPEC" if pred > SPEC_LIMIT else "◈ APPROACHING" if pred > SPEC_LIMIT * 0.9 else "✓ WITHIN SPEC"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card" style="border-color:{pred_color}"><div class="metric-value" style="color:{pred_color};font-size:1.0rem;padding:0.5rem 0">{alert_text} {delta_html}</div><div class="metric-label">Limit: {SPEC_LIMIT:.1f}%</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#a78bfa;font-size:1.1rem;padding:0.3rem 0">[{rf_lo:.2f}, {rf_hi:.2f}]</div><div class="metric-label">95% Conf. Interval</div></div>', unsafe_allow_html=True)

        if out_of_range:
            st.markdown(f'<div class="warning-box">⚠ Out of range: {", ".join(out_of_range)}</div>', unsafe_allow_html=True)

        st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({'Model': ['Lasso', 'Random Forest', 'XGBoost'], 'Pred': [lasso_pred, rf_pred, xgb_pred]})
        fig_comp = go.Figure(go.Bar(x=comp_df['Model'], y=comp_df['Pred'], marker_color=['#4ade80']*3, text=[f"{v:.3f}%" for v in comp_df['Pred']], textposition='outside'))
        fig_comp.add_hline(y=SPEC_LIMIT, line_dash='dash', line_color='#f87171')
        fig_comp.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor='#1a1d27', plot_bgcolor='#1a1d27', font={'color': '#e8eaf0'}, showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

        if st.button("📌 Log Prediction", use_container_width=True):
            st.session_state.trend.append(round(pred, 3))
            if len(st.session_state.trend) > 10: st.session_state.trend.pop(0)

        if len(st.session_state.trend) >= 2:
            fig_trend = go.Figure(go.Scatter(x=list(range(1, len(st.session_state.trend)+1)), y=st.session_state.trend, mode='lines+markers', line={'color': '#60a5fa'}))
            fig_trend.update_layout(height=180, margin=dict(t=20, b=20, l=10, r=10), paper_bgcolor='#1a1d27', plot_bgcolor='#1a1d27', font={'color': '#e8eaf0'})
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown('<p class="section-header">SHAP Explanation</p>', unsafe_allow_html=True)
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer(X_in_raw)
            fig_shap, _ = plt.subplots(figsize=(6, 4))
            fig_shap.patch.set_facecolor('#1a1d27')
            shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
            plt.gcf().patch.set_facecolor('#1a1d27')
            for t in plt.gcf().findobj(plt.Text): t.set_color('#e8eaf0')
            st.pyplot(fig_shap)
            plt.close()
        except: st.info("SHAP unavailable.")

# ════════════════════════════════════════
# PAGE 1.5 — BATCH PREDICT
# ════════════════════════════════════════
elif page == "📁 Batch Predict":
    st.markdown("## CSV Batch Prediction")
    st.markdown("---")
    st.markdown('<p class="section-header">1. Get the Template</p>', unsafe_allow_html=True)
    template_row = {f: ENG_DEFAULTS.get(f, DEFAULTS.get(f, 0.0)) for f in FEATURES}
    template_csv = pd.DataFrame([template_row]).to_csv(index=False).encode('utf-8')
    st.download_button("📄 Download Template", template_csv, "silica_template.csv", "text/csv")
    
    st.markdown('<p class="section-header">2. Upload & Predict</p>', unsafe_allow_html=True)
    up = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
    if up:
        df_up = pd.read_csv(up)
        df_p = df_up.copy()
        missing = [c for c in FEATURES if c not in df_p.columns]
        if missing:
            st.warning(f"⚠ Missing: {', '.join(missing)}")
            for c in missing: df_p[c] = ENG_DEFAULTS.get(c, DEFAULTS.get(c, 0.0))
        X_b = df_p[FEATURES].values
        rf_b, xgb_b = rf_model.predict(X_b), xgb_model.predict(X_b)
        lasso_b = lasso_model.predict(scaler.transform(X_b))
        res = df_up.copy()
        res.insert(0, 'RF Pred', rf_b.round(3)); res.insert(1, 'Lasso Pred', lasso_b.round(3)); res.insert(2, 'XGB Pred', xgb_b.round(3))
        st.dataframe(res.head(10), use_container_width=True)
        st.download_button("📥 Download Results", res.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")

# ════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## Model Performance")
    st.markdown("---")
    st.dataframe(metrics_df.style.format(precision=4).background_gradient(subset=['R²'], cmap='Greens'), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="metric-card"><div class="metric-value">0.608</div><div class="metric-label">Best R² (Lasso)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-value">0.502%</div><div class="metric-label">Best MAE</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-value">89.0%</div><div class="metric-label">Within ±1%</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════
# PAGE 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown("## Feature Importance")
    st.markdown("---")
    # Simplified importance visualization
    imp = {'Silica_lag_1': 0.612, 'Iron_Conc_lag1': 0.124, '% Iron Feed': 0.085, 'Amina Flow': 0.042, 'Others': 0.137}
    fig_imp = go.Figure(go.Pie(labels=list(imp.keys()), values=list(imp.values()), hole=.4))
    fig_imp.update_layout(paper_bgcolor='#1a1d27', font={'color': '#e8eaf0'})
    st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════
elif page == "📋 About":
    st.markdown("## About This Project")
    st.markdown("---")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("""
        ### Problem
        In iron ore froth flotation, silica (SiO₂) impurity in the final concentrate is measured
        by lab assay every 2 hours. That delay means operators are reacting to quality problems
        that occurred hours ago, which is too late to prevent off-specification product from reaching
        downstream blast furnace operations.

        ### Solution
        A data-driven soft sensor that predicts % SiO₂ in real time from continuously available
        process sensor readings like air flows, reagent dosing, column levels, and ore pulp
        properties. This replaces the slow lab feedback loop with immediate inference.

        ### Methodology
        - **Dataset**: 737,453 rows of 20-second sensor data (March-September 2017), resampled to 2-hour intervals to align with lab measurement cadence
        - **Models**: Lasso Regression, Random Forest, and XGBoost, compared against a rolling mean baseline
        - **Feature engineering**: Lag features (process memory anchors), rolling statistics (process stability), interaction terms (reagent chemistry)
        - **Leakage prevention**: Concurrent % Iron Concentrate excluded — arrives simultaneously with target and is unavailable at prediction time. Lagged values retained as legitimate anchors
        - **Hyperparameter tuning**: Bayesian optimisation via Optuna with TimeSeriesSplit cross-validation
        - **Explainability**: SHAP TreeExplainer for prediction-level interpretation
        - **Validation**: Strict chronological 80/20 split with no shuffling and no future data leakage

        ### Key Result
        After removing the concurrent % Iron Concentrate feature (data leakage), models were
        retrained on genuinely real-time sensor data. Lasso Regression achieved the best
        performance with **R² = 0.608**, **MAE = 0.502% SiO₂**, with **89.0%** of predictions
        falling within ±1% SiO₂ of actual lab measurements — compared to a conventional
        rolling mean baseline of R² = 0.42. The honest model confirms that process memory
        (lag features) is the dominant signal, with hardware sensors providing incremental
        real-time adjustment.
        """)

    with c2:
        st.markdown("""
        ### Project Details
        | | |
        |---|---|
        | **Author** | Rishav Kumar |
        | **Roll No** | 230107057 |
        | **Course** | CL653 |
        | **Institute** | IIT Guwahati |
        | **Best Model** | Lasso Regression |
        | **R²** | 0.608 |
        | **MAE** | 0.502% SiO₂ |
        | **Within ±1%** | 89.0% |

        ### Data Source
        [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)  
        Kaggle (edumagalhaes)

        ### Tech Stack
        `Python` `scikit-learn` `XGBoost`  
        `SHAP` `Optuna` `Streamlit` `Plotly`  
        `Pandas` `NumPy` `Matplotlib`

        ### Limitations
        - Concurrent % Iron Concentrate removed to fix data leakage — Iron Concentrate lags (2h, 4h) retained as anchors
        - % Iron Feed and % Silica Feed retained under slow-variation assumption — may need lagging in plants with rapid ore type transitions
        - Silica_lag_1 dominates predictions (SHAP 0.612) — model is anchored to the last known lab result with hardware sensors providing incremental adjustment
        - Prediction range compressed [1.12, 4.53] vs actual [0.66, 5.49] — model undersells extreme silica events
        - Static model subject to concept drift — requires retraining when ore mineralogy or sensor calibration changes
        - Trained on one plant's data — retraining needed for other sites
        """)