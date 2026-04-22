import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from config import BOUNDS, DEFAULTS, ENG_DEFAULTS, FEED_COLS, REAGENT_COLS, OTHER_COLS

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

                # Initialise BOTH widget keys in session state on first load
                if sl_key not in st.session_state:
                    st.session_state[sl_key] = default
                if ni_key not in st.session_state:
                    st.session_state[ni_key] = default

                # Callback: slider changed → copy value to number input
                def on_slider(slider_k=sl_key, num_k=ni_key):
                    st.session_state[num_k] = st.session_state[slider_k]

                # Callback: number input changed → copy value to slider
                def on_number(slider_k=sl_key, num_k=ni_key):
                    st.session_state[slider_k] = st.session_state[num_k]

                # Slider
                st.slider(
                    col,
                    min_value=s_min,
                    max_value=s_max,
                    key=sl_key,
                    on_change=on_slider,
                )

                # Number input
                st.number_input(
                    "Manual entry",
                    min_value=s_min,
                    max_value=s_max,
                    step=step,
                    format="%.2f",
                    key=ni_key,
                    on_change=on_number,
                    label_visibility="collapsed",
                )

                # Read the final value to use in your model inputs array
                inputs[col] = float(st.session_state[sl_key])
                if inputs[col] < lo or inputs[col] > hi:
                    out_of_range.append(col)

                st.markdown("<div style='margin-bottom:0.4rem'></div>", unsafe_allow_html=True)

        # Implement the Tabbed UI
        tab_feed, tab_reag, tab_air, tab_lvl, tab_out = st.tabs([
            "🪨 Feed", "🧪 Reagents", "💨 Air Flow", "📏 Levels", "🎯 Conc."
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
        with tab_out:
            st.markdown('<p class="section-header">Concentrate Properties</p>', unsafe_allow_html=True)
            render_sliders(OTHER_COLS)

    # ── RIGHT: Results ───────────────────────────────────────────
    with col_results:

        for feat in FEATURES:
            if feat not in inputs:
                inputs[feat] = ENG_DEFAULTS.get(feat, 0.0)

        X_in_raw = np.array([inputs[f] for f in FEATURES]).reshape(1, -1)

        rf_pred, rf_std = rf_confidence(rf_model, X_in_raw)
        rf_lo = rf_pred - 1.96 * rf_std
        rf_hi = rf_pred + 1.96 * rf_std

        X_in_sc = scaler.transform(X_in_raw)
        lasso_pred = float(lasso_model.predict(X_in_sc)[0])
        xgb_pred   = float(xgb_model.predict(X_in_raw)[0])

        pred = rf_pred

        if pred > SPEC_LIMIT:
            pred_color = "#f87171"
            alert_text = "⚠ ABOVE SPECIFICATION"
        elif pred > SPEC_LIMIT * 0.9:
            pred_color = "#fbbf24"
            alert_text = "◈ APPROACHING LIMIT"
        else:
            pred_color = "#4ade80"
            alert_text = "✓ WITHIN SPECIFICATION"

        # Delta Logic
        delta_html = ""
        if len(st.session_state.trend) > 0:
            last_pred = st.session_state.trend[-1]
            delta = pred - last_pred
            if delta > 0.005:
                delta_html = f"<span style='color:#f87171;font-size:0.9rem;margin-left:8px;vertical-align:middle;'>↑ +{delta:.2f}%</span>"
            elif delta < -0.005:
                delta_html = f"<span style='color:#4ade80;font-size:0.9rem;margin-left:8px;vertical-align:middle;'>↓ {delta:.2f}%</span>"
            else:
                delta_html = f"<span style='color:#8b8fa8;font-size:0.9rem;margin-left:8px;vertical-align:middle;'>→ 0.00%</span>"

        st.markdown('<p class="section-header">Predicted % SiO₂</p>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(pred, SPEC_LIMIT), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="metric-card" style="border-color:{pred_color}">
                <div class="metric-value" style="color:{pred_color};font-size:1.0rem;padding:0.5rem 0">
                    {alert_text} {delta_html}
                </div>
                <div class="metric-label">Spec limit: {SPEC_LIMIT:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#a78bfa;font-size:1.1rem;padding:0.3rem 0">
                    [{rf_lo:.2f}, {rf_hi:.2f}]
                </div>
                <div class="metric-label">95% Confidence Interval</div>
            </div>""", unsafe_allow_html=True)

        if out_of_range:
            st.markdown(f"""
            <div class="warning-box">
                ⚠ Inputs outside normal operating range — prediction may be less reliable:<br>
                {', '.join(out_of_range)}
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">Model Comparison</p>', unsafe_allow_html=True)
        comp_df = pd.DataFrame({
            'Model': ['Lasso', 'Random Forest', 'XGBoost'],
            'Prediction (% SiO₂)': [round(lasso_pred, 3), round(rf_pred, 3), round(xgb_pred, 3)]
        })
        fig_comp = go.Figure()
        colors_comp = []
        for p in [lasso_pred, rf_pred, xgb_pred]:
            if p > SPEC_LIMIT:
                colors_comp.append('#f87171')
            elif p > SPEC_LIMIT * 0.9:
                colors_comp.append('#fbbf24')
            else:
                colors_comp.append('#4ade80')

        fig_comp.add_trace(go.Bar(
            x=comp_df['Model'],
            y=comp_df['Prediction (% SiO₂)'],
            marker_color=colors_comp,
            text=[f"{v:.3f}%" for v in comp_df['Prediction (% SiO₂)']],
            textposition='outside',
            textfont={'color': '#e8eaf0', 'size': 11, 'family': 'IBM Plex Mono'},
        ))
        fig_comp.add_hline(y=SPEC_LIMIT, line_dash='dash', line_color='#f87171',
                           annotation_text=f"Spec ({SPEC_LIMIT}%)",
                           annotation_font_color='#f87171', annotation_font_size=10)
        fig_comp.update_layout(
            height=200, margin=dict(t=30, b=10, l=10, r=10),
            paper_bgcolor='#1a1d27', plot_bgcolor='#1a1d27',
            font={'color': '#e8eaf0'},
            xaxis={'tickfont': {'color': '#e8eaf0'}, 'gridcolor': '#2a2d3a'},
            yaxis={'tickfont': {'color': '#8b8fa8'}, 'gridcolor': '#2a2d3a',
                   'title': {'text': '% SiO₂', 'font': {'color': '#8b8fa8', 'size': 9}}},
            showlegend=False,
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown('<p class="section-header">Prediction Trend</p>', unsafe_allow_html=True)
        if st.button("📌 Log this prediction", use_container_width=True):
            st.session_state.trend.append(round(pred, 3))
            if len(st.session_state.trend) > 10:
                st.session_state.trend.pop(0)

        if len(st.session_state.trend) >= 2:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.trend) + 1)),
                y=st.session_state.trend,
                mode='lines+markers',
                line={'color': '#60a5fa', 'width': 2},
                marker={'size': 6, 'color': '#60a5fa'},
                name='Silica %'
            ))
            fig_trend.add_hline(y=SPEC_LIMIT, line_dash='dash', line_color='#f87171',
                                annotation_text=f"Spec ({SPEC_LIMIT}%)",
                                annotation_font_color='#f87171', annotation_font_size=9)
            fig_trend.update_layout(
                height=180, margin=dict(t=20, b=20, l=10, r=10),
                paper_bgcolor='#1a1d27', plot_bgcolor='#1a1d27',
                font={'color': '#e8eaf0'},
                xaxis={'title': {'text': 'Reading', 'font': {'size': 9, 'color': '#8b8fa8'}},
                       'tickfont': {'color': '#8b8fa8'}, 'gridcolor': '#2a2d3a'},
                yaxis={'title': {'text': '% SiO₂', 'font': {'size': 9, 'color': '#8b8fa8'}},
                       'tickfont': {'color': '#8b8fa8'}, 'gridcolor': '#2a2d3a'},
                showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.markdown("""
            <div style="color:#8b8fa8;font-size:0.82rem;padding:0.8rem 0;text-align:center;">
                Log 2+ predictions to see the trend chart.
            </div>""", unsafe_allow_html=True)

        st.markdown('<p class="section-header">SHAP — What drove this prediction</p>',
                    unsafe_allow_html=True)
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_vals = explainer(X_in_raw)
            fig_shap, ax = plt.subplots(figsize=(6, 4))
            fig_shap.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')
            shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
            plt.gcf().patch.set_facecolor('#1a1d27')
            for text in plt.gcf().findobj(plt.Text):
                text.set_color('#e8eaf0')
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()
        except Exception:
            st.info("SHAP explanation unavailable — check model file.")

# ════════════════════════════════════════
# PAGE 1.5 — BATCH PREDICT
# ════════════════════════════════════════
elif page == "📁 Batch Predict":
    st.markdown("## CSV Batch Prediction")
    st.markdown("Upload a CSV file containing historical sensor readings to generate bulk predictions.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV file (must contain process sensor columns)", type=['csv'])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✓ File loaded successfully: {len(df_upload)} rows found.")
            
            df_process = df_upload.copy()
            
            missing_cols = [col for col in FEATURES if col not in df_process.columns]
            if missing_cols:
                st.warning(f"⚠ Missing {len(missing_cols)} required columns. Filling with default values to allow prediction.")
                for col in missing_cols:
                    df_process[col] = ENG_DEFAULTS.get(col, DEFAULTS.get(col, 0.0))
            
            X_batch = df_process[FEATURES].values
            
            with st.spinner("Generating predictions..."):
                rf_preds   = rf_model.predict(X_batch)
                xgb_preds  = xgb_model.predict(X_batch)
                
                X_batch_sc = scaler.transform(X_batch)
                lasso_preds = lasso_model.predict(X_batch_sc)
                
            results_df = df_upload.copy()
            results_df.insert(0, 'Predicted % SiO₂ (XGBoost)', xgb_preds)
            results_df.insert(0, 'Predicted % SiO₂ (Lasso)', lasso_preds)
            results_df.insert(0, 'Predicted % SiO₂ (Random Forest)', rf_preds)
            
            results_df['Predicted % SiO₂ (Random Forest)'] = results_df['Predicted % SiO₂ (Random Forest)'].round(3)
            results_df['Predicted % SiO₂ (Lasso)'] = results_df['Predicted % SiO₂ (Lasso)'].round(3)
            results_df['Predicted % SiO₂ (XGBoost)'] = results_df['Predicted % SiO₂ (XGBoost)'].round(3)

            st.markdown('<p class="section-header">Prediction Preview</p>', unsafe_allow_html=True)
            st.dataframe(results_df.head(15), use_container_width=True)

            csv_export = results_df.to_csv(index=False).encode('utf-8')
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Complete Results",
                data=csv_export,
                file_name="silica_batch_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure the uploaded file is a valid CSV formatted with the correct sensor column names.")

# ════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## Model Performance")
    st.markdown("All models evaluated on a held-out chronological test set — last 20% of data, never seen during training.")
    st.markdown("---")

    st.markdown('<p class="section-header">Test Set Metrics</p>', unsafe_allow_html=True)
    styled = metrics_df.style\
        .format(precision=4)\
        .background_gradient(subset=['R²'], cmap='Greens')\
        .background_gradient(subset=['MAE', 'RMSE'], cmap='Reds_r')
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Key Takeaways</p>', unsafe_allow_html=True)

    best_r2  = metrics_df['R²'].max()
    best_mae = metrics_df['MAE'].min()
    best_mod = metrics_df['R²'].idxmax()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#4ade80">{best_r2:.3f}</div>
            <div class="metric-label">Best R² ({best_mod})</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#60a5fa">{best_mae:.3f}%</div>
            <div class="metric-label">Best MAE (% SiO₂)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#a78bfa">95.6%</div>
            <div class="metric-label">Within ±1% SiO₂</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Comparison against conventional approach</p>',
                unsafe_allow_html=True)
    st.markdown(f"""
    Without a soft sensor, operators estimate silica using a rolling mean of recent lab results.
    That baseline achieves **R² ≈ 0.42** on the same test set. The best ML model ({best_mod})
    achieves **R² = {best_r2:.3f}**, nearly doubling the explained variance. Mean absolute error
    drops from **0.59%** to **{best_mae:.3f}% SiO₂**.
    """)

# ════════════════════════════════════════
# PAGE 3 — FEATURE IMPORTANCE
# ════════════════════════════════════════
elif page == "🔍 Feature Importance":
    st.markdown("## Feature Importance")
    st.markdown("SHAP values computed on the XGBoost model across the full test set.")
    st.markdown("---")

    st.markdown('<p class="section-header">What drives silica variation</p>', unsafe_allow_html=True)

    top_features = {
        '% Iron Concentrate':                      0.590,
        'Silica_lag_1':                            0.333,
        'Iron_Concentrate_lag1':                   0.096,
        'Iron_Concentrate_lag2':                   0.037,
        'Silica_lag_2':                            0.029,
        'Flotation Column 01 Air Flow_roll_mean3': 0.016,
        'Ore Pulp Flow':                           0.018,
        'Flotation Column 06 Level':               0.018,
        'Flotation Column 03 Level':               0.015,
        'Silica_delta':                            0.017,
    }

    shap_series = pd.Series(top_features).sort_values()
    colors = ['#f87171' if v > 0.1 else '#60a5fa' if v > 0.02 else '#475569'
              for v in shap_series.values]

    fig_shap = go.Figure(go.Bar(
        x=shap_series.values,
        y=shap_series.index,
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in shap_series.values],
        textposition='outside',
        textfont={'color': '#e8eaf0', 'size': 10},
    ))
    fig_shap.update_layout(
        height=380, margin=dict(t=20, b=20, l=10, r=60),
        paper_bgcolor='#1a1d27', plot_bgcolor='#1a1d27',
        font={'color': '#e8eaf0'},
        xaxis={'title': {'text': 'Mean |SHAP Value|', 'font': {'color': '#8b8fa8', 'size': 10}},
               'tickfont': {'color': '#8b8fa8'}, 'gridcolor': '#2a2d3a'},
        yaxis={'tickfont': {'color': '#e8eaf0', 'size': 10}, 'gridcolor': '#2a2d3a'},
        title={'text': 'Top 10 Features — XGBoost SHAP Importance',
               'font': {'color': '#e8eaf0', 'size': 12, 'family': 'IBM Plex Mono'}},
        showlegend=False,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-header">Engineering Interpretation</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **% Iron Concentrate (SHAP: 0.590)** Dominant predictor — iron and silica are inversely related in the concentrate.
        High iron content reliably signals low silica and vice versa.

        **Silica_lag_1 (SHAP: 0.333)** Silica 2 hours ago is the second most important feature. The flotation circuit
        has process memory — conditions don't change instantaneously between measurements.

        **Iron_Concentrate_lag1 (SHAP: 0.096)** Iron concentrate trend matters as much as its current value. A falling iron
        reading predicts rising silica ahead.
        """)
    with c2:
        st.markdown("""
        **Iron_Concentrate_lag2 (SHAP: 0.037)** Extending the iron trend window to 4 hours back captures slower shift-level
        process changes that lag_1 alone misses.

        **Silica_lag_2 (SHAP: 0.029)** Silica 4 hours ago helps distinguish sustained trends from transient spikes,
        giving the model longer-range process context.

        **Col 01 Air Flow rolling mean (SHAP: 0.016)** Average air flow over the last 3 windows captures process stability better
        than an instantaneous reading alone.
        """)

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
        - **Feature engineering**: Lag features (process memory), rolling statistics (process stability), interaction terms (reagent chemistry)
        - **Hyperparameter tuning**: Bayesian optimisation via Optuna with TimeSeriesSplit cross-validation
        - **Explainability**: SHAP TreeExplainer for prediction-level interpretation
        - **Validation**: Strict chronological 80/20 split with no shuffling and no future data leakage

        ### Key Result
        Lasso Regression achieved the best performance with **R² = 0.825**, **MAE = 0.347% SiO₂**,
        with **95.6%** of predictions falling within ±1% SiO₂ of actual lab measurements,
        compared to a conventional rolling mean baseline of R² = 0.42.
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
        | **R²** | 0.825 |
        | **MAE** | 0.347% SiO₂ |
        | **Within ±1%** | 95.6% |

        ### Data Source
        [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process)  
        Kaggle (edumagalhaes)

        ### Tech Stack
        `Python` `scikit-learn` `XGBoost`  
        `SHAP` `Optuna` `Streamlit` `Plotly`  
        `Pandas` `NumPy` `Matplotlib`

        ### Limitations
        - % Iron Concentrate is a lab assay, so in true real-time deployment it would need its own soft sensor
        - Residual underprediction bias of ~0.08% SiO₂
        - Trained on one plant's data, meaning retraining is needed for other sites
        """)
