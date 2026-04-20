# app.py — Silica Soft Sensor Dashboard
# Run: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

SPEC_LIMIT = 2.0  # % SiO2 upper specification limit

@st.cache_resource
def load_artifacts():
    model   = joblib.load('xgb_silica_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    features = pd.read_csv('feature_list.csv')['feature'].tolist()
    explainer = shap.TreeExplainer(model)
    return model, scaler, features, explainer

model, scaler, FEATURES, explainer = load_artifacts()

st.title("Silica Soft Sensor Dashboard")
st.caption("Real-time % SiO₂ prediction for iron ore froth flotation")

st.sidebar.header("Enter Current Sensor Readings")
inputs = {f: st.sidebar.number_input(f, value=0.0, format="%.3f") for f in FEATURES}

X_in = np.array(list(inputs.values())).reshape(1, -1)
pred = model.predict(X_in)[0]

col1, col2 = st.columns(2)
col1.metric("Predicted % SiO₂", f"{pred:.3f}")
col2.metric("Specification Limit", f"{SPEC_LIMIT:.1f}")

if pred > SPEC_LIMIT:
    st.error(f"⚠️ ALERT: Predicted silica {pred:.3f}% exceeds spec limit {SPEC_LIMIT}%")
else:
    st.success(f"✅ Predicted silica {pred:.3f}% is within specification")

st.subheader("SHAP Explanation for This Prediction")
sv = explainer(X_in)
fig, ax = plt.subplots(figsize=(9, 5))
shap.plots.waterfall(sv[0], max_display=12, show=False)
st.pyplot(fig)