"""
Pipe Pile Setup Predictor (XGBoost)
Web-based tool accompanying the paper:
"Prediction of Pipe Pile Setup Using Machine Learning and Neural Network Models
with a Web-Based Application Tool"
"""
import json

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# -----------------------------------------------------------------------------
# Model and scaler (XGBoost model of the paper; scaler fitted on the training set)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    booster = xgb.Booster()
    booster.load_model("model/xgb_model.json")
    with open("model/scaler.json") as f:
        scaler = json.load(f)
    return booster, np.array(scaler["mean"]), np.array(scaler["scale"])


booster, MEAN, SCALE = load_model()

# Ranges of the database (Table 5 of the paper): (minimum, maximum)
RANGES = {
    "Soil type": (1.0, 2.0, ""),
    "Pile diameter": (0.305, 1.067, "m"),
    "Period": (1.44, 1896.0, "hours"),
    "Effective stress σ′": (41.30, 383.63, "kPa"),
    "Q₀": (591.61, 5529.14, "kN"),
}
# Performance of the XGBoost model on the independent test set (Table 9)
TEST_R2, TEST_RMSE, TEST_MAE = 0.92, 642.57, 459.13

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Pipe Pile Setup Predictor", page_icon="🏗️", layout="centered")

st.title("Pipe Pile Setup Predictor")
st.caption("XGBoost model developed from 104 pipe pile case records")
st.write(
    "Estimate the axial capacity of a driven pipe pile after setup, **QSet (kN)**, "
    "from five input parameters. Predictions are most reliable within the ranges "
    "of the database, which are shown under each input."
)

st.subheader("Input parameters")
col1, col2 = st.columns(2)

with col1:
    soil_type = st.number_input(
        "Soil type (1 – 2)",
        min_value=1.0, max_value=2.0, value=1.5, step=0.01,
        help="1 = granular (AASHTO A-1 to A-3); 2 = silt-clay (A-4 to A-7); "
             "intermediate values = mixed profiles, based on the percentage of each "
             "soil group along the embedded pile length.",
    )
    st.caption("1 = granular · 2 = silt-clay · 1–2 = mixed")

    diameter = st.number_input(
        "Pile diameter, d (m)", min_value=0.01, value=0.61, step=0.01, format="%.3f",
    )
    st.caption("Database range: 0.305 – 1.067 m")

    period = st.number_input(
        "Period after installation, t (hours)", min_value=0.01, value=240.0, step=1.0,
    )
    st.caption(f"Database range: 1.44 – 1896 hours  ·  entered = {period / 24:.1f} days")

with col2:
    stress = st.number_input(
        "Average vertical effective stress, σ′ (kPa)",
        min_value=0.01, value=200.0, step=1.0,
        help="Average vertical effective stress over the embedded pile length.",
    )
    st.caption("Database range: 41.30 – 383.63 kPa")

    q0 = st.number_input(
        "Initial capacity at end of driving, Q₀ (kN)", min_value=0.01, value=2500.0, step=10.0,
    )
    st.caption("Database range: 591.61 – 5529.14 kN")

values = {
    "Soil type": soil_type,
    "Pile diameter": diameter,
    "Period": period,
    "Effective stress σ′": stress,
    "Q₀": q0,
}

st.write("")
if st.button("Predict QSet", type="primary", use_container_width=True):
    # Input validation: flag inputs outside the database ranges
    outside = []
    for name, val in values.items():
        lo, hi, unit = RANGES[name]
        if val < lo or val > hi:
            outside.append(f"**{name}** = {val:g} {unit} (database range {lo:g} – {hi:g} {unit})")

    if outside:
        st.warning(
            "⚠️ **Outside the range of the database.** Tree-based models do not extrapolate "
            "beyond the range of the training data, so the prediction below should be "
            "treated with caution:\n\n- " + "\n- ".join(outside)
        )

    # Prediction (same feature order and scaling as in model development)
    x = np.array([[soil_type, diameter, period, stress, q0]])
    q_set = float(booster.predict(xgb.DMatrix((x - MEAN) / SCALE))[0])

    st.subheader("Result")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted QSet", f"{q_set:,.0f} kN")
    m2.metric("Setup ratio QSet / Q₀", f"{q_set / q0:.2f}")
    m3.metric("Capacity gain", f"{q_set - q0:,.0f} kN")

    st.info(
        f"**How to interpret this result.** QSet is a point estimate. On the independent "
        f"test set, the model achieved R² = {TEST_R2:.2f}, with a mean absolute error of "
        f"{TEST_MAE:.0f} kN and a root mean square error of {TEST_RMSE:.0f} kN. "
        "Use the prediction as a design-stage estimate together with engineering judgment, "
        "and verify setup by restrike or static load tests where possible."
    )

# -----------------------------------------------------------------------------
# Sidebar: about the model
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("How to use")
    st.write(
        "1. Enter the five input parameters.\n"
        "2. Keep the inputs within the database ranges.\n"
        "3. Click **Predict QSet**.\n"
        "4. A warning appears if any input is outside the database range."
    )

    st.header("Database ranges")
    st.table(pd.DataFrame(
        [(n, f"{lo:g}", f"{hi:g}", u or "–") for n, (lo, hi, u) in RANGES.items()],
        columns=["Input", "Min", "Max", "Unit"],
    ).set_index("Input"))

    st.header("About the model")
    st.write(
        "Extreme Gradient Boosting (XGBoost) model trained on 83 of 104 pipe pile case "
        "records compiled by Tarawneh (2013) and tested on the remaining 21. "
        "Hyperparameters were selected by 5-fold cross-validation "
        "(150 trees, maximum depth 3, learning rate 0.1, subsample 0.9)."
    )
    st.caption(
        "Applicable to driven pipe piles. Not intended for other pile materials, "
        "or for soil conditions and regions outside those of the database, without further validation."
    )
