import json
import os

import numpy as np
import streamlit as st
import xgboost as xgb

# Load the XGBoost model and the scaler (fitted on the training set)
folder = "model/" if os.path.exists("model/xgb_model.json") else ""
model = xgb.Booster()
model.load_model(folder + "xgb_model.json")
with open(folder + "scaler.json") as f:
    scaler = json.load(f)
MEAN, SCALE = np.array(scaler["mean"]), np.array(scaler["scale"])

# Ranges of the database (Table 5): (minimum, maximum)
RANGES = {
    "Soil Type": (1.0, 2.0),
    "Diameter (m)": (0.305, 1.067),
    "Period (Hours)": (1.44, 1896.0),
    "Effective Stress (kPa)": (41.30, 383.63),
    "Q0 (kN)": (591.61, 5529.14),
}

st.title("Geotechnical Prediction Web-App for Pipe Pile Setup")

st.header("Input Features")
soil_type = st.number_input("Soil Type (1 = granular, 2 = silt-clay, 1-2 = mixed)",
                            min_value=1.0, max_value=2.0, value=1.0, step=0.01)
diameter = st.number_input("Diameter (m)", min_value=0.0, value=0.6, step=0.01,
                           help="Database range: 0.305-1.067 m")
period = st.number_input("Period (Hours)", min_value=0.0, value=240.0, step=1.0,
                         help="Database range: 1.44-1896 hours")
effective_stress = st.number_input("Effective Stress (kPa)", min_value=0.0, value=200.0, step=1.0,
                                   help="Database range: 41.30-383.63 kPa")
q0 = st.number_input("Q0 (kN)", min_value=0.0, value=2500.0, step=10.0,
                     help="Database range: 591.61-5529.14 kN")

values = [soil_type, diameter, period, effective_stress, q0]

if st.button("Predict"):
    # Warning if any input is outside the range of the database
    outside = [f"{name} = {val:g} (range {lo:g}-{hi:g})"
               for (name, (lo, hi)), val in zip(RANGES.items(), values)
               if val < lo or val > hi]
    if outside:
        st.warning("Warning: these inputs are outside the range of database, "
                   "so the prediction should be treated with caution:\n\n- "
                   + "\n- ".join(outside))

    x = (np.array([values]) - MEAN) / SCALE
    prediction = float(model.predict(xgb.DMatrix(x))[0])
    st.success(f"Predicted QSet: {prediction:,.0f} kN")
    st.caption("Point estimate. Use it together with engineering judgment.")

st.sidebar.title("Instructions")
st.sidebar.write("""
1. Input the features (see the range in each help tip).
2. Click on the 'Predict' button to see the prediction.
3. A warning appears if any input is outside the database range.
""")
