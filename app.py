"""Browser interface for the Rainfall Prediction System."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rainfall_model.pkl"


@st.cache_resource
def load_model():
    """Load the saved classifier once per server session."""
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Rainfall Prediction System", page_icon="🌧️")
st.title("Rainfall Prediction System")
st.caption("Predict likely rainfall from current weather conditions.")

if not MODEL_PATH.exists():
    st.error("Saved model not found. Run `train_model.py` in VS Code first.", title="Model unavailable")
    st.stop()

artifact = load_model()

with st.form("prediction_form"):
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=60.0, value=25.0, step=0.5)
    pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1010.0, step=1.0)
    wind_speed = st.number_input("Wind speed (km/h)", min_value=0.0, max_value=150.0, value=10.0, step=1.0)
    submitted = st.form_submit_button("Predict rainfall", type="primary", width="stretch")

if submitted:
    sample = pd.DataFrame([[humidity, temperature, pressure, wind_speed]], columns=artifact["features"])
    prediction = artifact["model"].predict(sample)[0]
    probabilities = artifact["model"].predict_proba(sample)[0]
    yes_index = list(artifact["model"].classes_).index("Yes")
    rainfall_probability = probabilities[yes_index] * 100

    st.subheader("Prediction")
    if prediction == "Yes":
        st.success("Rainfall is likely.", icon="🌧️")
    else:
        st.info("Rainfall is not likely.", icon="☀️")
    st.metric("Estimated probability of rainfall", f"{rainfall_probability:.1f}%")

with st.expander("About this model"):
    st.write(f"Using: {artifact['model_name']}")
    st.caption("This is a learning project. Predictions should not be used as an official weather forecast.")
