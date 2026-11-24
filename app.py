"""Simple Streamlit interface for rainfall prediction."""
import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Rainfall Prediction System", page_icon="🌧️")
st.title("🌧️ Rainfall Prediction System")
st.write("Enter weather conditions to estimate whether rainfall is likely.")

humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
temperature = st.number_input("Temperature (°C)", value=25.0)
pressure = st.number_input("Atmospheric pressure", min_value=0.0, value=1013.0)
wind_speed = st.number_input("Wind speed", min_value=0.0, value=10.0)

if st.button("Predict rainfall"):
    try:
        result = predict(humidity, temperature, pressure, wind_speed)
        st.success(f"Prediction: {result}")
        st.caption("This is a model estimate, not an official weather forecast.")
    except FileNotFoundError as error:
        st.error(str(error))
    except Exception as error:
        st.error(f"Could not make prediction: {error}")
