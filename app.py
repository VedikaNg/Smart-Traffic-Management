import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("traffic_model.pkl")

# Title and description
st.title("Smart Traffic Management System")
st.write("This app predicts the average speed of vehicles based on traffic data.")

# Inputs for the model
st.header("Enter Traffic Information")
vehicle_count = st.slider("Number of Vehicles (vehicle_count)", 100, 1000, step=50)
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Snowy"])
incident = st.selectbox("Incident Type", ["None", "Accident", "Roadwork"])

# Create DataFrame from input
input_data = pd.DataFrame({
    "vehicle_count": [vehicle_count],
    "weather": [weather],
    "incident": [incident]
})

# Prediction button
if st.button("Predict Average Speed"):
    # Predict average speed
    predicted_speed = model.predict(input_data)[0]
    st.write(f"Predicted Average Speed: {predicted_speed:.2f} km/h")