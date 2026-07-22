"""
STEP 5: Simple web UI for house price prediction using Streamlit
Run with: streamlit run 5_app.py
"""
import streamlit as st
import pandas as pd
import joblib
import os

# Load the trained pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline = joblib.load(os.path.join(BASE_DIR, "models", "house_price_model.joblib"))

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Predictor")
st.write("Enter house details below to get a predicted price.")

# --- Input fields ---
col1, col2 = st.columns(2)

with col1:
    area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=20000, value=1500)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    age_years = st.number_input("Age (years)", min_value=0, max_value=150, value=10)
    distance_to_city_km = st.number_input("Distance to city (km)", min_value=0.0, max_value=100.0, value=5.0)

with col2:
    garage_spaces = st.number_input("Garage spaces", min_value=0, max_value=5, value=1)
    has_garden = st.selectbox("Has garden?", ["Yes", "No"])
    crime_index = st.slider("Crime index (0=low, 100=high)", 0, 100, 20)
    school_rating = st.slider("School rating (1-10)", 1, 10, 7)
    neighborhood = st.selectbox("Neighborhood", ["Downtown", "Suburb", "Rural"])  # update to match your actual categories

# --- Predict button ---
if st.button("Predict Price", type="primary"):
    new_house = pd.DataFrame([{
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "distance_to_city_km": distance_to_city_km,
        "garage_spaces": garage_spaces,
        "has_garden": 1 if has_garden == "Yes" else 0,
        "crime_index": crime_index,
        "school_rating": school_rating,
        "neighborhood": neighborhood,
    }])

    prediction = pipeline.predict(new_house)[0]
    st.success(f"### Predicted Price: ${prediction:,.0f}")