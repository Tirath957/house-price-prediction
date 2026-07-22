"""
STEP 4: Use the trained model to predict prices for new houses
------------------------------------------------------------------
This loads the saved pipeline (preprocessing + model bundled together)
and predicts prices for brand new, unseen houses.
"""

import joblib
import pandas as pd

# Load the saved pipeline
pipeline = joblib.load("models/house_price_model.joblib")

# Define new houses to predict (must have the SAME columns as training data,
# except "price")
new_houses = pd.DataFrame([
    {
    "area_sqft": 3500,
    "bedrooms": 5,
    "bathrooms": 4,
    "age_years": 1,
    "distance_to_city_km": 1.0,
    "garage_spaces": 2,
    "has_garden": 1,
    "crime_index": 10,
    "school_rating": 10,
    "neighborhood": "Downtown",
},  
    
    {
        "area_sqft": 1200,
        "bedrooms": 2,
        "bathrooms": 1,
        "age_years": 25,
        "distance_to_city_km": 20.0,
        "garage_spaces": 1,
        "has_garden": 0,
        "crime_index": 60,
        "school_rating": 4,
        "neighborhood": "Rural",
    },
])

predictions = pipeline.predict(new_houses)

for i, price in enumerate(predictions):
    print(f"House {i + 1}: Predicted price = ${price:,.0f}")
