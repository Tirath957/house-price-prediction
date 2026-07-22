"""
STEP 1: Generate the dataset
-----------------------------
In a real project you'd download a CSV (e.g. from Kaggle's "House Prices" or
"California Housing" datasets). Here we generate a synthetic dataset with
realistic relationships so the whole project runs offline, end-to-end, with
no downloads needed. Swap this file out for pd.read_csv("your_data.csv")
whenever you have real data.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 2000  # number of houses

# ---- Raw features ----
area_sqft        = np.random.normal(1800, 650, N).clip(400, 6000)
bedrooms         = np.random.randint(1, 6, N)
bathrooms        = np.random.randint(1, 4, N)
age_years        = np.random.randint(0, 60, N)
distance_to_city = np.random.exponential(8, N).clip(0.2, 50)   # km
garage_spaces    = np.random.randint(0, 3, N)
has_garden       = np.random.binomial(1, 0.4, N)
crime_index      = np.random.normal(50, 20, N).clip(0, 100)    # lower = safer
school_rating    = np.random.randint(1, 11, N)                 # 1-10
neighborhood     = np.random.choice(
    ["Downtown", "Suburb", "Rural"], N, p=[0.3, 0.5, 0.2]
)

# ---- Neighborhood price multiplier (simulates real-world effect) ----
neighborhood_multiplier = {"Downtown": 1.35, "Suburb": 1.0, "Rural": 0.75}
neigh_mult = np.array([neighborhood_multiplier[n] for n in neighborhood])

# ---- Ground-truth price formula (with noise) ----
base_price = (
      area_sqft * 120
    + bedrooms * 8000
    + bathrooms * 6000
    + garage_spaces * 5000
    + has_garden * 7000
    + school_rating * 4000
    - age_years * 900
    - distance_to_city * 1500
    - crime_index * 600
)

price = base_price * neigh_mult
price += np.random.normal(0, 15000, N)   # random market noise
price = price.clip(30000, None)          # no negative/absurdly low prices

df = pd.DataFrame({
    "area_sqft": area_sqft.round(0),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_to_city_km": distance_to_city.round(2),
    "garage_spaces": garage_spaces,
    "has_garden": has_garden,
    "crime_index": crime_index.round(1),
    "school_rating": school_rating,
    "neighborhood": neighborhood,
    "price": price.round(0),
})

df.to_csv("data/house_prices.csv", index=False)
print(f"Generated data/house_prices.csv with {len(df)} rows")
print(df.head())
