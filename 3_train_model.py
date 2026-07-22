"""
STEP 3: Preprocess data and train models
------------------------------------------
Pipeline:
  1. Split features (X) and target (y)
  2. Train/test split
  3. Preprocess: scale numeric features, one-hot encode categorical features
  4. Train several models and compare
  5. Evaluate with RMSE, MAE, R^2
  6. Save the best model + preprocessing pipeline to disk
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("data/house_prices.csv")

X = df.drop(columns=["price"])
y = df["price"]

numeric_features = [
    "area_sqft", "bedrooms", "bathrooms", "age_years",
    "distance_to_city_km", "garage_spaces", "has_garden",
    "crime_index", "school_rating",
]
categorical_features = ["neighborhood"]

# ---------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

# ---------------------------------------------------------------
# 3. Preprocessing pipeline
#    - StandardScaler for numeric columns (mean=0, std=1)
#    - OneHotEncoder for categorical columns
# ---------------------------------------------------------------
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---------------------------------------------------------------
# 4. Define candidate models
# ---------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42),
}

results = []
fitted_pipelines = {}

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    fitted_pipelines[name] = pipe

# ---------------------------------------------------------------
# 5. Compare results
# ---------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("RMSE")
print("\n" + "=" * 60)
print("MODEL COMPARISON (sorted by RMSE, lower is better)")
print("=" * 60)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["Model"]
best_pipeline = fitted_pipelines[best_model_name]
print(f"\nBest model: {best_model_name}")

# ---------------------------------------------------------------
# 6. Feature importance (if available) for the best tree-based model
# ---------------------------------------------------------------
regressor = best_pipeline.named_steps["regressor"]
if hasattr(regressor, "feature_importances_"):
    ohe = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    all_feature_names = numeric_features + cat_names
    importances = pd.Series(regressor.feature_importances_, index=all_feature_names)
    print("\nTop feature importances:")
    print(importances.sort_values(ascending=False).head(10))

# ---------------------------------------------------------------
# 7. Save the best pipeline (preprocessing + model bundled together)
# ---------------------------------------------------------------
joblib.dump(best_pipeline, "models/house_price_model.joblib")
print("\nSaved best pipeline to models/house_price_model.joblib")
