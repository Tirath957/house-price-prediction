# House Price Prediction — ML Project (From Scratch)

A complete, runnable machine learning project that predicts house prices
from features like area, bedrooms, location, and more.

## Project Structure

```
house-price-prediction/
├── 1_generate_data.py   # Creates the dataset (data/house_prices.csv)
├── 2_explore_data.py    # EDA: missing values, stats, correlations
├── 3_train_model.py     # Preprocessing + trains & compares 4 models
├── 4_predict.py         # Loads the saved model, predicts new prices
├── data/
│   └── house_prices.csv
├── models/
│   └── house_price_model.joblib   # saved after training
└── README.md
```

## Requirements

```bash
pip install numpy pandas scikit-learn joblib
```

## How to run (in order)

```bash
python 1_generate_data.py   # Step 1: build the dataset
python 2_explore_data.py    # Step 2: look at the data
python 3_train_model.py     # Step 3: train & compare models, save the best
python 4_predict.py         # Step 4: predict prices for new houses
```

---

## Step-by-step explanation

### Step 1 — Get the data
Real projects usually start with `pd.read_csv("data.csv")` using a dataset
from Kaggle (search "House Prices" or "California Housing"). Here,
`1_generate_data.py` **synthesizes** a realistic dataset instead, so the
whole project runs with zero downloads. It picks features known to matter
in real estate — square footage, bedrooms, bathrooms, age, distance to
city center, crime index, school rating, neighborhood — and builds prices
from a formula plus random noise, mimicking how real housing markets work.

**To use your own data:** replace this file with your CSV and update the
column names in the other scripts to match.

### Step 2 — Explore the data (EDA)
Before touching any model, always check:
- **Missing values** — do you need to impute or drop rows?
- **Distributions** — are there outliers or skewed features?
- **Correlations** — which features actually relate to price? (Here,
  `area_sqft` turns out to be the strongest single predictor.)
- **Categorical breakdowns** — e.g. average price per neighborhood.

This tells you what preprocessing you'll need and sets expectations for
model performance.

### Step 3 — Preprocess and train models
Key ML concepts demonstrated here:

1. **Feature/target split** — `X` (inputs) vs `y` (the price to predict).
2. **Train/test split** — 80% train, 20% test, so we evaluate on data the
   model has never seen (avoids fooling yourself with overfitting).
3. **Preprocessing pipeline**:
   - `StandardScaler` — normalizes numeric features to mean 0 / std 1.
     Linear models need this; tree models don't strictly, but it doesn't
     hurt.
   - `OneHotEncoder` — turns the categorical `neighborhood` column into
     binary columns (`neighborhood_Downtown`, `neighborhood_Rural`, etc.)
     since ML models need numbers, not text.
   - Wrapping this in a `ColumnTransformer` + `Pipeline` means preprocessing
     and the model travel together — no risk of mismatched steps at
     prediction time.
4. **Model comparison** — trains 4 models so you see the tradeoffs:
   - **Linear Regression** — simplest baseline, assumes a straight-line
     relationship.
   - **Ridge Regression** — linear regression with regularization to
     reduce overfitting.
   - **Random Forest** — an ensemble of decision trees, handles
     non-linear relationships well.
   - **Gradient Boosting** — builds trees sequentially, each correcting
     the previous one's errors; often the strongest of the four.
5. **Evaluation metrics**:
   - **RMSE** (Root Mean Squared Error) — average prediction error in
     dollars, penalizes big misses more.
   - **MAE** (Mean Absolute Error) — average absolute error in dollars,
     easier to interpret.
   - **R²** — fraction of price variance the model explains (1.0 = perfect,
     0 = no better than guessing the average).
6. **Feature importance** — for tree-based models, shows which inputs
   drove the predictions most.
7. **Saving the model** — `joblib.dump()` saves the *entire pipeline*
   (preprocessing + trained model) as one file, so you never have to
   redo preprocessing logic by hand at prediction time.

### Step 4 — Predict on new data
Loads the saved pipeline and feeds it brand-new houses (never seen during
training) to get price predictions. This is the same code pattern you'd
use in a web app or API backend to serve predictions.

---

## Where to go from here

- **Swap in real data**: download a Kaggle housing dataset and point
  `pd.read_csv()` at it (update column names accordingly).
- **Hyperparameter tuning**: use `GridSearchCV` or `RandomizedSearchCV` to
  tune `n_estimators`, `max_depth`, `learning_rate`, etc.
- **Cross-validation**: use `cross_val_score` instead of a single train/test
  split for a more robust performance estimate.
- **More features**: try polynomial features, interaction terms, or
  external data (e.g. proximity to transit).
- **Try XGBoost/LightGBM**: often outperform scikit-learn's gradient
  boosting on tabular data.
- **Deploy it**: wrap `4_predict.py`'s logic in a Flask/FastAPI endpoint
  so others can query it over HTTP.
- **Build a UI**: a simple Streamlit app with sliders for each feature
  is a popular way to demo this kind of project.
