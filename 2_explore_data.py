"""
STEP 2: Exploratory Data Analysis (EDA)
-----------------------------------------
Always look at your data before modeling. This checks for missing values,
distributions, and correlations with the target.
"""

import pandas as pd

df = pd.read_csv("data/house_prices.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)

print("\nCOLUMN TYPES & MISSING VALUES:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("SUMMARY STATISTICS:")
print(df.describe())

print("\n" + "=" * 60)
print("CORRELATION WITH PRICE (numeric features):")
numeric_df = df.select_dtypes(include="number")
print(numeric_df.corr()["price"].sort_values(ascending=False))

print("\n" + "=" * 60)
print("AVERAGE PRICE BY NEIGHBORHOOD:")
print(df.groupby("neighborhood")["price"].mean().sort_values(ascending=False))
