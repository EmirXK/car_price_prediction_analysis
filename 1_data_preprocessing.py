import pandas as pd

# Load
try:
    df = pd.read_csv('data/raw/used_cars.csv')
    print(f"File loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
except Exception as e:
    print(f"Load Error: {e}")
    raise

def core_diagnostic(df):
    """Fundamentals for data health and feature engineering."""
    print("\n--- Structural Metadata ---")
    df.info()

    print("\n--- Missing Values ---")
    null_counts = df.isna().sum()
    print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No missing values.")

    print(f"\n--- Duplicates: {df.duplicated().sum()} ---")

    print("\n--- High-Level Cardinality ---")
    # Crucial for identifying features that need encoding or stripping
    print(df.nunique().sort_values(ascending=False))

    print("\n--- Raw Sample ---")
    print(df.head())

core_diagnostic(df)



import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# 1. Primary Numeric Cleanup & Outlier Removal
df_clean = df.copy()
for col in ['price', 'milage']:
    df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(r'[$, mi.]', '', regex=True), errors='coerce')

# Calculate the bounds for the middle 90% of data
lower_cutoff = df_clean['price'].quantile(0.05)
upper_cutoff = df_clean['price'].quantile(0.95)

# Calculate the retention rate
retained_df = df_clean[(df_clean['price'] >= lower_cutoff) & (df_clean['price'] <= upper_cutoff)]
retention_pct = (len(retained_df) / len(df_clean)) * 100

print(f"Lower Cutoff (5th percentile): ${lower_cutoff:,.2f}")
print(f"Upper Cutoff (95th percentile): ${upper_cutoff:,.2f}")
print(f"Data Retained: {retention_pct:.2f}% ({len(retained_df)} rows)")

# Filter outliers early to reduce processing overhead
df_clean = df_clean[df_clean['price'] < upper_cutoff].copy()
df_clean = df_clean[df_clean['price'] > lower_cutoff].copy()

# 2. Feature Extraction & Imputation
# Extract engine size (liters) and handle missing categorical/numeric values
df_clean['engine_size'] = df_clean['engine'].str.extract(r'(\d+\.?\d*)').astype(float)
df_clean['engine_size'] = df_clean['engine_size'].fillna(df_clean['engine_size'].median())
df_clean['fuel_type'] = df_clean['fuel_type'].fillna(df_clean['fuel_type'].mode()[0])
df_clean['has_accident'] = np.where(df_clean['accident'].fillna('None reported') == 'None reported', 0, 1)
df_clean['car_age'] = 2026 - df_clean['model_year']
df_clean['car_text_profile'] = (
    "A " + df_clean['brand'].astype(str) + " " + 
    df_clean['model'].astype(str) + " (" + 
    df_clean['model_year'].astype(str) + "). " +
    "Engine specs: " + df_clean['engine'].astype(str) + ". " +
    "Transmission: " + df_clean['transmission'].astype(str) + ". " +
    "Colors: Exterior " + df_clean['ext_col'].astype(str) + ", Interior " + df_clean['int_col'].astype(str) + ". " +
    "Accident status: " + df_clean['accident'].fillna('None reported').astype(str)
)

# 3. Categorical Encoding & Dimensionality Reduction
cols_to_drop = ['model', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title', 'model_year']
text_profiles = df_clean['car_text_profile'].copy() # Save this array for later step processing
df_final = df_clean.drop(columns=cols_to_drop + ['car_text_profile'])

# Convert strings to indicators (int) and purge remaining NaNs
df_final = pd.get_dummies(df_final, columns=['brand', 'fuel_type'], drop_first=True, dtype=int)
df_final.dropna(inplace=True)

# 4. Feature Scaling
scaler = RobustScaler()
scale_cols = ['milage', 'car_age', 'engine_size']
df_final[scale_cols] = scaler.fit_transform(df_final[scale_cols])

# Validation
print(f"Cleanup Complete. Final Shape: {df_final.shape} | NaNs: {df_final.isna().sum().sum()}")
core_diagnostic(df_final)

import os
os.makedirs('data/processed', exist_ok=True)

# Add an explicit alignment index to link features back up later
df_final['car_id'] = range(len(df_final))
df_clean['car_id'] = range(len(df_final))

# Export to your modular pipeline
df_final.to_parquet('data/processed/base_tabular.parquet', index=False)
df_clean[['car_id', 'car_text_profile', 'price']].to_parquet('data/processed/text_profiles.parquet', index=False)
print("Data exported to data/processed/ successfully!")