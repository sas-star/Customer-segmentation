import pandas as pd
import numpy as np
import os
import pickle

def clean_data():
    raw_path = "/Users/croma/Desktop/unsupervised_customer_segmentation/data/Re-Optimizing-Food-Systems new.csv"
    models_dir = "/Users/croma/Desktop/unsupervised_customer_segmentation/models"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    print(f"Loading raw dataset from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    print(f"Raw dataset shape: {df.shape}")
    
    # 1. Drop duplicates
    df = df.drop_duplicates()
    
    # 2. Clean the value column (format like '4154500000000,00%')
    def parse_value(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()
        if val_str.endswith('%'):
            val_str = val_str[:-1].strip()
        # Replace comma with dot
        val_str = val_str.replace(',', '.')
        try:
            return float(val_str)
        except ValueError:
            return np.nan

    print("Cleaning 'value' column...")
    df['value'] = df['value'].apply(parse_value)
    
    # Impute missing values in 'value' with median of its category
    category_medians = df.groupby('category')['value'].transform('median')
    df['value'] = df['value'].fillna(category_medians)
    
    # If there are still NaN values, fill with overall median
    overall_median = df['value'].median()
    df['value'] = df['value'].fillna(overall_median if not pd.isna(overall_median) else 0.0)
    
    # 3. Clean 'year' column
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['year'] = df['year'].fillna(2000).astype(int)
    
    # 4. Handle other missing values
    df['element'] = df['element'].fillna('Stocks')
    df['unit'] = df['unit'].fillna('Head')
    df['country_or_area'] = df['country_or_area'].fillna('Unknown')
    df['category'] = df['category'].fillna('Unknown')
    
    # Print statistics of the cleaned values
    print("Value statistics after cleaning:")
    print(df['value'].describe())
    
    # Save clean dataset
    csv_out = os.path.join(models_dir, "cleaned_food_data.csv")
    pkl_out = os.path.join(models_dir, "cleaned_food_data.pkl")
    
    df.to_csv(csv_out, index=False)
    with open(pkl_out, "wb") as f:
        pickle.dump(df, f)
        
    print(f"Cleaned dataset saved to:\n  CSV: {csv_out}\n  Pickle: {pkl_out}")

if __name__ == "__main__":
    clean_data()
