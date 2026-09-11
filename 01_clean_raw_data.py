import pandas as pd
import os

def main():
    """Extract top 500 borrowers from raw Kaggle dataset."""
    print("Initializing raw dataset extraction...")
    
    raw_path = "data/raw/application_train.csv"
    processed_dir = "data/processed"
    
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return
        
    os.makedirs(processed_dir, exist_ok=True)
    
    # Read raw data
    df = pd.read_csv(raw_path)
    
    # We want 500 clean rows. We'll drop rows with missing values in crucial columns
    crucial_cols = [
        'SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 
        'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    
    df_clean = df.dropna(subset=crucial_cols).copy()
    
    # Take top 500
    df_500 = df_clean.head(500)
    
    output_path = os.path.join(processed_dir, "01_cleaned_borrowers.csv")
    df_500.to_csv(output_path, index=False)
    
    print(f"Successfully processed {len(df_500)} borrowers.")

if __name__ == "__main__":
    main()
