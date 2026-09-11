print("Initializing PD Model Training... (Loading libraries, this may take a moment)")
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def calculate_risk_band(pd_score):
    if pd_score < 0.02:
        return 'Low risk'
    elif pd_score < 0.05:
        return 'Moderate risk'
    elif pd_score < 0.07:
        return 'Elevated risk'
    else:
        return 'High risk'

def map_features(df):
    out = pd.DataFrame()
    out["annual_income"] = df["AMT_INCOME_TOTAL"].clip(lower=0)
    out["loan_amount"] = df["AMT_CREDIT"].clip(lower=0)
    out["annuity"] = df["AMT_ANNUITY"].fillna(0).clip(lower=0)
    out["ext_source_2"] = df["EXT_SOURCE_2"].fillna(0.5)
    out["ext_source_3"] = df["EXT_SOURCE_3"].fillna(0.5)
    
    # Fill NAs
    return out.fillna(0)

def main():
    """Train PD Model and apply scoring to processed borrowers."""
    
    raw_path = "data/raw/application_train.csv"
    borrowers_path = "data/processed/01_cleaned_borrowers.csv"
    output_path = "data/processed/02_borrowers_with_pd.csv"
    
    # 1. Train Model on a subset to save time (10,000 rows)
    print("Loading training data...")
    train_df = pd.read_csv(raw_path, nrows=10500)
    # Exclude the first 500 which we are using as our experimental set
    train_df = train_df.iloc[500:]
    
    X_train = map_features(train_df)
    y_train = train_df['TARGET']
    
    print("Training Logistic Regression PD Model...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500))
    ])
    pipeline.fit(X_train, y_train)
    
    # 2. Score the 500 Borrowers
    print("Loading 500 cleaned borrowers...")
    borrowers_df = pd.read_csv(borrowers_path)
    X_borrowers = map_features(borrowers_df)
    
    print("Scoring borrowers...")
    # predict_proba returns [prob_0, prob_1]
    pd_scores = pipeline.predict_proba(X_borrowers)[:, 1]
    
    # De-fragment the dataframe before adding new columns to prevent PerformanceWarning
    borrowers_df = borrowers_df.copy()
    borrowers_df['pd_score'] = pd_scores
    borrowers_df['pd_risk_band'] = borrowers_df['pd_score'].apply(calculate_risk_band)
    
    # Save
    borrowers_df.to_csv(output_path, index=False)
    
    print(f"Scoring complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
