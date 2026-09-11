import pandas as pd
import numpy as np
import os

def main():
    """Build fixed master dataset from PD-scored borrowers."""
    print("Initializing master dataset mapping...")
    
    input_path = "data/processed/02_borrowers_with_pd.csv"
    output_path = "data/processed/03_master_fixed_dataset.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run step 2 first.")
        return
        
    df = pd.read_csv(input_path)
    master = pd.DataFrame()
    
    # 1. Applicant Identification
    master['Applicant Code'] = ["P" + str(i).zfill(3) for i in range(1, len(df)+1)]
    master['Original_SK_ID'] = df['SK_ID_CURR']
    
    # 2. Rich Context fields for the LLM
    master['Annual Income'] = df['AMT_INCOME_TOTAL']
    master['Loan Requested'] = df['AMT_CREDIT']
    master['Monthly Annuity'] = df['AMT_ANNUITY']
    master['Age (Years)'] = (df['DAYS_BIRTH'].abs() / 365.25).round(1)
    master['Employment Length (Years)'] = (df['DAYS_EMPLOYED'].abs() / 365.25).round(1)
    
    # 3. Mapped Commercial Rule Fields
    # Leverage = Total Debt / Income -> AMT_CREDIT / AMT_INCOME_TOTAL
    master['Leverage'] = (df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].replace(0, np.nan)).round(2)
    master['Leverage'] = master['Leverage'].fillna(0).clip(0, 10.0)
    
    # DSCR = Monthly Income / Monthly Payment
    monthly_income = df['AMT_INCOME_TOTAL'] / 12
    master['DSCR'] = (monthly_income / df['AMT_ANNUITY'].replace(0, np.nan)).round(2)
    master['DSCR'] = master['DSCR'].fillna(1.5).clip(0, 5.0)
    
    # LTV = Credit / Goods Price * 100
    goods_price = df['AMT_GOODS_PRICE'].fillna(df['AMT_CREDIT'])
    master['LTV'] = (df['AMT_CREDIT'] / goods_price.replace(0, np.nan) * 100).round(1)
    master['LTV'] = master['LTV'].fillna(80.0).clip(0, 150.0)
    
    # Current Ratio (Proxy using EXT_SOURCE_2 * 2.5 to simulate liquidity ratio around 1.25)
    master['Current Ratio'] = (df['EXT_SOURCE_2'].fillna(0.5) * 2.5).round(2)
    
    # Documentation (Randomized boolean for rule testing, with 95% chance of being complete)
    np.random.seed(42)
    master['Documentation Complete'] = np.random.choice([True, False], size=len(df), p=[0.95, 0.05])
    
    # 4. The Frozen PD Output
    master['Predicted PD'] = df['pd_score']
    master['PD Risk Band'] = df['pd_risk_band']
    
    # Save
    master.to_csv(output_path, index=False)
    
    print(f"Master dataset built and saved to {output_path}")

if __name__ == "__main__":
    main()
