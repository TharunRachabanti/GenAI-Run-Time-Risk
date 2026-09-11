print("Initializing Rule-Based Benchmark... (Loading libraries, this may take a moment)")
import pandas as pd
import os

def evaluate_borrower(row):
    # Rule 1 - Incomplete
    if not row['Documentation Complete']:
        return "INCOMPLETE"
        
    # Calculate Material Exceptions
    exceptions = 0
    if row['Leverage'] > 4.50:
        exceptions += 1
    if row['DSCR'] < 1.10:
        exceptions += 1
    if row['Current Ratio'] < 1.00:
        exceptions += 1
    if row['LTV'] > 85.0:
        exceptions += 1
        
    # Rule 2 - Decline
    if row['Predicted PD'] >= 0.07 and row['Leverage'] > 4.50:
        return "DECLINE"
    if row['Predicted PD'] >= 0.07 and row['DSCR'] < 1.10:
        return "DECLINE"
    if exceptions >= 2:
        return "DECLINE"
        
    # Rule 3 - Approve with Conditions: Senior Credit Officer
    if row['Leverage'] > 4.50:
        return "APPROVE WITH CONDITIONS"
        
    # Rule 4 - Approve with Conditions: Credit Manager
    if (row['Leverage'] > 4.00 or 
        row['DSCR'] < 1.25 or 
        row['Current Ratio'] < 1.20 or 
        row['LTV'] > 75.0 or 
        row['Predicted PD'] >= 0.05):
        return "APPROVE WITH CONDITIONS"
        
    # Rule 5 - Approve
    return "APPROVE"

def main():
    """Run the strict rule-based decision engine over the master dataset."""
    
    input_path = "data/processed/03_master_fixed_dataset.csv"
    output_path = "data/processed/04_benchmark_results.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run step 3 first.")
        return
        
    df = pd.read_csv(input_path)
    
    print("Applying Benchmark Guide Rules to all 500 borrowers...")
    df['Rule Based'] = df.apply(evaluate_borrower, axis=1)
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"Benchmark complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
