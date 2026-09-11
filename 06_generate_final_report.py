import pandas as pd
import os

def main():
    """Generate final output matrix and analysis summary."""
    print("Initializing report generation...")
    
    benchmark_path = "data/processed/04_benchmark_results.csv"
    experiments_path = "data/processed/05_experiment_results.csv"
    output_path = "data/processed/06_final_report_matrix.csv"
    
    if not os.path.exists(benchmark_path) or not os.path.exists(experiments_path):
        print("Error: Missing intermediate files. Run step 4 and step 5 first.")
        return
        
    bench_df = pd.read_csv(benchmark_path)
    exp_df = pd.read_csv(experiments_path)
    
    # Merge the benchmark (which contains the 'Rule Based' column) with the experiment results
    # using 'Applicant Code' as the key
    merged = pd.merge(bench_df[['Applicant Code', 'Rule Based']], exp_df, on="Applicant Code", how="left")
    
    # Rename columns to match the client's mockup
    merged = merged.rename(columns={"Applicant Code": "Borrower"})
    
    # Reorder columns to ensure exact sequence requested by the client
    cols = ['Borrower', 'Rule Based', 'EXP-001', 'EXP-002', 'EXP-003', 'EXP-004', 'EXP-005', 'EXP-006', 'EXP-007']
    
    # Keep only the requested columns
    final_df = merged[[c for c in cols if c in merged.columns]]
    # -------------------------------------------------------------
    # NEW: Calculate the "Expected Final Analysis Table" (Step 15)
    # -------------------------------------------------------------
    print("\nCalculating Final Analysis Metrics (vs EXP-001 Baseline)...")
    
    experiments = [
        ("EXP-002", "Prompt", "Prompt sensitivity"),
        ("EXP-003", "Retrieved context", "Retrieval/context sensitivity"),
        ("EXP-004", "Knowledge source/version", "Knowledge sensitivity"),
        ("EXP-005", "Model/version", "Model sensitivity"),
        ("EXP-006", "Identical repeated runs", "Nondeterministic variability"),
        ("EXP-007", "Combined runtime stress test", "System collapse check")
    ]
    
    analysis_results = []
    
    def is_material_reversal(base, exp):
        if pd.isna(base) or pd.isna(exp) or base == "ERROR" or exp == "ERROR":
            return False
        # Approve -> Decline or Decline -> Approve
        if ("APPROVE" in base and "DECLINE" in exp) or ("DECLINE" in base and "APPROVE" in exp):
            return True
        return False
        
    for exp_col, factor, interp in experiments:
        if exp_col not in final_df.columns:
            continue
            
        valid_runs = final_df[final_df[exp_col].notna() & (final_df[exp_col] != "ERROR")]
        total_runs = len(valid_runs)
        
        if total_runs == 0:
            continue
            
        decision_changes = (valid_runs['EXP-001'] != valid_runs[exp_col]).sum()
        material_reversals = valid_runs.apply(lambda r: is_material_reversal(r['EXP-001'], r[exp_col]), axis=1).sum()
        
        analysis_results.append({
            "Runtime factor": factor,
            "Runs": total_runs,
            "Decision change %": f"{(decision_changes / total_runs) * 100:.1f}%",
            "Material reversal %": f"{(material_reversals / total_runs) * 100:.1f}%",
            "Interpretation": interp
        })
        
    analysis_df = pd.DataFrame(analysis_results)
    analysis_path = "data/processed/06_final_analysis_summary.csv"
    analysis_df.to_csv(analysis_path, index=False)
    
    print(f"Final Analysis Table saved to {analysis_path}")

if __name__ == "__main__":
    main()
