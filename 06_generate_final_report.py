print("Initializing Final Report Generation... (Loading libraries, this may take a moment)")
import pandas as pd
import os

def is_material_reversal(base, exp):
    """Checks if the decision flipped between Approve and Decline (the worst case)."""
    if pd.isna(base) or pd.isna(exp) or base == "ERROR" or exp == "ERROR":
        return False
    base_approve = "APPROVE" in str(base).upper() and "DECLINE" not in str(base).upper()
    exp_decline  = "DECLINE" in str(exp).upper()
    base_decline = "DECLINE" in str(base).upper()
    exp_approve  = "APPROVE" in str(exp).upper() and "DECLINE" not in str(exp).upper()
    return (base_approve and exp_decline) or (base_decline and exp_approve)

def main():
    """Merges benchmark and experiment results and generates a final Excel report."""

    benchmark_path  = "data/processed/04_benchmark_results.csv"
    experiments_path = "data/processed/05_experiment_results.csv"
    output_excel    = "data/processed/06_final_report.xlsx"

    if not os.path.exists(benchmark_path) or not os.path.exists(experiments_path):
        print("Error: Missing intermediate files. Run step 4 and step 5 first.")
        return

    bench_df = pd.read_csv(benchmark_path)
    exp_df   = pd.read_csv(experiments_path)

    # -----------------------------------------------------------------------
    # SHEET 1: Decision Matrix
    # Borrower | Rule Based | EXP-001 | EXP-002 | ... | EXP-007
    # -----------------------------------------------------------------------
    bench_subset = bench_df[['Applicant Code', 'Rule_Based_Decision']].rename(
        columns={'Rule_Based_Decision': 'Rule Based'}
    )
    merged = pd.merge(bench_subset, exp_df, on='Applicant Code', how='left')
    merged = merged.rename(columns={'Applicant Code': 'Borrower'})

    cols = ['Borrower', 'Rule Based', 'EXP-001', 'EXP-002', 'EXP-003',
            'EXP-004', 'EXP-005', 'EXP-006', 'EXP-007']
    matrix_df = merged[[c for c in cols if c in merged.columns]]

    # -----------------------------------------------------------------------
    # SHEET 2: Final Analysis Summary
    # Runtime Factor | Runs | Decision Change % | Material Reversal % | Interpretation
    # -----------------------------------------------------------------------
    experiments_meta = [
        ("EXP-002", "Prompt wording",              "Prompt sensitivity"),
        ("EXP-003", "Retrieved context (reduced)",  "Retrieval/context sensitivity"),
        ("EXP-004", "Knowledge source/version",     "Knowledge sensitivity"),
        ("EXP-005", "Model / version",              "Model sensitivity"),
        ("EXP-006", "Identical repeated runs",       "Nondeterministic variability"),
        ("EXP-007", "Combined runtime stress test",  "System collapse check"),
    ]

    analysis_rows = []
    for exp_col, factor, interp in experiments_meta:
        if exp_col not in matrix_df.columns:
            continue
        valid = matrix_df[matrix_df[exp_col].notna() & (matrix_df[exp_col] != "ERROR")]
        total = len(valid)
        if total == 0:
            continue
        changes   = (valid['EXP-001'] != valid[exp_col]).sum()
        reversals = valid.apply(lambda r: is_material_reversal(r['EXP-001'], r[exp_col]), axis=1).sum()
        analysis_rows.append({
            "Runtime Factor":       factor,
            "Runs":                 total,
            "Decision Change %":    f"{(changes / total) * 100:.1f}%",
            "Material Reversal %":  f"{(reversals / total) * 100:.1f}%",
            "Interpretation":       interp,
        })
    analysis_df = pd.DataFrame(analysis_rows)

    # -----------------------------------------------------------------------
    # Write both sheets into one single Excel file
    # -----------------------------------------------------------------------
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        matrix_df.to_excel(writer,   sheet_name="Decision Matrix",   index=False)
        analysis_df.to_excel(writer, sheet_name="Analysis Summary",  index=False)

    print(f"\nFinal Excel Report saved to: {output_excel}")
    print(f"  Sheet 1 — Decision Matrix:  {len(matrix_df)} borrowers × {len(matrix_df.columns)} columns")
    print(f"  Sheet 2 — Analysis Summary: {len(analysis_df)} runtime factors")
    print("\n--- Preview of Decision Matrix (first 10 rows) ---")
    print(matrix_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
