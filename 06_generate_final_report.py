"""
06_generate_final_report.py
============================
Merges LLM experiment results with the rule-based ground truth
and writes the final evaluation Excel file.

Output columns (per row = one borrower × one experiment):
  Run                 — EXP-001, EXP-002-P2, EXP-003-K3, etc.
  What Changed        — Baseline | Prompt: Conservative | Top-K: 3 | etc.
  Applicant Code      — P001 … P500
  Retrieved Context   — POL-01,POL-02,POL-04  (which policies RAG fetched)
  LLM Decision        — APPROVE | APPROVE_WITH_CONDITIONS | DECLINE | ERROR
  Ground Truth        — APPROVE | APPROVE WITH CONDITIONS | DECLINE
  Match               — True / False
  Reasoning           — 1-sentence LLM reasoning

Two Excel sheets:
  1. Full Evaluation  — every (borrower × experiment) row
  2. Decision Matrix  — pivot: borrowers as rows, experiments as columns
"""

print("Generating final report...")
import pandas as pd
import os

EXPERIMENT_RESULTS  = "data/processed/05_experiment_results.csv"
BENCHMARK_RESULTS   = "data/processed/04_benchmark_results.csv"
OUTPUT_EXCEL        = "data/processed/06_final_evaluation_report.xlsx"


def normalise_decision(raw: str) -> str:
    """Normalise decision strings to a canonical form for comparison."""
    if not isinstance(raw, str):
        return "ERROR"
    raw = raw.strip().upper().replace("_", " ").replace("-", " ")
    if "APPROVE WITH" in raw or "CONDITIONS" in raw:
        return "APPROVE WITH CONDITIONS"
    if raw == "APPROVE":
        return "APPROVE"
    if raw == "DECLINE":
        return "DECLINE"
    return "ERROR"


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    if not os.path.exists(EXPERIMENT_RESULTS):
        print(f"ERROR: {EXPERIMENT_RESULTS} not found. Run 05_run_llm_experiments.py first.")
        return
    if not os.path.exists(BENCHMARK_RESULTS):
        print(f"ERROR: {BENCHMARK_RESULTS} not found. Run 04_run_rule_based_benchmark.py first.")
        return

    exp_df   = pd.read_csv(EXPERIMENT_RESULTS)
    bench_df = pd.read_csv(BENCHMARK_RESULTS)

    # Keep only needed benchmark columns
    ground_truth = bench_df[["Applicant Code", "Rule_Based_Decision"]].copy()
    ground_truth.columns = ["applicant_code", "Ground Truth"]

    # Merge
    merged = exp_df.merge(ground_truth, on="applicant_code", how="left")

    # ── Normalise and compare ──────────────────────────────────────────────────
    merged["LLM Decision"] = merged["recommendation"].apply(normalise_decision)
    merged["Ground Truth"] = merged["Ground Truth"].apply(normalise_decision)
    merged["Match"]        = merged["LLM Decision"] == merged["Ground Truth"]

    # ── Full Evaluation sheet ──────────────────────────────────────────────────
    full_eval = merged[[
        "exp_id",
        "what_changed",
        "applicant_code",
        "retrieved_context",
        "LLM Decision",
        "Ground Truth",
        "Match",
        "reasoning_summary",
    ]].copy()

    full_eval.columns = [
        "Run",
        "What Changed",
        "Applicant Code",
        "Retrieved Context",
        "LLM Decision",
        "Ground Truth",
        "Match",
        "Reasoning",
    ]

    # Sort for readability: by Applicant Code then Run
    full_eval = full_eval.sort_values(["Applicant Code", "Run"]).reset_index(drop=True)

    # ── Decision Matrix pivot sheet ────────────────────────────────────────────
    pivot = merged.pivot_table(
        index   = "applicant_code",
        columns = "exp_id",
        values  = "LLM Decision",
        aggfunc = "first",
    ).reset_index()

    pivot.columns.name = None
    pivot = pivot.rename(columns={"applicant_code": "Borrower"})

    # Add Ground Truth column next to borrower ID
    gt_map = ground_truth.set_index("applicant_code")["Ground Truth"].to_dict()
    pivot.insert(1, "Ground Truth", pivot["Borrower"].map(gt_map))

    # ── Write Excel with two sheets ────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        full_eval.to_excel(writer, sheet_name="Full Evaluation",  index=False)
        pivot.to_excel    (writer, sheet_name="Decision Matrix",  index=False)

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\nReport saved → {OUTPUT_EXCEL}")
    print(f"\n── Match Rate by Experiment ──────────────────────────────────")
    match_summary = full_eval[full_eval["LLM Decision"] != "ERROR"].groupby("Run")["Match"].mean()
    for run, rate in match_summary.items():
        print(f"  {run:20s}: {rate*100:.1f}% match with ground truth")

    error_summary = full_eval[full_eval["LLM Decision"] == "ERROR"].groupby("Run").size()
    if not error_summary.empty:
        print(f"\n── Error Count by Experiment ─────────────────────────────────")
        for run, count in error_summary.items():
            print(f"  {run:20s}: {count} errors")

    print(f"\nTotal rows: {len(full_eval)} | Borrowers: {full_eval['Applicant Code'].nunique()}")


if __name__ == "__main__":
    main()
