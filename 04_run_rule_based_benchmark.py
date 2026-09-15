print("Initializing Rule-Based Benchmark... (Loading libraries, this may take a moment)")
import pandas as pd
import os

# =============================================================================
# FROZEN GROUND-TRUTH DECISION ENGINE
# Based on: Rule_Based_Decision_Engine_Frozen_Ground_Truth.docx (v1.0)
#
# Decision Table (from client specification):
#   2+ High / High Concern findings          → DECLINE
#   Exactly 1 High/High Concern
#     OR at least 1 Review/Elevated finding  → APPROVE WITH CONDITIONS
#   No High and no Review/Elevated findings  → APPROVE
#
# Five classification rules (2026.1 policy versions):
#   POL-01: PTI (%)  — ≤20% Standard | >20–30% Review | >30% High
#   POL-02: CTI (x)  — ≤3x Standard  | >3–5x Review   | >5x High
#   POL-03: LGV (%)  — ≤100% Standard| >100–110% Rev.  | >110% High
#   POL-04: PD (%)   — <2% Low | 2–<5% Moderate | 5–<7% Elevated | ≥7% High
#   POL-05: Employment & Inquiries
#              Stable (emp ≥2yr AND low inquiries)  → Standard
#              Review (emp <2yr OR elevated inquiries) → Review
#              High Concern (very short emp AND high inquiries) → High Concern
#
# Frozen inquiry thresholds (POL-05):
#   Elevated:      Monthly > 2  OR  Quarterly > 4  OR  Yearly > 6
#   High:          Monthly > 4  AND Quarterly > 8
# =============================================================================

def classify_pti(pti):
    """POL-01 2026.1: Payment-to-Income classification."""
    if pti <= 20.0:
        return "Standard"
    elif pti <= 30.0:
        return "Review"
    else:
        return "High"

def classify_cti(cti):
    """POL-02 2026.1: Credit-to-Income classification."""
    if cti <= 3.0:
        return "Standard"
    elif cti <= 5.0:
        return "Review"
    else:
        return "High"

def classify_lgv(lgv):
    """POL-03 2026.1: Loan-to-Goods-Value classification."""
    if lgv <= 100.0:
        return "Standard"
    elif lgv <= 110.0:
        return "Review"
    else:
        return "High"

def classify_pd(pd_score):
    """POL-04 2026.1: PD Risk classification.
    Returns 'Standard'/'Review'/'High' for use in decision engine.
    (Low/Moderate → Standard for decision purposes;
     Elevated → Review; High → High)
    """
    if pd_score < 0.05:
        return "Standard"   # Low (<2%) and Moderate (2–<5%) = no exception
    elif pd_score < 0.07:
        return "Review"     # Elevated (5–<7%) = Review finding
    else:
        return "High"       # High (≥7%) = High finding

def classify_stability(emp_years, inq_mon, inq_qrt, inq_year):
    """POL-05 2026.1: Borrower Stability classification.
    Frozen inquiry thresholds:
      Elevated inquiries: monthly > 2 OR quarterly > 4 OR yearly > 6
      High inquiries:     monthly > 4 AND quarterly > 8
    """
    elevated_inq = (inq_mon > 2) or (inq_qrt > 4) or (inq_year > 6)
    high_inq     = (inq_mon > 4) and (inq_qrt > 8)
    short_emp    = emp_years < 2.0
    very_short   = emp_years < 0.5

    if very_short and high_inq:
        return "High"
    elif short_emp or elevated_inq:
        return "Review"
    else:
        return "Standard"

def evaluate_borrower(row):
    """Apply the frozen ground-truth decision engine to a single borrower row."""

    # --- Classify each policy dimension ---
    pti_class = classify_pti(row['PTI (%)'])
    cti_class = classify_cti(row['CTI (x)'])
    lgv_class = classify_lgv(row['LGV (%)'])
    pd_class  = classify_pd(row['Predicted PD'])
    stab_class = classify_stability(
        row['Employment Tenure (Years)'],
        row['Inquiries (Month)'],
        row['Inquiries (Quarter)'],
        row['Inquiries (Year)']
    )

    classifications = [pti_class, cti_class, lgv_class, pd_class, stab_class]

    # --- Apply client decision table ---
    high_count   = classifications.count("High")
    review_count = classifications.count("Review")

    if high_count >= 2:
        return "DECLINE"
    elif high_count == 1 or review_count >= 1:
        return "APPROVE WITH CONDITIONS"
    else:
        return "APPROVE"

def main():
    """Run the frozen ground-truth decision engine over the master dataset."""
    
    input_path  = "data/processed/03_master_fixed_dataset.csv"
    output_path = "data/processed/04_benchmark_results.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run step 3 first.")
        return
        
    df = pd.read_csv(input_path)
    
    print("Applying frozen ground-truth rules to all 500 borrowers...")
    df['Rule_Based_Decision'] = df.apply(evaluate_borrower, axis=1)
    
    # Save
    df.to_csv(output_path, index=False)
    
    # Summary stats
    counts = df['Rule_Based_Decision'].value_counts()
    print(f"\nDecision Distribution:")
    for decision, count in counts.items():
        print(f"  {decision}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nBenchmark complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
