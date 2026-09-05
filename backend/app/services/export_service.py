"""
Export Service
Generates research-grade output files from experiment results.
Produces timestamped CSV + latest copy + JSON aggregate summary.
"""
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

PROCESSED_DIR = Path("data/processed")  # Keeping for backwards compat if needed, but not used
RESULTS_DIR = Path("results")

def _read_latest_results() -> List[Dict]:
    """Read the latest experiment results CSV from results/."""
    # Find the single experiment_results_*.csv file
    files = list(RESULTS_DIR.glob("experiment_results_*.csv"))
    if not files:
        return []
    # If multiple exist somehow, take the most recent
    latest = sorted(files)[-1]
    with open(latest, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _compute_aggregate_summary(rows: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate metrics grouped by experiment for research paper."""
    total = len(rows)
    if not total:
        return {}

    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_exp[r.get("exp_code", "UNKNOWN")].append(r)

    experiments_summary = {}
    for exp_code, exp_rows in by_exp.items():
        n = len(exp_rows)
        agrees = [r for r in exp_rows if str(r.get("agrees_with_reference")) == "True"]
        changed = [r for r in exp_rows if str(r.get("decision_changed")) == "True"]
        pd_overrides = [r for r in exp_rows if str(r.get("pd_override_detected")) == "True"]
        policy_conflicts = [r for r in exp_rows if str(r.get("policy_conflict_flag")) == "True"]
        unsupported = [r for r in exp_rows if str(r.get("unsupported_conclusion_flag")) == "True"]

        severity_counts: Dict[str, int] = defaultdict(int)
        for r in exp_rows:
            severity_counts[r.get("severity", "UNKNOWN")] += 1

        decision_dist: Dict[str, int] = defaultdict(int)
        for r in exp_rows:
            decision_dist[r.get("llm_decision", "UNKNOWN")] += 1

        deviation_dist: Dict[str, int] = defaultdict(int)
        for r in exp_rows:
            deviation_dist[r.get("deviation_classification", "UNKNOWN")] += 1

        experiments_summary[exp_code] = {
            "experiment_name": exp_rows[0].get("exp_name", ""),
            "total_runs": n,
            "agreement_rate": round(len(agrees) / n, 4) if n else 0,
            "agreement_count": len(agrees),
            "decision_change_rate": round(len(changed) / n, 4) if n else 0,
            "decision_change_count": len(changed),
            "pd_override_rate": round(len(pd_overrides) / n, 4) if n else 0,
            "pd_override_count": len(pd_overrides),
            "policy_conflict_rate": round(len(policy_conflicts) / n, 4) if n else 0,
            "unsupported_conclusion_rate": round(len(unsupported) / n, 4) if n else 0,
            "llm_decision_distribution": dict(decision_dist),
            "severity_distribution": dict(severity_counts),
            "deviation_classification_distribution": dict(deviation_dist),
        }

    # Overall metrics
    overall_agrees = [r for r in rows if str(r.get("agrees_with_reference")) == "True"]
    overall_pd_overrides = [r for r in rows if str(r.get("pd_override_detected")) == "True"]

    return {
        "generated_at": datetime.now().isoformat(),
        "total_experiment_runs": total,
        "overall_agreement_rate": round(len(overall_agrees) / total, 4) if total else 0,
        "overall_pd_override_rate": round(len(overall_pd_overrides) / total, 4) if total else 0,
        "llm_provider": rows[0].get("llm_provider", "") if rows else "",
        "llm_model": rows[0].get("llm_model", "") if rows else "",
        "experiment_results": experiments_summary,
    }


def export_research_outputs() -> Dict[str, str]:
    """
    Export all research outputs to results/ directory, overwriting old ones.
    Returns dict of file paths created.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean up old outputs
    for old_csv in RESULTS_DIR.glob("research_output_*.csv"):
        try:
            old_csv.unlink()
        except OSError:
            pass
    for old_json in RESULTS_DIR.glob("research_summary_*.json"):
        try:
            old_json.unlink()
        except OSError:
            pass
            
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = _read_latest_results()
    if not rows:
        console.print("[red]No experiment results found. Run 'run-experiments' first.[/red]")
        return {}

    outputs = {}

    # 1. Full research CSV (all runs)
    csv_ts = RESULTS_DIR / f"research_output_{ts}.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_ts, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    outputs["research_csv"] = str(csv_ts)
    console.print(f"[bold green]OK Research CSV:[/bold green] {csv_ts}")

    # 2. Aggregate JSON summary
    summary = _compute_aggregate_summary(rows)
    json_ts = RESULTS_DIR / f"research_summary_{ts}.json"
    with open(json_ts, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    outputs["research_summary_json"] = str(json_ts)
    console.print(f"[bold green]OK Research Summary JSON:[/bold green] {json_ts}")

    console.print(f"\n[bold]All research outputs saved to:[/bold] {RESULTS_DIR.resolve()}")
    return outputs

