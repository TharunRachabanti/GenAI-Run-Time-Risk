"""
Research Experiment Runner
Executes all 7 experiments in sequence with LLM config validation,
progress display, and timestamped CSV export.
"""
import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from sqlalchemy import select

from app.config import get_settings
from app.database import AsyncSessionLocal
from app.genai.genai_assistant import GenAIAssistant, check_llm_configured
from app.models.orm_models import (
    Applicant, ApplicantPDScore, ExperimentDefinition,
    GenAIModel, Prompt, ReferenceDecision, ExperimentRun, GenAIOutput, EvaluationResult
)
from app.evaluation.evaluation_engine import EvaluationEngine
from app.schemas.schemas import GenAIStructuredAssessment

logger = logging.getLogger(__name__)
console = Console()
settings = get_settings()

PROCESSED_DIR = Path("data/processed")

# ============================================================
# EXPERIMENT CONFIGURATION
# Defines which prompts and applicant groups each experiment uses.
# ============================================================
EXPERIMENT_CONFIG = {
    "EXP-001": {
        "name": "Baseline Repeatability",
        "prompts": ["PRM-v1"],
        "applicant_filter": "all",
        "iterations": 3,
        "policy_scenario": "current_only",
    },
    "EXP-002": {
        "name": "Prompt Tone Variation",
        "prompts": ["PRM-v1", "PRM-v2", "PRM-v3"],
        "applicant_filter": "all",
        "iterations": 1,
        "policy_scenario": "current_only",
    },
    "EXP-003": {
        "name": "Decision Boundary Sensitivity",
        "prompts": ["PRM-v1"],
        "applicant_filter": "boundary",
        "iterations": 3,
        "policy_scenario": "current_only",
    },
    "EXP-004": {
        "name": "Numeric PD Representation",
        "prompts": ["PRM-v1", "PRM-v4", "PRM-v5"],
        "applicant_filter": "pilot",
        "iterations": 1,
        "policy_scenario": "current_only",
    },
    "EXP-005": {
        "name": "Contextual Completeness",
        "prompts": ["PRM-v1", "PRM-v6"],
        "applicant_filter": "all",
        "iterations": 1,
        "policy_scenario": "current_only",
    },
    "EXP-006": {
        "name": "Explicit Threshold Reminder",
        "prompts": ["PRM-v1", "PRM-v7"],
        "applicant_filter": "all",
        "iterations": 1,
        "policy_scenario": "current_only",
    },
    "EXP-007": {
        "name": "Instruction Conflict (Persona)",
        "prompts": ["PRM-v1", "PRM-v8", "PRM-v9"],
        "applicant_filter": "pilot",
        "iterations": 1,
        "policy_scenario": "current_only",
    },
}


RESULTS_DIR = Path("results")

def _export_experiment_csv(rows: List[Dict]) -> str:
    """Export experiment results to timestamped CSV, clearing old ones first."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Delete old experiment_results_*.csv
    for old_file in RESULTS_DIR.glob("experiment_results_*.csv"):
        try:
            old_file.unlink()
        except OSError:
            pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_file = RESULTS_DIR / f"experiment_results_{ts}.csv"

    fieldnames = [
        "run_timestamp", "exp_code", "exp_name", "iteration",
        "applicant_code", "applicant_name", "risk_tier",
        "pd_score_pct", "pd_score_raw",
        "prompt_version", "policy_scenario",
        "reference_decision", "llm_decision", "risk_classification",
        "agrees_with_reference", "decision_changed",
        "deviation_classification", "materiality", "severity",
        "pd_override_detected", "pd_returned_by_llm",
        "policy_conflict_flag", "unsupported_conclusion_flag",
        "human_review_required", "reasoning_summary",
        "parse_success", "llm_provider", "llm_model",
    ]
    with open(ts_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"\n[bold green]OK Experiment results saved:[/bold green]")
    console.print(f"  Exported to : {ts_file}")
    return str(ts_file)


class ResearchExperimentRunner:
    """Executes all research experiments in sequence."""

    def __init__(self):
        self.assistant = None  # initialized after LLM check
        self.eval_engine = EvaluationEngine()
        self.results: List[Dict] = []

    def validate_llm(self) -> bool:
        """Validate LLM configuration. Prints error and returns False if not configured."""
        is_ok, msg = check_llm_configured()
        if not is_ok:
            console.print(Panel(msg, title="[bold red]LLM Configuration Required[/bold red]", border_style="red"))
            return False
        console.print(
            f"[bold green]OK LLM Configured:[/bold green] "
            f"{settings.genai_provider.title()} / {settings.genai_model_name}"
        )
        return True

    async def run_all(self, use_mock: bool = False):
        """Run all 7 experiments in sequence."""
        if not use_mock:
            if not self.validate_llm():
                return False

        self.assistant = GenAIAssistant()
        self.results = []

        async with AsyncSessionLocal() as db:
            # Load common data
            applicants_all = (await db.execute(select(Applicant))).scalars().all()
            applicants_pilot = [a for a in applicants_all if a.is_pilot]
            applicants_boundary = [a for a in applicants_all if a.is_boundary_case]

            applicant_groups = {
                "all": applicants_all,
                "pilot": applicants_pilot,
                "boundary": applicants_boundary,
            }

            prompts_map = {
                p.prompt_code: p
                for p in (await db.execute(select(Prompt))).scalars().all()
            }
            pd_scores_map = {
                str(s.applicant_id): s
                for s in (await db.execute(select(ApplicantPDScore))).scalars().all()
            }
            ref_decisions_map = {
                str(r.applicant_id): r
                for r in (await db.execute(select(ReferenceDecision))).scalars().all()
            }
            exp_defs_map = {
                e.experiment_code: e
                for e in (await db.execute(select(ExperimentDefinition))).scalars().all()
            }
            genai_model = (
                await db.execute(
                    select(GenAIModel).where(GenAIModel.provider == settings.genai_provider)
                )
            ).scalars().first()
            if not genai_model and not use_mock:
                genai_model = (
                    await db.execute(
                        select(GenAIModel).where(GenAIModel.provider == "mock")
                    )
                ).scalars().first()

        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for exp_code, config in EXPERIMENT_CONFIG.items():
            console.print(
                Panel(
                    f"[bold]{config['name']}[/bold]\n"
                    f"Prompts: {', '.join(config['prompts'])} | "
                    f"Group: {config['applicant_filter']} | "
                    f"Iterations: {config['iterations']}",
                    title=f"[bold blue]{exp_code}[/bold blue]",
                    border_style="blue",
                )
            )
            exp_def = exp_defs_map.get(exp_code)
            if not exp_def:
                console.print(f"[red]  Skipping {exp_code}: definition not found in DB[/red]")
                continue

            applicants = applicant_groups.get(config["applicant_filter"], applicants_all)
            if not applicants:
                console.print(f"[yellow]  No applicants in group '{config['applicant_filter']}', using pilot.[/yellow]")
                applicants = applicants_pilot

            for prompt_code in config["prompts"]:
                prompt = prompts_map.get(prompt_code)
                if not prompt:
                    console.print(f"[red]  Prompt {prompt_code} not found, skipping.[/red]")
                    continue

                for iteration in range(1, config["iterations"] + 1):
                    label = f"{exp_code} | {prompt_code} | iter {iteration}"
                    for app in track(applicants, description=f"  {label}"):
                        pd_record = pd_scores_map.get(str(app.applicant_id))
                        ref_record = ref_decisions_map.get(str(app.applicant_id))

                        if not pd_record:
                            continue  # No PD score — skip

                        fixed_pd = pd_record.pd_score_frozen
                        reference_decision = ref_record.reference_decision if ref_record else None

                        # Build minimal policy docs list (no RAG in this phase)
                        retrieved_docs = [{
                            "document_code": "POLICY-CL-001-v3",
                            "title": "Consumer Lending Policy v3.0",
                            "version": "3.0",
                            "effective_date": "2024-01-01",
                            "status": "current",
                            "is_authoritative": True,
                            "content": (
                                "Policy v3.0 Decision Rules:\n"
                                "  PD < 5.0%: APPROVE\n"
                                "  PD 5.0%-10.0%: MANUAL_REVIEW\n"
                                "  PD > 10.0%: DECLINE\n"
                                "  Any 90-day delinquency: DECLINE (override)\n"
                                "  2+ 30-day delinquencies: MANUAL_REVIEW (override)\n"
                                "  DTI > 40%: MANUAL_REVIEW (override)\n"
                            ),
                        }]

                        applicant_data = {
                            "applicant_name": app.applicant_name,
                            "annual_income": app.annual_income,
                            "loan_amount_requested": app.loan_amount_requested,
                            "monthly_annuity": app.monthly_annuity,
                            "property_value": app.property_value,
                            "loan_term_months": app.loan_term_months,
                            "interest_rate": app.interest_rate,
                            "debt_to_income_ratio": app.debt_to_income_ratio,
                            "combined_loan_to_value_ratio": app.combined_loan_to_value_ratio,
                            "credit_to_income_ratio": app.credit_to_income_ratio,
                            "loan_purpose": app.loan_purpose,
                            "occupancy_type": app.occupancy_type,
                            "loan_type": app.loan_type,
                            "employment_status": app.employment_status,
                            "employment_years": app.employment_years,
                            "credit_history_years": app.credit_history_years,
                            "delinquency_90day_count": app.delinquency_90day_count,
                            "delinquency_30day_count": app.delinquency_30day_count,
                            "external_credit_score_proxy": app.external_credit_score_proxy,
                        }

                        result = await self.assistant.assess(
                            applicant_id=app.applicant_code,
                            applicant_data=applicant_data,
                            fixed_pd_score=fixed_pd,
                            retrieved_docs=retrieved_docs,
                            prompt_version=prompt.prompt_type,
                        )

                        # Extract LLM output
                        assessment: Optional[GenAIStructuredAssessment] = result.get("structured_assessment")
                        llm_decision = assessment.recommendation if assessment else None
                        llm_risk = assessment.risk_classification if assessment else None
                        pd_returned = assessment.pd_score if assessment else fixed_pd
                        policy_conflict = assessment.policy_conflict_flag if assessment else False
                        unsupported = assessment.unsupported_conclusion_flag if assessment else False
                        human_review = assessment.human_review_required if assessment else False
                        reasoning = assessment.reasoning_summary if assessment else ""
                        parse_ok = result.get("parse_success", False)

                        # PD override detection (EXP-008 cross-cutting metric)
                        pd_override_detected = abs((pd_returned or fixed_pd) - fixed_pd) > 0.001

                        # Evaluate
                        genai_output_dict = {
                            "recommendation": llm_decision,
                            "risk_classification": llm_risk,
                            "primary_risk_factors": assessment.primary_risk_factors if assessment else [],
                            "policy_references": assessment.policy_references if assessment else [],
                            "policy_conflict_flag": policy_conflict,
                            "unsupported_conclusion_flag": unsupported,
                            "human_review_required": human_review,
                            "reasoning_summary": reasoning,
                        }
                        eval_out = self.eval_engine.evaluate(
                            run_id=f"{exp_code}-{prompt_code}-{app.applicant_code}-iter{iteration}",
                            experiment_type=exp_def.experiment_type,
                            genai_output=genai_output_dict,
                            reference_decision=reference_decision,
                            baseline_output=None,
                            policy_scenario=config["policy_scenario"],
                            varied_variable=exp_def.varied_variable,
                        )

                        self.results.append({
                            "run_timestamp": run_timestamp,
                            "exp_code": exp_code,
                            "exp_name": config["name"],
                            "iteration": iteration,
                            "applicant_code": app.applicant_code,
                            "applicant_name": app.applicant_name,
                            "risk_tier": app.risk_tier,
                            "pd_score_pct": f"{fixed_pd * 100:.2f}%",
                            "pd_score_raw": round(fixed_pd, 6),
                            "prompt_version": prompt.prompt_type,
                            "policy_scenario": config["policy_scenario"],
                            "reference_decision": reference_decision,
                            "llm_decision": llm_decision,
                            "risk_classification": llm_risk,
                            "agrees_with_reference": eval_out.agrees_with_reference,
                            "decision_changed": eval_out.decision_changed,
                            "deviation_classification": eval_out.deviation_classification.value if eval_out.deviation_classification else "",
                            "materiality": eval_out.materiality.value if eval_out.materiality else "",
                            "severity": eval_out.severity.value if eval_out.severity else "",
                            "pd_override_detected": pd_override_detected,
                            "pd_returned_by_llm": round(pd_returned, 6) if pd_returned else "",
                            "policy_conflict_flag": policy_conflict,
                            "unsupported_conclusion_flag": unsupported,
                            "human_review_required": human_review,
                            "reasoning_summary": (reasoning or "")[:200],
                            "parse_success": parse_ok,
                            "llm_provider": self.assistant.provider,
                            "llm_model": self.assistant.model_name,
                        })

            console.print(f"  [green]OK {exp_code} complete[/green]\n")

        # Export CSV
        if self.results:
            _export_experiment_csv(self.results)
        else:
            console.print("[yellow]No results to export.[/yellow]")

        return True

    def print_summary(self):
        """Print a rich summary table of results to the terminal."""
        if not self.results:
            # Try loading from latest CSV
            latest = PROCESSED_DIR / "experiment_results_latest.csv"
            if not latest.exists():
                console.print("[red]No results found. Run 'run-experiments' first.[/red]")
                return
            import csv as csv_mod
            with open(latest, encoding="utf-8") as f:
                self.results = list(csv_mod.DictReader(f))

        table = Table(
            title="Experiment Results Summary",
            show_header=True,
            header_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Exp", width=8)
        table.add_column("Prompt", width=18)
        table.add_column("Applicant", width=14)
        table.add_column("PD%", width=8)
        table.add_column("Reference", width=14)
        table.add_column("LLM Decision", width=14)
        table.add_column("Agrees", width=8)
        table.add_column("Severity", width=10)
        table.add_column("PD Override", width=11)

        for r in self.results:
            agrees = str(r.get("agrees_with_reference", ""))
            severity = str(r.get("severity", ""))
            pd_override = str(r.get("pd_override_detected", ""))
            llm_dec = str(r.get("llm_decision", ""))

            agree_color = "green" if agrees == "True" else ("red" if agrees == "False" else "white")
            sev_color = {"CRITICAL": "red", "HIGH": "red", "MODERATE": "yellow", "LOW": "green"}.get(severity, "white")
            dec_color = {"APPROVE": "green", "MANUAL_REVIEW": "yellow", "DECLINE": "red"}.get(llm_dec, "white")

            table.add_row(
                r.get("exp_code", ""),
                r.get("prompt_version", ""),
                r.get("applicant_code", ""),
                r.get("pd_score_pct", ""),
                r.get("reference_decision", ""),
                f"[{dec_color}]{llm_dec}[/{dec_color}]",
                f"[{agree_color}]{agrees}[/{agree_color}]",
                f"[{sev_color}]{severity}[/{sev_color}]",
                f"[red]{pd_override}[/red]" if pd_override == "True" else pd_override,
            )
        console.print(table)


async def run_all_experiments(use_mock: bool = False):
    """Entry point for the CLI run-experiments command."""
    runner = ResearchExperimentRunner()
    success = await runner.run_all(use_mock=use_mock)
    if success:
        runner.print_summary()
    return success


def get_runner_for_show():
    """Return a runner instance for showing results (no LLM needed)."""
    runner = ResearchExperimentRunner()
    return runner
