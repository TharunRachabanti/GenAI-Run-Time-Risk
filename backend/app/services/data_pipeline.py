"""
Data Pipeline Service
Ingests Kaggle Home Credit dataset, trains the fixed PD model,
generates frozen baseline scores for all applicants,
and exports results to a timestamped CSV file.
"""
import asyncio
import csv
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models.orm_models import (
    Applicant, ApplicantPDScore, PDModel, Policy, ReferenceDecision
)
from app.pd.pd_model_service import PDModelService, prepare_home_credit_features
from app.policies.reference_decision_engine import ReferenceDecisionEngine
from app.policies.reference_decision_engine import CURRENT_POLICY_RULES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
console = Console()

RESULTS_DIR = Path("results")


def _export_pd_csv(rows: list):
    """Export PD results to timestamped CSV, clearing out old ones first."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Delete old pd_results_*.csv
    for old_file in RESULTS_DIR.glob("pd_results_*.csv"):
        try:
            old_file.unlink()
        except OSError:
            pass
            
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_file = RESULTS_DIR / f"pd_results_{ts}.csv"

    fieldnames = [
        "run_timestamp", "applicant_code", "applicant_name", "risk_tier",
        "annual_income", "loan_amount_requested",
        "pd_score", "pd_score_pct", "reference_decision",
        "triggered_rule", "is_boundary_case", "is_pilot",
    ]
    with open(ts_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    console.print(f"\n[bold green]OK PD results saved:[/bold green]")
    console.print(f"  Exported to : {ts_file}")
    return ts_file


class DataPipelineService:
    def __init__(self, raw_data_path: str = "data/raw/application_train.csv"):
        self.raw_data_path = Path(raw_data_path)
        self.pd_service = PDModelService(model_dir="models/pd")

    async def run_pipeline(self):
        """Execute the full data and scoring pipeline."""
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Starting pipeline. Data source: {self.raw_data_path}")

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {self.raw_data_path}")

        # 1. Load Data
        console.print("[bold blue]Loading Kaggle Home Credit dataset...[/bold blue]")
        df = pd.read_csv(self.raw_data_path)
        y = df["TARGET"]
        X = prepare_home_credit_features(df)
        logger.info(f"Loaded {len(df)} records.")

        # 2. Train PD Model
        console.print("[bold blue]Training Logistic Regression PD Model...[/bold blue]")
        validation_results = self.pd_service.train_and_validate(X, y)
        console.print(f"  ROC-AUC  : [bold]{validation_results['roc_auc']:.4f}[/bold]")
        console.print(f"  Accuracy : [bold]{validation_results['accuracy']:.4f}[/bold]")
        console.print(f"  F1 Score : [bold]{validation_results['f1_score']:.4f}[/bold]")

        # 3. Freeze Model
        self.pd_service.freeze()
        model_path = self.pd_service.save()["model_path"]
        frozen_at = self.pd_service._metadata.get("frozen_at")
        if isinstance(frozen_at, str):
            frozen_at = datetime.fromisoformat(frozen_at)

        csv_rows = []

        async with AsyncSessionLocal() as db:
            try:
                # 4. Save / fetch PDModel in DB
                stmt = select(PDModel).where(
                    PDModel.model_name == "LR-PD-HomeCredit-v1",
                    PDModel.model_version == "1.0.0"
                )
                existing = (await db.execute(stmt)).scalars().first()
                if existing:
                    db_model = existing
                else:
                    db_model = PDModel(
                        pd_model_id=uuid.uuid4(),
                        model_name="LR-PD-HomeCredit-v1",
                        model_version="1.0.0",
                        model_type="LogisticRegression",
                        training_dataset="Kaggle Home Credit Default Risk",
                        training_dataset_version="v1",
                        roc_auc=validation_results["roc_auc"],
                        brier_score=validation_results["brier_score"],
                        accuracy=validation_results["accuracy"],
                        precision=validation_results["precision"],
                        recall=validation_results["recall"],
                        f1_score=validation_results["f1_score"],
                        ks_statistic=validation_results["ks_statistic"],
                        model_artifact_path=str(model_path),
                        feature_list=validation_results["feature_list"],
                        validation_results=validation_results,
                        is_frozen=True,
                        frozen_at=frozen_at,
                    )
                    db.add(db_model)
                    await db.flush()

                # 5. Fetch policy
                result = await db.execute(
                    select(Policy).where(Policy.policy_code == "POLICY-CL-001-v3")
                )
                policy = result.scalar_one_or_none()
                if not policy:
                    raise RuntimeError("Authoritative policy 'POLICY-CL-001-v3' not found. Run setup first.")

                # 6. Clear existing PD scores and decisions (fresh overwrite)
                console.print("[bold yellow]Clearing existing PD scores (fresh run)...[/bold yellow]")
                await db.execute(delete(ReferenceDecision))
                await db.execute(delete(ApplicantPDScore))
                await db.flush()

                # 7. Fetch applicants and score them
                result = await db.execute(select(Applicant))
                applicants = result.scalars().all()
                console.print(f"[bold blue]Scoring {len(applicants)} applicants...[/bold blue]")

                engine = ReferenceDecisionEngine(CURRENT_POLICY_RULES)

                for app in applicants:
                    features = {
                        "annual_income": app.annual_income,
                        "loan_amount_requested": app.loan_amount_requested,
                        "monthly_annuity": app.monthly_annuity or 0.0,
                        "debt_to_income_ratio": app.debt_to_income_ratio,
                        "credit_to_income_ratio": app.credit_to_income_ratio or 0.0,
                        "employment_years": app.employment_years or 0.0,
                        "credit_history_years": app.credit_history_years or 0.0,
                        "delinquency_90day_count": app.delinquency_90day_count,
                        "delinquency_30day_count": app.delinquency_30day_count,
                        "external_credit_score_proxy": app.external_credit_score_proxy or 0.5,
                        "combined_loan_to_value_ratio": app.combined_loan_to_value_ratio or 80.0,
                    }
                    df_features = pd.DataFrame([features])
                    pd_prob = float(self.pd_service.pipeline.predict_proba(df_features)[0, 1])

                    pd_record = ApplicantPDScore(
                        applicant_id=app.applicant_id,
                        pd_model_id=db_model.pd_model_id,
                        pd_score=pd_prob,
                        pd_score_frozen=pd_prob,
                        is_frozen=True,
                        frozen_at=frozen_at,
                        feature_values=features,
                        score_metadata={"scored_by_pipeline": True},
                    )
                    db.add(pd_record)
                    await db.flush()

                    decision_result = engine.calculate(
                        applicant_id=str(app.applicant_id),
                        pd_score=pd_prob,
                        dti_ratio=app.debt_to_income_ratio,
                        delinquency_90day_count=app.delinquency_90day_count,
                        delinquency_30day_count=app.delinquency_30day_count,
                    )

                    ref_decision = ReferenceDecision(
                        applicant_id=app.applicant_id,
                        pd_score_record_id=pd_record.score_id,
                        policy_id=policy.policy_id,
                        pd_score=pd_prob,
                        dti_ratio=app.debt_to_income_ratio,
                        reference_decision=decision_result.reference_decision.value,
                        triggered_rule=decision_result.triggered_rule,
                        decision_explanation=decision_result.decision_explanation,
                        rule_details=[r.__dict__ for r in decision_result.rule_details],
                    )
                    db.add(ref_decision)

                    csv_rows.append({
                        "run_timestamp": run_timestamp,
                        "applicant_code": app.applicant_code,
                        "applicant_name": app.applicant_name,
                        "risk_tier": app.risk_tier,
                        "annual_income": app.annual_income,
                        "loan_amount_requested": app.loan_amount_requested,
                        "pd_score": round(pd_prob, 6),
                        "pd_score_pct": f"{pd_prob * 100:.2f}%",
                        "reference_decision": decision_result.reference_decision.value,
                        "triggered_rule": decision_result.triggered_rule,
                        "is_boundary_case": app.is_boundary_case,
                        "is_pilot": app.is_pilot,
                    })

                await db.commit()
                logger.info("Pipeline completed. PD scores and reference decisions generated.")

            except Exception as e:
                await db.rollback()
                logger.error(f"Pipeline failed: {e}")
                raise

        # 8. Print summary table
        table = Table(title="PD Results Summary", show_header=True, header_style="bold cyan")
        table.add_column("Applicant", width=20)
        table.add_column("Risk Tier", width=12)
        table.add_column("PD Score", width=10)
        table.add_column("Decision", width=16)
        table.add_column("Rule Triggered", width=25)
        for row in csv_rows:
            dec = row["reference_decision"]
            color = {"APPROVE": "green", "MANUAL_REVIEW": "yellow", "DECLINE": "red"}.get(dec, "white")
            table.add_row(
                row["applicant_code"],
                row["risk_tier"],
                row["pd_score_pct"],
                f"[{color}]{dec}[/{color}]",
                row["triggered_rule"],
            )
        console.print(table)

        # 9. Export CSV
        _export_pd_csv(csv_rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    pipeline = DataPipelineService(raw_data_path="data/raw/application_train.csv")
    asyncio.run(pipeline.run_pipeline())
