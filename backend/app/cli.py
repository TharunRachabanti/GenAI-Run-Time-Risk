"""
GenAI Runtime Risk Research Platform — CLI
==========================================
Step-by-step research pipeline commands.

STEP-BY-STEP MODE (run one at a time):
  python cli.py setup
  python cli.py train-pd
  python cli.py show-pd-results
  python cli.py build-rag
  python cli.py run-experiments
  python cli.py show-results
  python cli.py export

FULL-AUTO MODE (single command, no intervention needed):
  python cli.py run-all

STATUS:
  python cli.py status
"""
import asyncio
import logging
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()
logging.basicConfig(level=logging.WARNING)  # Suppress verbose SQLAlchemy output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


@click.group()
def cli():
    """GenAI Runtime Risk Research Platform CLI."""
    pass


# ============================================================
# STEP 1: SETUP
# ============================================================
@cli.command()
def setup():
    """[STEP 1] Initialize database and seed all configuration data."""
    console.print(Panel(
        "[bold blue]Initializing database and seeding configuration...[/bold blue]",
        title="STEP 1 — Setup", border_style="blue"
    ))
    from app.services.seed_service import setup_db
    asyncio.run(setup_db())
    console.print("[bold green]OK Setup complete.[/bold green]")
    console.print("  Next: [bold]python cli.py train-pd[/bold]")


# ============================================================
# STEP 2: TRAIN PD
# ============================================================
@cli.command("train-pd")
def train_pd():
    """[STEP 2] Train the Logistic Regression PD model and score all applicants."""
    console.print(Panel(
        "[bold blue]Training PD model and generating applicant scores...[/bold blue]\n"
        "Results will be saved to [bold]results/pd_results_[timestamp].csv[/bold]",
        title="STEP 2 — Train PD Model", border_style="blue"
    ))
    from app.services.data_pipeline import DataPipelineService
    pipeline = DataPipelineService(raw_data_path="data/raw/application_train.csv")
    asyncio.run(pipeline.run_pipeline())
    console.print("\n[bold green]OK PD training complete.[/bold green]")
    console.print("  Next: [bold]python cli.py show-pd-results[/bold]")


# ============================================================
# STEP 3: SHOW PD RESULTS
# ============================================================
@cli.command("show-pd-results")
def show_pd_results():
    """[STEP 3] Show PD model metrics and all applicant PD scores from database."""
    import csv as csv_mod
    from app.database import AsyncSessionLocal
    from sqlalchemy import select
    from app.models.orm_models import PDModel

    async def _show():
        async with AsyncSessionLocal() as db:
            model = (
                await db.execute(
                    select(PDModel).order_by(PDModel.created_at.desc()).limit(1)
                )
            ).scalars().first()
            if model:
                console.print(Panel(
                    f"[bold]Model:[/bold]    {model.model_name} v{model.model_version}\n"
                    f"[bold]ROC AUC:[/bold]  {model.roc_auc:.4f}\n"
                    f"[bold]Accuracy:[/bold] {model.accuracy:.4f}\n"
                    f"[bold]F1 Score:[/bold] {model.f1_score:.4f}\n"
                    f"[bold]Frozen:[/bold]   {'Yes OK' if model.is_frozen else 'No'}",
                    title="PD Model Metrics", border_style="green"
                ))
            else:
                console.print("[yellow]No model found. Run 'train-pd' first.[/yellow]")
                return

    asyncio.run(_show())

    # Show CSV results
    results_dir = Path("results")
    pd_files = list(results_dir.glob("pd_results_*.csv"))
    if pd_files:
        # Sort to get the latest timestamp
        pd_latest = sorted(pd_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        table = Table(title=f"Applicant PD Scores (from {pd_latest.name})",
                      header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Code", width=12)
        table.add_column("Name", width=20)
        table.add_column("Tier", width=12)
        table.add_column("PD Score", width=10)
        table.add_column("Decision", width=14)
        table.add_column("Rule", width=25)
        table.add_column("Boundary", width=9)

        with open(pd_latest, encoding="utf-8") as f:
            for row in csv_mod.DictReader(f):
                dec = row.get("reference_decision", "")
                color = {"APPROVE": "green", "MANUAL_REVIEW": "yellow", "DECLINE": "red"}.get(dec, "white")
                table.add_row(
                    row.get("applicant_code", ""),
                    row.get("applicant_name", "")[:18],
                    row.get("risk_tier", ""),
                    row.get("pd_score_pct", ""),
                    f"[{color}]{dec}[/{color}]",
                    row.get("triggered_rule", "")[:23],
                    "OK" if row.get("is_boundary_case") == "True" else "",
                )
        console.print(table)
        console.print(f"[dim]File: {pd_latest.resolve()}[/dim]")
    else:
        console.print("[yellow]CSV file not found in results folder. Run 'train-pd' first.[/yellow]")

    console.print("\n  Next: [bold]python cli.py build-rag[/bold]")


# ============================================================
# STEP 4: BUILD RAG
# ============================================================
@cli.command("build-rag")
def build_rag():
    """[STEP 4] Index policy documents into the ChromaDB vector store."""
    console.print(Panel(
        "[bold blue]Indexing policy documents into ChromaDB...[/bold blue]",
        title="STEP 4 — Build RAG Index", border_style="blue"
    ))

    async def _build():
        from sqlalchemy import select
        from app.database import AsyncSessionLocal
        from app.models.orm_models import PolicyDocument
        from app.config import get_settings
        settings = get_settings()

        async with AsyncSessionLocal() as db:
            docs = (await db.execute(select(PolicyDocument))).scalars().all()

        if not docs:
            console.print("[yellow]No policy documents found in DB. Run 'setup' first.[/yellow]")
            return

        try:
            from app.rag.rag_service import RAGService
            rag = RAGService(
                persist_directory=settings.chroma_persist_directory,
                embedding_model_name=settings.embedding_model,
                top_k=settings.rag_top_k,
            )
            doc_dicts = [
                {
                    "document_code": d.document_code,
                    "title": d.title,
                    "version": d.version,
                    "status": d.status,
                    "is_authoritative": d.is_authoritative,
                    "effective_date": str(d.effective_date) if d.effective_date else "",
                    "content": d.content or d.summary or "",
                }
                for d in docs
            ]
            rag.index_documents(doc_dicts)
            console.print(f"[bold green]OK Indexed {len(docs)} policy documents into ChromaDB.[/bold green]")
        except Exception as e:
            console.print(f"[yellow]RAG indexing skipped: {e}[/yellow]")
            console.print("[dim]This is non-critical. Experiments will use inline policy context.[/dim]")

    asyncio.run(_build())
    console.print("  Next: [bold]python cli.py run-experiments[/bold]")


# ============================================================
# STEP 5: RUN EXPERIMENTS
# ============================================================
@cli.command("run-experiments")
@click.option("--mock", is_flag=True, default=False,
              help="Use mock LLM for testing (no API key required).")
def run_experiments(mock: bool):
    """[STEP 5] Run all 7 research experiments against the configured LLM."""
    if mock:
        console.print(Panel(
            "[yellow]Running in MOCK mode (no real LLM calls).[/yellow]\n"
            "Results will use the built-in mock adapter for testing.",
            title="STEP 5 — Run Experiments (MOCK)", border_style="yellow"
        ))
    else:
        console.print(Panel(
            "[bold blue]Running all 7 research experiments...[/bold blue]\n"
            "Results will be saved to [bold]data/processed/experiment_results_latest.csv[/bold]",
            title="STEP 5 — Run Experiments", border_style="blue"
        ))

    from app.services.cli_experiment_runner import run_all_experiments
    success = asyncio.run(run_all_experiments(use_mock=mock))

    if success:
        console.print("\n[bold green]OK All experiments complete.[/bold green]")
        console.print("  Next: [bold]python cli.py show-results[/bold]")
    else:
        console.print("\n[bold red]FAIL Experiments did not complete. Check LLM configuration.[/bold red]")


# ============================================================
# STEP 6: SHOW RESULTS
# ============================================================
@cli.command("show-results")
def show_results():
    """[STEP 6] Display a summary table of all experiment results."""
    from app.services.cli_experiment_runner import get_runner_for_show
    runner = get_runner_for_show()
    runner.print_summary()
    console.print("\n  Next: [bold]python cli.py export[/bold]")


# ============================================================
# STEP 7: EXPORT
# ============================================================
@cli.command()
def export():
    """[STEP 7] Export final research-grade CSV and JSON outputs to results/."""
    console.print(Panel(
        "[bold blue]Generating final research outputs...[/bold blue]",
        title="STEP 7 — Export", border_style="blue"
    ))
    from app.services.export_service import export_research_outputs
    outputs = export_research_outputs()
    if outputs:
        console.print("\n[bold green]OK Export complete. Files ready for research analysis.[/bold green]")
    console.print("  Done! Your research outputs are in: [bold]results/[/bold]")


# ============================================================
# FULL-AUTO: RUN ALL
# ============================================================
@cli.command("run-all")
@click.option("--mock", is_flag=True, default=False,
              help="Use mock LLM for testing (no API key required).")
def run_all(mock: bool):
    """[AUTO] Run the complete pipeline: setup → train-pd → build-rag → run-experiments → export."""
    console.print(Panel(
        "[bold blue]GenAI Runtime Risk — Full Pipeline[/bold blue]\n\n"
        "This will execute all steps automatically:\n"
        "  Step 1: Setup (DB init + seed)\n"
        "  Step 2: Train PD Model\n"
        "  Step 3: Build RAG Index\n"
        "  Step 4: Run All Experiments\n"
        "  Step 5: Export Results\n\n"
        + ("[yellow]Running in MOCK mode.[/yellow]" if mock else
           "[bold]Checking LLM configuration first...[/bold]"),
        title="[bold cyan]FULL AUTO RUN[/bold cyan]",
        border_style="cyan",
    ))

    # Pre-flight LLM check (before doing any work)
    if not mock:
        from app.genai.genai_assistant import check_llm_configured
        from rich.panel import Panel as P
        is_ok, msg = check_llm_configured()
        if not is_ok:
            console.print(Panel(msg, title="[bold red]LLM Configuration Required[/bold red]", border_style="red"))
            console.print("[bold red]FAIL Pipeline aborted. Configure LLM in .env and run again.[/bold red]")
            return

    steps = [
        ("Step 1/5 — Setup", _step_setup),
        ("Step 2/5 — Train PD", _step_train_pd),
        ("Step 3/5 — Build RAG", _step_build_rag),
        (f"Step 4/5 — Run Experiments {'(mock)' if mock else ''}", lambda: _step_run_experiments(mock)),
        ("Step 5/5 — Export", _step_export),
    ]

    for label, fn in steps:
        console.print(f"\n[bold cyan]--- {label} ---[/bold cyan]")
        try:
            fn()
        except Exception as e:
            console.print(f"[bold red]FAIL Failed at {label}: {e}[/bold red]")
            console.print("[yellow]Pipeline stopped. Fix the error and re-run.[/yellow]")
            return

        console.print(Panel(
            "[green]OK Full pipeline complete![/green]\n\n"
            "Check your results:\n"
            "  PD Results     : results/pd_results_[TIMESTAMP].csv\n"
            "  Exp Results    : results/experiment_results_[TIMESTAMP].csv\n"
            "  Research Output: results/research_output_[TIMESTAMP].csv\n"
            "  Summary JSON   : results/research_summary_[TIMESTAMP].json\n",
            title="[bold green]Pipeline Complete[/bold green]",
            border_style="green"
        ))


def _step_setup():
    from app.services.seed_service import setup_db
    asyncio.run(setup_db())
    console.print("[green]  OK Database initialized and seeded.[/green]")


def _step_train_pd():
    from app.services.data_pipeline import DataPipelineService
    pipeline = DataPipelineService(raw_data_path="data/raw/application_train.csv")
    asyncio.run(pipeline.run_pipeline())
    console.print("[green]  OK PD model trained. Results in data/processed/pd_results_latest.csv[/green]")


def _step_build_rag():
    from sqlalchemy import select
    from app.database import AsyncSessionLocal
    from app.models.orm_models import PolicyDocument
    from app.config import get_settings
    settings = get_settings()

    async def _b():
        async with AsyncSessionLocal() as db:
            docs = (await db.execute(select(PolicyDocument))).scalars().all()
        if not docs:
            return
        try:
            from app.rag.rag_service import RAGService
            rag = RAGService(settings.chroma_persist_directory, settings.embedding_model, settings.rag_top_k)
            rag.index_documents([{
                "document_code": d.document_code, "title": d.title,
                "version": d.version, "status": d.status,
                "is_authoritative": d.is_authoritative,
                "effective_date": str(d.effective_date) if d.effective_date else "",
                "content": d.full_text or d.summary or "",
            } for d in docs])
        except Exception:
            pass  # Non-critical

    asyncio.run(_b())
    console.print("[green]  OK RAG index built.[/green]")


def _step_run_experiments(use_mock: bool):
    from app.services.cli_experiment_runner import run_all_experiments
    success = asyncio.run(run_all_experiments(use_mock=use_mock))
    if not success:
        raise RuntimeError("Experiments failed. Check LLM configuration.")
    console.print("[green]  OK Experiments complete. Results in data/processed/experiment_results_latest.csv[/green]")


def _step_export():
    from app.services.export_service import export_research_outputs
    export_research_outputs()
    console.print("[green]  OK Research outputs exported to results/[/green]")


# ============================================================
# STATUS COMMAND
# ============================================================
@cli.command()
def status():
    """Show the current pipeline status (what has been completed)."""
    from app.database import AsyncSessionLocal
    from sqlalchemy import select, func
    from app.models.orm_models import Applicant, ApplicantPDScore, PDModel, ExperimentRun

    async def _status():
        async with AsyncSessionLocal() as db:
            n_applicants = (await db.execute(select(func.count(Applicant.applicant_id)))).scalar()
            n_pd_scores = (await db.execute(select(func.count(ApplicantPDScore.score_id)))).scalar()
            n_exp_runs = (await db.execute(select(func.count(ExperimentRun.run_id)))).scalar()
            pd_model = (await db.execute(
                select(PDModel).order_by(PDModel.created_at.desc()).limit(1)
            )).scalars().first()

        pd_results = list(Path("results").glob("pd_results_*.csv"))
        exp_results = list(Path("results").glob("experiment_results_*.csv"))
        research_out = list(Path("results").glob("research_output_*.csv"))

        pd_csv_path = pd_results[0] if pd_results else None
        exp_csv_path = exp_results[0] if exp_results else None
        res_csv_path = research_out[0] if research_out else None

        table = Table(title="Pipeline Status", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Step", width=30)
        table.add_column("Status", width=20)
        table.add_column("Details", width=40)

        def row(step, done, detail=""):
            status_str = "[green]DONE[/green]" if done else "[yellow]PENDING[/yellow]"
            table.add_row(step, status_str, detail)

        row("Step 1: Setup (DB + Seed)", n_applicants > 0, f"{n_applicants} applicants seeded")
        row("Step 2: Train PD Model", pd_model is not None,
            f"v{pd_model.model_version} ROC-AUC={pd_model.roc_auc:.4f}" if pd_model else "")
        row("Step 3: PD Scores Generated", n_pd_scores > 0, f"{n_pd_scores} applicants scored")
        row("Step 3b: PD Results CSV", pd_csv_path is not None,
            str(pd_csv_path) if pd_csv_path else "Run 'train-pd'")
        row("Step 4: RAG Index", Path("data/chroma_db").exists(), "ChromaDB index present")
        row("Step 5: Experiments Run", n_exp_runs > 0 or exp_csv_path is not None,
            f"{n_exp_runs} DB runs" + (" + CSV" if exp_csv_path else ""))
        row("Step 6: Research Output", res_csv_path is not None,
            str(res_csv_path) if res_csv_path else "Run 'export'")

        console.print(table)
        console.print("\n[bold]Quick run options:[/bold]")
        console.print("  Individual steps : [bold]python cli.py <command>[/bold]")
        console.print("  Full auto (real) : [bold]python cli.py run-all[/bold]")
        console.print("  Full auto (mock) : [bold]python cli.py run-all --mock[/bold]")

    asyncio.run(_status())


# ============================================================
# VALIDATE DATA (kept for compatibility)
# ============================================================
@cli.command("validate-data")
def validate_data():
    """Validate the Home Credit dataset file exists and is readable."""
    raw_path = Path("data/raw/application_train.csv")
    if not raw_path.exists():
        console.print(f"[red]FAIL Dataset not found: {raw_path}[/red]")
        console.print("  Download from: https://www.kaggle.com/competitions/home-credit-default-risk/data")
        return
    import pandas as pd
    df = pd.read_csv(raw_path, nrows=5)
    console.print(f"[green]OK Dataset found:[/green] {raw_path}")
    console.print(f"  Columns: {list(df.columns[:10])} ...")
    console.print(f"  Run 'train-pd' to begin.")
