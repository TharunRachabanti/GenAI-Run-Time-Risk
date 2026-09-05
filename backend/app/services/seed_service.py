"""
Database Seeder for CLI
Seeds all required initial data: applicants, policy docs, prompts, models, experiment definitions.
"""
import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import AsyncSessionLocal, engine, Base
from app.models.orm_models import (
    Applicant, PolicyDocument, Prompt, GenAIModel, ExperimentDefinition
)
from app.config import get_settings
from data.applicants.applicant_data import PILOT_APPLICANTS, MAIN_STUDY_APPLICANTS
from policies.policy_knowledge_base import POLICY_DOCUMENTS

logger = logging.getLogger(__name__)
settings = get_settings()


async def seed_database():
    """Seed the database with all required initial data."""
    async with AsyncSessionLocal() as db:

        # ---- Applicants ----
        for app_data in PILOT_APPLICANTS + MAIN_STUDY_APPLICANTS:
            existing = await db.execute(
                select(Applicant).where(Applicant.applicant_code == app_data["applicant_code"])
            )
            if not existing.scalars().first():
                db.add(Applicant(**app_data))

        # ---- Policy Documents ----
        for doc_data in POLICY_DOCUMENTS:
            existing = await db.execute(
                select(PolicyDocument).where(PolicyDocument.document_code == doc_data["document_code"])
            )
            if not existing.scalars().first():
                # Remove string IDs that should be auto-generated UUIDs by the ORM
                # Also convert effective_date string to datetime if needed
                from datetime import datetime, timezone as tz
                doc_copy = {
                    k: v for k, v in doc_data.items()
                    if k not in ("document_id", "policy_id")  # policy_id is FK — leave NULL
                }
                # Parse date strings to datetime objects for SQLite compatibility
                for date_field in ("effective_date", "expiry_date"):
                    val = doc_copy.get(date_field)
                    if isinstance(val, str):
                        try:
                            doc_copy[date_field] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                        except ValueError:
                            doc_copy[date_field] = None

                # Ensure full_text exists if the model expects it
                if "full_text" in doc_copy and doc_copy.get("full_text") is None:
                    doc_copy.pop("full_text", None)
                # Map 'content' to what the model uses
                db.add(PolicyDocument(**doc_copy))

        # ---- Master Policy (needed by Data Pipeline) ----
        from app.models.orm_models import Policy
        from datetime import datetime, timezone as tz
        policy_code = "POLICY-CL-001-v3"
        existing_policy = await db.execute(
            select(Policy).where(Policy.policy_code == policy_code)
        )
        if not existing_policy.scalars().first():
            policy = Policy(
                policy_code=policy_code,
                policy_name="Consumer Lending Policy",
                policy_version="3.0",
                effective_date=datetime(2024, 1, 1, tzinfo=tz.utc),
                status="current",
                is_authoritative=True,
                pd_approve_threshold=0.05,
                pd_decline_threshold=0.10,
                dti_review_threshold=40.0
            )
            db.add(policy)

        # ---- Prompts (9 templates matching prompt_templates.py) ----
        prompts = [
            # EXP-001 Baseline (also used as neutral in EXP-002)
            {"prompt_code": "PRM-v1", "prompt_name": "Neutral Baseline",
             "prompt_version": "1.0", "prompt_type": "v1_neutral",
             "prompt_hash": "v1_neutral", "system_prompt": "standard",
             "user_prompt_template": "v1_neutral", "is_baseline": True, "is_active": True},

            # EXP-002 Tone Variation
            {"prompt_code": "PRM-v2", "prompt_name": "Risk Focused",
             "prompt_version": "2.0", "prompt_type": "v2_risk_focused",
             "prompt_hash": "v2_risk_focused", "system_prompt": "standard",
             "user_prompt_template": "v2_risk_focused", "is_baseline": False, "is_active": True},
            {"prompt_code": "PRM-v3", "prompt_name": "Business Oriented",
             "prompt_version": "3.0", "prompt_type": "v3_business",
             "prompt_hash": "v3_business", "system_prompt": "standard",
             "user_prompt_template": "v3_business", "is_baseline": False, "is_active": True},

            # EXP-004 Numeric PD Representation
            {"prompt_code": "PRM-v4", "prompt_name": "PD as Decimal",
             "prompt_version": "4.0", "prompt_type": "v4_pd_decimal",
             "prompt_hash": "v4_pd_decimal", "system_prompt": "standard",
             "user_prompt_template": "v4_pd_decimal", "is_baseline": False, "is_active": True},
            {"prompt_code": "PRM-v5", "prompt_name": "PD Approximate",
             "prompt_version": "5.0", "prompt_type": "v5_pd_approximate",
             "prompt_hash": "v5_pd_approximate", "system_prompt": "standard",
             "user_prompt_template": "v5_pd_approximate", "is_baseline": False, "is_active": True},

            # EXP-005 Contextual Completeness
            {"prompt_code": "PRM-v6", "prompt_name": "Partial Context",
             "prompt_version": "6.0", "prompt_type": "v6_partial_context",
             "prompt_hash": "v6_partial_context", "system_prompt": "standard",
             "user_prompt_template": "v6_partial_context", "is_baseline": False, "is_active": True},

            # EXP-006 Explicit Threshold Reminder
            {"prompt_code": "PRM-v7", "prompt_name": "Explicit Thresholds",
             "prompt_version": "7.0", "prompt_type": "v7_explicit_thresh",
             "prompt_hash": "v7_explicit_thresh", "system_prompt": "standard",
             "user_prompt_template": "v7_explicit_thresh", "is_baseline": False, "is_active": True},

            # EXP-007 Instruction Conflict
            {"prompt_code": "PRM-v8", "prompt_name": "Conservative Persona",
             "prompt_version": "8.0", "prompt_type": "v8_conservative",
             "prompt_hash": "v8_conservative", "system_prompt": "conservative",
             "user_prompt_template": "v8_conservative", "is_baseline": False, "is_active": True},
            {"prompt_code": "PRM-v9", "prompt_name": "Lenient Persona",
             "prompt_version": "9.0", "prompt_type": "v9_lenient",
             "prompt_hash": "v9_lenient", "system_prompt": "lenient",
             "user_prompt_template": "v9_lenient", "is_baseline": False, "is_active": True},
        ]
        for p in prompts:
            existing = await db.execute(
                select(Prompt).where(Prompt.prompt_code == p["prompt_code"])
            )
            if not existing.scalars().first():
                db.add(Prompt(**p))

        # ---- GenAI Model (single provider from config) ----
        provider = settings.genai_provider
        model_name = settings.genai_model_name
        model_version = settings.genai_model_version

        # Always seed a mock model for testing
        for m in [
            {"provider": "mock", "model_name": "mock-model", "model_version": "1.0",
             "display_name": "Mock LLM (Testing)", "is_active": True, "model_metadata": {}},
            {"provider": provider, "model_name": model_name, "model_version": model_version,
             "display_name": f"{provider.title()} / {model_name}", "is_active": True,
             "model_metadata": {"configured_via": ".env"}},
        ]:
            existing = await db.execute(
                select(GenAIModel).where(
                    GenAIModel.provider == m["provider"],
                    GenAIModel.model_name == m["model_name"],
                )
            )
            if not existing.scalars().first():
                db.add(GenAIModel(**m))

        # ---- Experiment Definitions (EXP-001 to EXP-007) ----
        experiment_defs = [
            {
                "experiment_code": "EXP-001",
                "experiment_name": "Baseline Repeatability",
                "experiment_type": "baseline",
                "description": "Run identical inputs 3 times to measure intra-run LLM consistency.",
                "hypothesis": "An LLM should produce identical decisions for identical inputs at low temperature.",
                "fixed_variables": ["applicant", "prompt", "model", "policy"],
                "varied_variable": "none",
                "expected_controlled_vars": [],
                "status": "active",
            },
            {
                "experiment_code": "EXP-002",
                "experiment_name": "Prompt Tone Variation",
                "experiment_type": "prompt_variation",
                "description": "Test 3 prompt framings: neutral, risk-focused, business-oriented.",
                "hypothesis": "Prompt tone alone should not change the credit decision if PD is fixed.",
                "fixed_variables": ["applicant", "model", "policy", "pd_score"],
                "varied_variable": "prompt_template",
                "expected_controlled_vars": [],
                "status": "active",
            },
            {
                "experiment_code": "EXP-003",
                "experiment_name": "Decision Boundary Sensitivity",
                "experiment_type": "boundary",
                "description": "Run boundary-case applicants (near 5% and 10% PD) 3 times each.",
                "hypothesis": "LLM decisions should be stable even for near-threshold applicants.",
                "fixed_variables": ["prompt", "model", "policy"],
                "varied_variable": "applicant_pd_proximity",
                "expected_controlled_vars": [],
                "status": "active",
            },
            {
                "experiment_code": "EXP-004",
                "experiment_name": "Numeric PD Representation",
                "experiment_type": "numeric_format",
                "description": "Present PD as percentage (9.17%), decimal (0.0917), and approximate (~9%).",
                "hypothesis": "LLMs may be sensitive to numeric formatting, potentially misinterpreting decimal vs. percentage.",
                "fixed_variables": ["applicant", "model", "policy", "pd_score_value"],
                "varied_variable": "pd_numeric_format",
                "expected_controlled_vars": [],
                "status": "active",
            },
            {
                "experiment_code": "EXP-005",
                "experiment_name": "Contextual Completeness",
                "experiment_type": "context_variation",
                "description": "Full applicant info vs. primary indicators only (PD score kept fixed).",
                "hypothesis": "If PD is fixed, removing secondary context should not change the final decision.",
                "fixed_variables": ["model", "policy", "pd_score"],
                "varied_variable": "applicant_info_completeness",
                "expected_controlled_vars": [],
                "status": "active",
            },
            {
                "experiment_code": "EXP-006",
                "experiment_name": "Explicit Threshold Reminder",
                "experiment_type": "instruction_variation",
                "description": "Compare prompts with and without explicit numeric policy thresholds stated.",
                "hypothesis": "Explicitly stating 'PD < 5% = APPROVE' improves policy compliance vs. relying on RAG context alone.",
                "fixed_variables": ["applicant", "model", "pd_score"],
                "varied_variable": "threshold_explicitness",
                "expected_controlled_vars": [],
                "status": "active",
            },
            {
                "experiment_code": "EXP-007",
                "experiment_name": "Instruction Conflict (Persona)",
                "experiment_type": "persona_conflict",
                "description": "Conservative vs. lenient system-level persona instructions.",
                "hypothesis": "A persona shift in the system prompt should not override the policy-mandated decision.",
                "fixed_variables": ["applicant", "model", "policy", "pd_score"],
                "varied_variable": "system_persona",
                "expected_controlled_vars": [],
                "status": "active",
            },
        ]
        for d in experiment_defs:
            existing = await db.execute(
                select(ExperimentDefinition).where(
                    ExperimentDefinition.experiment_code == d["experiment_code"]
                )
            )
            if not existing.scalars().first():
                db.add(ExperimentDefinition(**d))

        await db.commit()
        logger.info("Database successfully seeded.")


async def setup_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    await seed_database()
