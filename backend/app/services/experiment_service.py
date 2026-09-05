"""
API Router — Experiments
Handles experiment creation and execution including:
- Baseline / Repeatability
- Prompt Variation
- Policy/RAG Variation
- Model Change
- Decision Boundary
"""
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.orm_models import (
    Applicant, ApplicantPDScore, ExperimentDefinition, ExperimentRun,
    GenAIModel, GenAIOutput, Prompt, ReferenceDecision, RetrievedDocument, PolicyDocument,
)
from app.schemas.schemas import (
    ExperimentDefinitionCreate, ExperimentRunRequest, ExperimentRunResponse, APIResponse,
)
from app.genai.genai_assistant import GenAIAssistant
from app.rag.rag_service import RAGService
from app.evaluation.evaluation_engine import EvaluationEngine
from app.config import get_settings
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)
settings = get_settings()


async def create_experiment_definition(
    definition: ExperimentDefinitionCreate,
    db: AsyncSession,
):
    """Create a new experiment definition."""
    db_def = ExperimentDefinition(**definition.model_dump())
    db.add(db_def)
    await db.flush()
    return {"experiment_def_id": str(db_def.experiment_def_id)}


async def list_experiment_definitions(db: AsyncSession):
    """List all experiment definitions."""
    result = await db.execute(select(ExperimentDefinition))
    defs = result.scalars().all()
    return [
        {
            "experiment_def_id": str(d.experiment_def_id),
            "experiment_code": d.experiment_code,
            "experiment_name": d.experiment_name,
            "experiment_type": d.experiment_type,
            "status": d.status,
            "varied_variable": d.varied_variable,
        }
        for d in defs
    ]


async def create_and_run_experiment(
    request: ExperimentRunRequest,
    
    db: AsyncSession,
):
    """
    Create and execute an experiment run.
    Validates that the correct variables are held constant.
    """
    # Fetch required entities
    applicant = await db.get(Applicant, request.applicant_id)
    if not applicant:
        raise ValueError("Applicant not found")

    prompt = await db.get(Prompt, request.prompt_id)
    if not prompt:
        raise ValueError("Prompt not found")

    genai_model = await db.get(GenAIModel, request.genai_model_id)
    if not genai_model:
        raise ValueError("GenAI model not found")

    experiment_def = await db.get(ExperimentDefinition, request.experiment_def_id)
    if not experiment_def:
        raise ValueError("Experiment definition not found")

    # Fetch frozen PD score
    pd_result = await db.execute(
        select(ApplicantPDScore).where(ApplicantPDScore.applicant_id == request.applicant_id)
    )
    pd_score_record = pd_result.scalar_one_or_none()
    if not pd_score_record:
        raise ValueError("No frozen PD score found for applicant")

    fixed_pd_score = pd_score_record.pd_score_frozen

    # Fetch reference decision
    ref_result = await db.execute(
        select(ReferenceDecision).where(ReferenceDecision.applicant_id == request.applicant_id)
    )
    ref_decision = ref_result.scalar_one_or_none()

    # Validate paired experiment integrity
    if request.baseline_run_id:
        baseline_run = await db.get(ExperimentRun, request.baseline_run_id)
        if not baseline_run:
            raise ValueError("Baseline run not found")
        if baseline_run.applicant_id != request.applicant_id:
            raise HTTPException(
                status_code=400,
                detail="Paired experiment integrity violation: applicant_id must match baseline run"
            )
        if abs(baseline_run.fixed_pd_score - fixed_pd_score) > 0.0001:
            raise HTTPException(
                status_code=400,
                detail="Paired experiment integrity violation: PD score mismatch with baseline"
            )

    # Create run record
    run = ExperimentRun(
        experiment_def_id=request.experiment_def_id,
        applicant_id=request.applicant_id,
        prompt_id=request.prompt_id,
        genai_model_id=request.genai_model_id,
        reference_decision_id=ref_decision.decision_id if ref_decision else None,
        dataset_version=settings.dataset_version,
        applicant_data_version=applicant.data_version,
        fixed_pd_score=fixed_pd_score,
        pd_model_version=settings.pd_model_version,
        prompt_version=prompt.prompt_version,
        prompt_hash=prompt.prompt_hash,
        genai_provider=genai_model.provider,
        genai_model_name=genai_model.model_name,
        genai_model_version=genai_model.model_version,
        model_parameters=request.model_parameters or {
            "temperature": settings.genai_temperature,
            "max_tokens": settings.genai_max_tokens,
        },
        retrieval_config=request.retrieval_config,
        policy_scenario=request.policy_scenario,
        baseline_run_id=request.baseline_run_id,
        is_baseline_run=request.is_baseline_run,
        status="pending",
    )
    db.add(run)
    await db.flush()
    run_id = run.run_id
    await db.commit()

    # Execute in background
    await _execute_run(
        run_id=run_id,
        applicant_data=_applicant_to_dict(applicant),
        fixed_pd_score=fixed_pd_score,
        prompt=prompt,
        genai_model=genai_model,
        policy_scenario=request.policy_scenario or "current_only",
        reference_decision=ref_decision.reference_decision if ref_decision else None,
        baseline_run_id=request.baseline_run_id,
        experiment_type=experiment_def.experiment_type,
        varied_variable=experiment_def.varied_variable,
    )

    return APIResponse(
        message="Experiment run created and executing in background",
        data={"run_id": str(run_id), "status": "pending"}
    )


async def get_run(run_id: uuid.UUID, db: AsyncSession):
    """Get experiment run details and results."""
    run = await db.get(ExperimentRun, run_id)
    if not run:
        raise ValueError("Run not found")

    output = None
    if run.genai_output:
        output = {
            "recommendation": run.genai_output.recommendation,
            "risk_classification": run.genai_output.risk_classification,
            "policy_conflict_flag": run.genai_output.policy_conflict_flag,
            "unsupported_conclusion_flag": run.genai_output.unsupported_conclusion_flag,
            "human_review_required": run.genai_output.human_review_required,
            "parse_success": run.genai_output.parse_success,
        }

    eval_result = None
    if run.evaluation_result:
        eval_result = {
            "deviation_classification": run.evaluation_result.deviation_classification,
            "materiality": run.evaluation_result.materiality,
            "severity": run.evaluation_result.severity,
            "transition": run.evaluation_result.transition,
            "agrees_with_reference": run.evaluation_result.agrees_with_reference,
            "evaluation_notes": run.evaluation_result.evaluation_notes,
        }

    return APIResponse(data={
        "run_id": str(run.run_id),
        "experiment_def_id": str(run.experiment_def_id),
        "applicant_id": str(run.applicant_id),
        "status": run.status,
        "fixed_pd_score": run.fixed_pd_score,
        "prompt_version": run.prompt_version,
        "genai_provider": run.genai_provider,
        "genai_model_name": run.genai_model_name,
        "policy_scenario": run.policy_scenario,
        "genai_output": output,
        "evaluation": eval_result,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    })





def _applicant_to_dict(applicant: Applicant) -> Dict[str, Any]:
    """Convert ORM Applicant to dict for GenAI assistant."""
    return {
        "applicant_name": applicant.applicant_name,
        "annual_income": applicant.annual_income,
        "loan_amount_requested": applicant.loan_amount_requested,
        "monthly_annuity": applicant.monthly_annuity,
        "property_value": applicant.property_value,
        "loan_term_months": applicant.loan_term_months,
        "interest_rate": applicant.interest_rate,
        "debt_to_income_ratio": applicant.debt_to_income_ratio,
        "combined_loan_to_value_ratio": applicant.combined_loan_to_value_ratio,
        "credit_to_income_ratio": applicant.credit_to_income_ratio,
        "loan_purpose": applicant.loan_purpose,
        "occupancy_type": applicant.occupancy_type,
        "loan_type": applicant.loan_type,
        "employment_status": applicant.employment_status,
        "employment_years": applicant.employment_years,
        "credit_history_years": applicant.credit_history_years,
        "delinquency_90day_count": applicant.delinquency_90day_count,
        "delinquency_30day_count": applicant.delinquency_30day_count,
        "external_credit_score_proxy": applicant.external_credit_score_proxy,
    }


async def _execute_run(
    run_id: uuid.UUID,
    applicant_data: Dict[str, Any],
    fixed_pd_score: float,
    prompt,
    genai_model,
    policy_scenario: str,
    reference_decision: Optional[str],
    baseline_run_id: Optional[uuid.UUID],
    experiment_type: str,
    varied_variable: str,
):
    """Background task: execute GenAI run, store output, evaluate."""
    from app.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        try:
            # Update run status
            run = await db.get(ExperimentRun, run_id)
            run.status = "running"
            run.started_at = datetime.now(timezone.utc)
            await db.commit()

            # Initialize services
            rag_service = RAGService(
                persist_directory=settings.chroma_persist_directory,
                embedding_model_name=settings.embedding_model,
                top_k=settings.rag_top_k,
            )

            assistant = GenAIAssistant(
                provider=genai_model.provider,
                model_name=genai_model.model_name,
                model_version=genai_model.model_version,
                temperature=settings.genai_temperature,
                max_tokens=settings.genai_max_tokens,
            )

            # Build RAG query
            query = (
                f"Consumer loan application review. "
                f"PD score: {fixed_pd_score:.4f}. "
                f"DTI: {applicant_data.get('debt_to_income_ratio', 0):.1f}%. "
                f"Loan purpose: {applicant_data.get('loan_purpose', '')}. "
                f"Delinquencies 90-day: {applicant_data.get('delinquency_90day_count', 0)}, "
                f"30-day: {applicant_data.get('delinquency_30day_count', 0)}."
            )

            retrieved_docs = rag_service.retrieve(
                query=query,
                policy_scenario=policy_scenario,
                top_k=settings.rag_top_k,
            )

            # Store retrieved documents
            for doc in retrieved_docs:
                # Find policy document in DB
                doc_result = await db.execute(
                    select(PolicyDocument).where(PolicyDocument.document_code == doc["document_code"])
                )
                policy_doc = doc_result.scalar_one_or_none()
                if policy_doc:
                    retrieved = RetrievedDocument(
                        experiment_run_id=run_id,
                        policy_document_id=policy_doc.document_id,
                        retrieval_rank=doc["retrieval_rank"],
                        retrieval_score=doc.get("retrieval_score"),
                        document_version_at_retrieval=doc["version"],
                        document_status_at_retrieval=doc["status"],
                        is_authoritative_at_retrieval=doc["is_authoritative"],
                        retrieved_content_snippet=doc["content"][:500] if doc.get("content") else None,
                    )
                    db.add(retrieved)

            # Run GenAI assessment
            applicant_id_str = str(run_id)[:8]  # Use run_id prefix as applicant identifier for the model
            result = await assistant.assess(
                applicant_id=applicant_id_str,
                applicant_data=applicant_data,
                fixed_pd_score=fixed_pd_score,
                retrieved_docs=retrieved_docs,
                prompt_version=prompt.prompt_type,
                rag_service=rag_service,
            )

            # Store GenAI output
            assessment = result.get("structured_assessment")
            genai_out = GenAIOutput(
                experiment_run_id=run_id,
                raw_model_response=result.get("raw_response", ""),
                raw_response_metadata=result.get("response_metadata"),
                parse_success=result.get("parse_success", False),
                parse_error=result.get("parse_error"),
            )

            if assessment:
                genai_out.applicant_id_returned = assessment.applicant_id
                genai_out.pd_score_returned = assessment.pd_score
                genai_out.risk_classification = assessment.risk_classification
                genai_out.recommendation = assessment.recommendation
                genai_out.pd_interpretation = assessment.pd_interpretation
                genai_out.primary_risk_factors = assessment.primary_risk_factors
                genai_out.supporting_applicant_evidence = assessment.supporting_applicant_evidence
                genai_out.policy_references = assessment.policy_references
                genai_out.policy_conflict_flag = assessment.policy_conflict_flag
                genai_out.unsupported_conclusion_flag = assessment.unsupported_conclusion_flag
                genai_out.human_review_required = assessment.human_review_required
                genai_out.reasoning_summary = assessment.reasoning_summary
                genai_out.runtime_configuration_returned = assessment.runtime_configuration.model_dump()

            db.add(genai_out)
            await db.flush()

            # Evaluate
            from app.evaluation.evaluation_engine import EvaluationEngine, EvaluationOutput
            from app.models.orm_models import EvaluationResult

            engine = EvaluationEngine()
            genai_output_dict = {
                "recommendation": genai_out.recommendation,
                "risk_classification": genai_out.risk_classification,
                "primary_risk_factors": genai_out.primary_risk_factors or [],
                "policy_references": genai_out.policy_references or [],
                "policy_conflict_flag": genai_out.policy_conflict_flag,
                "unsupported_conclusion_flag": genai_out.unsupported_conclusion_flag,
                "human_review_required": genai_out.human_review_required,
                "reasoning_summary": genai_out.reasoning_summary or "",
            }

            # Fetch baseline output if paired
            baseline_output_dict = None
            if baseline_run_id:
                baseline_run = await db.get(ExperimentRun, baseline_run_id)
                if baseline_run and baseline_run.genai_output:
                    bo = baseline_run.genai_output
                    baseline_output_dict = {
                        "run_id": str(baseline_run_id),
                        "recommendation": bo.recommendation,
                        "risk_classification": bo.risk_classification,
                        "primary_risk_factors": bo.primary_risk_factors or [],
                        "policy_references": bo.policy_references or [],
                    }

            eval_output = engine.evaluate(
                run_id=str(run_id),
                experiment_type=experiment_type,
                genai_output=genai_output_dict,
                reference_decision=reference_decision,
                baseline_output=baseline_output_dict,
                policy_scenario=policy_scenario,
                varied_variable=varied_variable,
            )

            eval_record = EvaluationResult(
                experiment_run_id=run_id,
                genai_output_id=genai_out.output_id,
                baseline_run_id=baseline_run_id,
                reference_decision=eval_output.reference_decision,
                genai_decision=eval_output.genai_decision,
                agrees_with_reference=eval_output.agrees_with_reference,
                decision_changed=eval_output.decision_changed,
                risk_classification_changed=eval_output.risk_classification_changed,
                policy_interpretation_changed=eval_output.policy_interpretation_changed,
                evidence_changed=eval_output.evidence_changed,
                human_review_changed=eval_output.human_review_changed,
                policy_conflict_detected=eval_output.policy_conflict_detected,
                unsupported_conclusion=eval_output.unsupported_conclusion,
                pd_correctly_interpreted=eval_output.pd_correctly_interpreted,
                correct_policy_cited=eval_output.correct_policy_cited,
                obsolete_policy_used=eval_output.obsolete_policy_used,
                failed_to_escalate=eval_output.failed_to_escalate,
                deviation_classification=eval_output.deviation_classification.value,
                materiality=eval_output.materiality.value,
                severity=eval_output.severity.value,
                transition=eval_output.transition,
                evaluation_notes=eval_output.evaluation_notes,
            )
            db.add(eval_record)

            # Mark run complete
            run.status = "completed"
            run.completed_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info(f"Run {run_id} completed. Decision: {genai_out.recommendation} | Severity: {eval_output.severity.value}")

        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            async with AsyncSessionLocal() as err_db:
                err_run = await err_db.get(ExperimentRun, run_id)
                if err_run:
                    err_run.status = "failed"
                    err_run.error_message = str(e)
                    await err_db.commit()
