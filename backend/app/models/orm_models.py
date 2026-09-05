"""
SQLAlchemy ORM Models — Complete Database Schema
GenAI Runtime Risk Research Platform
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def uuid_pk():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def now_utc():
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# ============================================================
# APPLICANTS
# ============================================================
class Applicant(Base):
    __tablename__ = "applicants"

    applicant_id: Mapped[uuid.UUID] = uuid_pk()
    applicant_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    applicant_name: Mapped[str] = mapped_column(String(200), nullable=False)
    data_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1.0.0")

    # Financial characteristics
    annual_income: Mapped[float] = mapped_column(Float, nullable=False)
    loan_amount_requested: Mapped[float] = mapped_column(Float, nullable=False)
    monthly_annuity: Mapped[float] = mapped_column(Float, nullable=True)
    property_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    loan_term_months: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    interest_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Risk indicators
    debt_to_income_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    combined_loan_to_value_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    credit_to_income_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Loan characteristics
    loan_purpose: Mapped[str] = mapped_column(String(50), nullable=False)
    occupancy_type: Mapped[str] = mapped_column(String(50), nullable=False)
    loan_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Employment
    employment_status: Mapped[str] = mapped_column(String(50), nullable=False)
    employment_years: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Credit history
    credit_history_years: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    delinquency_90day_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    delinquency_30day_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    external_credit_score_proxy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Sampling metadata
    risk_tier: Mapped[str] = mapped_column(String(50), nullable=False)
    is_pilot: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_boundary_case: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    boundary_threshold: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    sampling_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = now_utc()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    pd_scores: Mapped[list["ApplicantPDScore"]] = relationship(back_populates="applicant")
    reference_decisions: Mapped[list["ReferenceDecision"]] = relationship(back_populates="applicant")
    experiment_runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="applicant")

    __table_args__ = (
        CheckConstraint("debt_to_income_ratio >= 0 AND debt_to_income_ratio <= 100", name="chk_dti_range"),
        CheckConstraint("annual_income > 0", name="chk_income_positive"),
        Index("ix_applicants_risk_tier", "risk_tier"),
        Index("ix_applicants_is_pilot", "is_pilot"),
    )


# ============================================================
# PD MODELS
# ============================================================
class PDModel(Base):
    __tablename__ = "pd_models"

    pd_model_id: Mapped[uuid.UUID] = uuid_pk()
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False, default="LogisticRegression")
    training_dataset: Mapped[str] = mapped_column(String(200), nullable=False)
    training_dataset_version: Mapped[str] = mapped_column(String(50), nullable=False)

    # Validation metrics
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brier_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ks_statistic: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Artifacts
    model_artifact_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    preprocessor_artifact_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    feature_list: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    training_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    validation_results: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Status
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    frozen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = now_utc()
    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    pd_scores: Mapped[list["ApplicantPDScore"]] = relationship(back_populates="pd_model")

    __table_args__ = (
        UniqueConstraint("model_name", "model_version", name="uq_pd_model_version"),
    )


class ApplicantPDScore(Base):
    __tablename__ = "applicant_pd_scores"

    score_id: Mapped[uuid.UUID] = uuid_pk()
    applicant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applicants.applicant_id"), nullable=False)
    pd_model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pd_models.pd_model_id"), nullable=False)

    pd_score: Mapped[float] = mapped_column(Float, nullable=False)
    pd_score_frozen: Mapped[float] = mapped_column(Float, nullable=False)  # Immutable after freezing
    is_frozen: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    frozen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    feature_values: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    score_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = now_utc()

    # Relationships
    applicant: Mapped["Applicant"] = relationship(back_populates="pd_scores")
    pd_model: Mapped["PDModel"] = relationship(back_populates="pd_scores")
    reference_decisions: Mapped[list["ReferenceDecision"]] = relationship(back_populates="pd_score_record")

    __table_args__ = (
        UniqueConstraint("applicant_id", "pd_model_id", name="uq_applicant_pd_model"),
        CheckConstraint("pd_score >= 0 AND pd_score <= 1", name="chk_pd_score_range"),
        Index("ix_pd_scores_applicant", "applicant_id"),
    )


# ============================================================
# POLICIES
# ============================================================
class Policy(Base):
    __tablename__ = "policies"

    policy_id: Mapped[uuid.UUID] = uuid_pk()
    policy_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    policy_name: Mapped[str] = mapped_column(String(200), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(50), nullable=False)
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="current")  # current / superseded
    is_authoritative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Policy rules (JSON — structured rules engine)
    pd_approve_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    pd_decline_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.10)
    dti_review_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=40.0)
    dti_decline_threshold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    delinquency_decline_rules: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    additional_rules: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = now_utc()

    # Relationships
    documents: Mapped[list["PolicyDocument"]] = relationship(back_populates="policy")
    reference_decisions: Mapped[list["ReferenceDecision"]] = relationship(back_populates="policy")

    __table_args__ = (
        CheckConstraint("pd_approve_threshold < pd_decline_threshold", name="chk_pd_thresholds"),
        CheckConstraint("status IN ('current', 'superseded', 'draft', 'archived')", name="chk_policy_status"),
    )


class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    document_id: Mapped[uuid.UUID] = uuid_pk()
    document_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    policy_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("policies.policy_id"), nullable=True)

    title: Mapped[str] = mapped_column(String(500), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expiry_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="current")
    is_authoritative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    policy_category: Mapped[str] = mapped_column(String(100), nullable=False)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    source_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Embedding stored externally in Chroma but referenced here
    embedding_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    created_at: Mapped[datetime] = now_utc()

    # Relationships
    policy: Mapped[Optional["Policy"]] = relationship(back_populates="documents")
    retrieved_in: Mapped[list["RetrievedDocument"]] = relationship(back_populates="policy_document")

    __table_args__ = (
        CheckConstraint("status IN ('current', 'superseded', 'draft', 'archived', 'conflict')", name="chk_doc_status"),
        Index("ix_policy_docs_status", "status"),
        Index("ix_policy_docs_category", "policy_category"),
    )


# ============================================================
# REFERENCE DECISIONS (Policy-Based Deterministic Decisions)
# ============================================================
class ReferenceDecision(Base):
    __tablename__ = "reference_decisions"

    decision_id: Mapped[uuid.UUID] = uuid_pk()
    applicant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applicants.applicant_id"), nullable=False)
    pd_score_record_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applicant_pd_scores.score_id"), nullable=False)
    policy_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("policies.policy_id"), nullable=False)

    pd_score: Mapped[float] = mapped_column(Float, nullable=False)
    dti_ratio: Mapped[float] = mapped_column(Float, nullable=False)

    reference_decision: Mapped[str] = mapped_column(String(20), nullable=False)  # APPROVE / MANUAL_REVIEW / DECLINE
    triggered_rule: Mapped[str] = mapped_column(String(200), nullable=False)
    rule_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    decision_explanation: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = now_utc()

    # Relationships
    applicant: Mapped["Applicant"] = relationship(back_populates="reference_decisions")
    pd_score_record: Mapped["ApplicantPDScore"] = relationship(back_populates="reference_decisions")
    policy: Mapped["Policy"] = relationship(back_populates="reference_decisions")
    experiment_runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="reference_decision")

    __table_args__ = (
        CheckConstraint(
            "reference_decision IN ('APPROVE', 'MANUAL_REVIEW', 'DECLINE')",
            name="chk_reference_decision",
        ),
        UniqueConstraint("applicant_id", "policy_id", name="uq_applicant_policy_reference"),
        Index("ix_reference_decisions_applicant", "applicant_id"),
    )


# ============================================================
# PROMPTS
# ============================================================
class Prompt(Base):
    __tablename__ = "prompts"

    prompt_id: Mapped[uuid.UUID] = uuid_pk()
    prompt_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    prompt_name: Mapped[str] = mapped_column(String(200), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_type: Mapped[str] = mapped_column(String(50), nullable=False)  # baseline / variant_risk / variant_business
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256 of content

    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_prompt_template: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    change_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    is_baseline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    created_at: Mapped[datetime] = now_utc()

    # Relationships
    experiment_runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="prompt")

    __table_args__ = (
        UniqueConstraint("prompt_code", "prompt_version", name="uq_prompt_version"),
        Index("ix_prompts_type", "prompt_type"),
    )


# ============================================================
# GENAI MODELS (Provider Registry)
# ============================================================
class GenAIModel(Base):
    __tablename__ = "genai_models"

    model_id: Mapped[uuid.UUID] = uuid_pk()
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # openai / anthropic / google
    model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    model_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = now_utc()

    # Relationships
    experiment_runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="genai_model")

    __table_args__ = (
        UniqueConstraint("provider", "model_name", "model_version", name="uq_genai_model"),
    )


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================
class ExperimentDefinition(Base):
    __tablename__ = "experiment_definitions"

    experiment_def_id: Mapped[uuid.UUID] = uuid_pk()
    experiment_code: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    experiment_name: Mapped[str] = mapped_column(String(200), nullable=False)
    experiment_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # Types: baseline / prompt_variation / policy_rag / model_change / boundary / repeatability

    description: Mapped[str] = mapped_column(Text, nullable=False)
    hypothesis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Which variables are FIXED vs VARIED
    fixed_variables: Mapped[list] = mapped_column(JSON, nullable=False)
    varied_variable: Mapped[str] = mapped_column(String(100), nullable=False)
    expected_controlled_vars: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String(20), nullable=False, default="planned")
    created_at: Mapped[datetime] = now_utc()

    # Relationships
    runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="experiment_definition")


# ============================================================
# EXPERIMENT RUNS
# ============================================================
class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    run_id: Mapped[uuid.UUID] = uuid_pk()
    experiment_def_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment_definitions.experiment_def_id"), nullable=False
    )
    applicant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("applicants.applicant_id"), nullable=False)
    prompt_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("prompts.prompt_id"), nullable=False)
    genai_model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("genai_models.model_id"), nullable=False)
    reference_decision_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("reference_decisions.decision_id"), nullable=True
    )

    # Runtime configuration
    dataset_version: Mapped[str] = mapped_column(String(50), nullable=False)
    applicant_data_version: Mapped[str] = mapped_column(String(50), nullable=False)
    fixed_pd_score: Mapped[float] = mapped_column(Float, nullable=False)
    pd_model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(50), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    genai_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    genai_model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    genai_model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_parameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Retrieval configuration
    retrieval_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    policy_scenario: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    # Scenarios: current_only / current_plus_superseded / superseded_only / current_plus_conflict / authoritative_missing

    # Run status
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Experiment pairing
    baseline_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("experiment_runs.run_id"), nullable=True
    )
    is_baseline_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = now_utc()
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    experiment_definition: Mapped["ExperimentDefinition"] = relationship(back_populates="runs")
    applicant: Mapped["Applicant"] = relationship(back_populates="experiment_runs")
    prompt: Mapped["Prompt"] = relationship(back_populates="experiment_runs")
    genai_model: Mapped["GenAIModel"] = relationship(back_populates="experiment_runs")
    reference_decision: Mapped[Optional["ReferenceDecision"]] = relationship(back_populates="experiment_runs")
    retrieved_documents: Mapped[list["RetrievedDocument"]] = relationship(back_populates="experiment_run")
    genai_output: Mapped[Optional["GenAIOutput"]] = relationship(back_populates="experiment_run", uselist=False)
    evaluation_result: Mapped[Optional["EvaluationResult"]] = relationship(
        back_populates="experiment_run",
        uselist=False,
        foreign_keys="[EvaluationResult.experiment_run_id]",
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="chk_run_status",
        ),
        Index("ix_experiment_runs_experiment", "experiment_def_id"),
        Index("ix_experiment_runs_applicant", "applicant_id"),
        Index("ix_experiment_runs_baseline", "baseline_run_id"),
        Index("ix_experiment_runs_status", "status"),
    )


# ============================================================
# RETRIEVED DOCUMENTS (RAG results per run)
# ============================================================
class RetrievedDocument(Base):
    __tablename__ = "retrieved_documents"

    retrieval_id: Mapped[uuid.UUID] = uuid_pk()
    experiment_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment_runs.run_id"), nullable=False
    )
    policy_document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("policy_documents.document_id"), nullable=False
    )

    retrieval_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    document_version_at_retrieval: Mapped[str] = mapped_column(String(50), nullable=False)
    document_status_at_retrieval: Mapped[str] = mapped_column(String(20), nullable=False)
    is_authoritative_at_retrieval: Mapped[bool] = mapped_column(Boolean, nullable=False)
    retrieved_content_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = now_utc()

    # Relationships
    experiment_run: Mapped["ExperimentRun"] = relationship(back_populates="retrieved_documents")
    policy_document: Mapped["PolicyDocument"] = relationship(back_populates="retrieved_in")

    __table_args__ = (
        Index("ix_retrieved_docs_run", "experiment_run_id"),
    )


# ============================================================
# GENAI OUTPUTS
# ============================================================
class GenAIOutput(Base):
    __tablename__ = "genai_outputs"

    output_id: Mapped[uuid.UUID] = uuid_pk()
    experiment_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment_runs.run_id"), unique=True, nullable=False
    )

    # Raw output
    raw_model_response: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Structured output
    applicant_id_returned: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    pd_score_returned: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    risk_classification: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    recommendation: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    pd_interpretation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    primary_risk_factors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    supporting_applicant_evidence: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    policy_references: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    policy_conflict_flag: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    unsupported_conclusion_flag: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    human_review_required: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    reasoning_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    runtime_configuration_returned: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Parsing
    parse_success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parse_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = now_utc()

    # Relationships
    experiment_run: Mapped["ExperimentRun"] = relationship(back_populates="genai_output")
    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(back_populates="genai_output")

    __table_args__ = (
        CheckConstraint(
            "recommendation IS NULL OR recommendation IN ('APPROVE', 'MANUAL_REVIEW', 'DECLINE')",
            name="chk_genai_recommendation",
        ),
    )


# ============================================================
# EVALUATION RESULTS
# ============================================================
class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    eval_id: Mapped[uuid.UUID] = uuid_pk()
    experiment_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment_runs.run_id"), unique=True, nullable=False
    )
    genai_output_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("genai_outputs.output_id"), nullable=False
    )
    baseline_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("experiment_runs.run_id"), nullable=True
    )

    # Agreement metrics
    reference_decision: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    genai_decision: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    agrees_with_reference: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Change detection (compared to baseline run)
    decision_changed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    risk_classification_changed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    policy_interpretation_changed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    evidence_changed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    human_review_changed: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Quality flags
    policy_conflict_detected: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    unsupported_conclusion: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    pd_correctly_interpreted: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    correct_policy_cited: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    obsolete_policy_used: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    failed_to_escalate: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Classification
    deviation_classification: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Values: EXPECTED_POLICY_CORRECT / EXPECTED_NON_MATERIAL / UNEXPECTED_NON_MATERIAL /
    #         UNEXPECTED_MATERIAL / CRITICAL_POLICY_DEVIATION / STABLE / N_A

    materiality: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # Values: LOW / MODERATE / HIGH / CRITICAL

    severity: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    # Values: LOW / MODERATE / HIGH / CRITICAL

    transition: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # e.g., "APPROVE->DECLINE"

    evaluation_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evaluation_detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    evaluated_at: Mapped[datetime] = now_utc()

    # Relationships
    experiment_run: Mapped["ExperimentRun"] = relationship(
        back_populates="evaluation_result",
        foreign_keys=[experiment_run_id],
    )
    genai_output: Mapped["GenAIOutput"] = relationship(back_populates="evaluation_results")

    __table_args__ = (
        CheckConstraint(
            "materiality IS NULL OR materiality IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')",
            name="chk_materiality",
        ),
        CheckConstraint(
            "severity IS NULL OR severity IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL')",
            name="chk_severity",
        ),
        Index("ix_eval_results_run", "experiment_run_id"),
        Index("ix_eval_results_severity", "severity"),
        Index("ix_eval_results_materiality", "materiality"),
        Index("ix_eval_results_deviation", "deviation_classification"),
    )


# ============================================================
# AUDIT LOGS
# ============================================================
class AuditLog(Base):
    __tablename__ = "audit_logs"

    log_id: Mapped[uuid.UUID] = uuid_pk()
    run_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("experiment_runs.run_id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    entity_id: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = now_utc()

    __table_args__ = (
        Index("ix_audit_logs_run", "run_id"),
        Index("ix_audit_logs_event_type", "event_type"),
        Index("ix_audit_logs_created", "created_at"),
    )
