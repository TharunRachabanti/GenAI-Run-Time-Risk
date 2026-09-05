"""
Pydantic Schemas — Request/Response models for API
GenAI Runtime Risk Research Platform
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================
# SHARED BASE
# ============================================================
class TimestampMixin(BaseModel):
    created_at: Optional[datetime] = None


# ============================================================
# APPLICANT SCHEMAS
# ============================================================
class ApplicantCreate(BaseModel):
    applicant_code: str = Field(..., max_length=50)
    applicant_name: str = Field(..., max_length=200)
    data_version: str = Field(default="v1.0.0")
    annual_income: float = Field(..., gt=0)
    loan_amount_requested: float = Field(..., gt=0)
    monthly_annuity: Optional[float] = None
    property_value: Optional[float] = None
    loan_term_months: Optional[int] = None
    interest_rate: Optional[float] = None
    debt_to_income_ratio: float = Field(..., ge=0, le=100)
    combined_loan_to_value_ratio: Optional[float] = None
    credit_to_income_ratio: Optional[float] = None
    loan_purpose: str
    occupancy_type: str
    loan_type: str
    employment_status: str
    employment_years: Optional[float] = None
    credit_history_years: Optional[float] = None
    delinquency_90day_count: int = Field(default=0, ge=0)
    delinquency_30day_count: int = Field(default=0, ge=0)
    external_credit_score_proxy: Optional[float] = Field(None, ge=0, le=1)
    risk_tier: str
    is_pilot: bool = False
    is_boundary_case: bool = False
    boundary_threshold: Optional[str] = None
    sampling_notes: Optional[str] = None


class ApplicantResponse(ApplicantCreate, TimestampMixin):
    applicant_id: uuid.UUID
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ApplicantSummary(BaseModel):
    applicant_id: uuid.UUID
    applicant_code: str
    applicant_name: str
    risk_tier: str
    is_pilot: bool
    is_boundary_case: bool
    annual_income: float
    debt_to_income_ratio: float
    loan_amount_requested: float

    class Config:
        from_attributes = True


# ============================================================
# PD MODEL SCHEMAS
# ============================================================
class PDModelCreate(BaseModel):
    model_name: str
    model_version: str
    model_type: str = "LogisticRegression"
    training_dataset: str
    training_dataset_version: str
    training_config: Optional[Dict[str, Any]] = None


class PDModelResponse(PDModelCreate, TimestampMixin):
    pd_model_id: uuid.UUID
    roc_auc: Optional[float] = None
    brier_score: Optional[float] = None
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    ks_statistic: Optional[float] = None
    model_artifact_path: Optional[str] = None
    feature_list: Optional[List[str]] = None
    validation_results: Optional[Dict[str, Any]] = None
    is_frozen: bool = False
    frozen_at: Optional[datetime] = None
    is_active: bool = True

    class Config:
        from_attributes = True


class PDScoreResponse(BaseModel):
    score_id: uuid.UUID
    applicant_id: uuid.UUID
    pd_model_id: uuid.UUID
    pd_score: float
    pd_score_frozen: float
    is_frozen: bool
    frozen_at: Optional[datetime] = None
    feature_values: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


# ============================================================
# POLICY SCHEMAS
# ============================================================
class PolicyCreate(BaseModel):
    policy_code: str = Field(..., max_length=50)
    policy_name: str
    policy_version: str
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    status: str = "current"
    is_authoritative: bool = True
    pd_approve_threshold: float = Field(default=0.05, ge=0, le=1)
    pd_decline_threshold: float = Field(default=0.10, ge=0, le=1)
    dti_review_threshold: float = Field(default=40.0, ge=0, le=100)
    dti_decline_threshold: Optional[float] = None
    delinquency_decline_rules: Optional[Dict[str, Any]] = None
    additional_rules: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    @model_validator(mode="after")
    def check_pd_thresholds(self) -> "PolicyCreate":
        if self.pd_approve_threshold >= self.pd_decline_threshold:
            raise ValueError("pd_approve_threshold must be less than pd_decline_threshold")
        return self


class PolicyResponse(PolicyCreate, TimestampMixin):
    policy_id: uuid.UUID

    class Config:
        from_attributes = True


class PolicyDocumentCreate(BaseModel):
    document_code: str = Field(..., max_length=50)
    policy_id: Optional[uuid.UUID] = None
    title: str
    version: str
    effective_date: datetime
    expiry_date: Optional[datetime] = None
    status: str = "current"
    is_authoritative: bool = True
    policy_category: str
    content: str
    summary: Optional[str] = None
    source_metadata: Optional[Dict[str, Any]] = None


class PolicyDocumentResponse(PolicyDocumentCreate, TimestampMixin):
    document_id: uuid.UUID
    source_file_path: Optional[str] = None

    class Config:
        from_attributes = True


# ============================================================
# REFERENCE DECISION SCHEMAS
# ============================================================
class ReferenceDecisionResponse(BaseModel):
    decision_id: uuid.UUID
    applicant_id: uuid.UUID
    pd_score: float
    dti_ratio: float
    reference_decision: str
    triggered_rule: str
    decision_explanation: str
    rule_details: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================
# PROMPT SCHEMAS
# ============================================================
class PromptCreate(BaseModel):
    prompt_code: str = Field(..., max_length=50)
    prompt_name: str
    prompt_version: str
    prompt_type: str
    system_prompt: str
    user_prompt_template: str
    description: Optional[str] = None
    change_summary: Optional[str] = None
    is_baseline: bool = False


class PromptResponse(PromptCreate, TimestampMixin):
    prompt_id: uuid.UUID
    prompt_hash: str
    is_active: bool = True

    class Config:
        from_attributes = True


# ============================================================
# GENAI MODEL SCHEMAS
# ============================================================
class GenAIModelCreate(BaseModel):
    provider: str
    model_name: str
    model_version: str
    display_name: str
    model_metadata: Optional[Dict[str, Any]] = None


class GenAIModelResponse(GenAIModelCreate, TimestampMixin):
    model_id: uuid.UUID
    is_active: bool = True

    class Config:
        from_attributes = True


# ============================================================
# EXPERIMENT SCHEMAS
# ============================================================
class ExperimentDefinitionCreate(BaseModel):
    experiment_code: str
    experiment_name: str
    experiment_type: str
    description: str
    hypothesis: Optional[str] = None
    fixed_variables: List[str]
    varied_variable: str
    expected_controlled_vars: Optional[Dict[str, Any]] = None


class ExperimentRunRequest(BaseModel):
    experiment_def_id: uuid.UUID
    applicant_id: uuid.UUID
    prompt_id: uuid.UUID
    genai_model_id: uuid.UUID
    policy_scenario: Optional[str] = "current_only"
    baseline_run_id: Optional[uuid.UUID] = None
    is_baseline_run: bool = False
    retrieval_config: Optional[Dict[str, Any]] = None
    model_parameters: Optional[Dict[str, Any]] = None


class ExperimentRunResponse(BaseModel):
    run_id: uuid.UUID
    experiment_def_id: uuid.UUID
    applicant_id: uuid.UUID
    status: str
    fixed_pd_score: Optional[float] = None
    genai_decision: Optional[str] = None
    reference_decision: Optional[str] = None
    agrees_with_reference: Optional[bool] = None
    severity: Optional[str] = None
    deviation_classification: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================
# GENAI STRUCTURED OUTPUT SCHEMA
# ============================================================
class RuntimeConfiguration(BaseModel):
    prompt_version: str
    model: str
    model_version: str


class GenAIStructuredAssessment(BaseModel):
    """The structured JSON output expected from the GenAI assistant."""
    applicant_id: str
    pd_score: float = Field(..., ge=0, le=1)
    risk_classification: str
    recommendation: str = Field(..., pattern="^(APPROVE|MANUAL_REVIEW|DECLINE)$")
    pd_interpretation: str
    primary_risk_factors: List[str]
    supporting_applicant_evidence: List[str]
    policy_references: List[str]
    policy_conflict_flag: bool = False
    unsupported_conclusion_flag: bool = False
    human_review_required: bool = False
    reasoning_summary: str
    runtime_configuration: RuntimeConfiguration

    @field_validator("pd_score")
    @classmethod
    def validate_pd_score(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError("pd_score must be between 0 and 1")
        return round(v, 6)


# ============================================================
# EVALUATION SCHEMAS
# ============================================================
class EvaluationResultResponse(BaseModel):
    eval_id: uuid.UUID
    experiment_run_id: uuid.UUID
    reference_decision: Optional[str] = None
    genai_decision: Optional[str] = None
    agrees_with_reference: Optional[bool] = None
    decision_changed: Optional[bool] = None
    risk_classification_changed: Optional[bool] = None
    policy_conflict_detected: Optional[bool] = None
    unsupported_conclusion: Optional[bool] = None
    obsolete_policy_used: Optional[bool] = None
    failed_to_escalate: Optional[bool] = None
    deviation_classification: Optional[str] = None
    materiality: Optional[str] = None
    severity: Optional[str] = None
    transition: Optional[str] = None
    evaluation_notes: Optional[str] = None
    evaluated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================
# METRICS SCHEMAS
# ============================================================
class AggregateMetrics(BaseModel):
    experiment_type: str
    total_runs: int
    total_applicants: int
    reference_decision_agreement_rate: float
    output_stability_rate: Optional[float] = None
    material_decision_variation_rate: float
    decision_reversal_rate: float
    pd_interpretation_accuracy_rate: float
    policy_consistency_rate: float
    unsupported_conclusion_rate: float
    policy_conflict_detection_rate: Optional[float] = None
    severity_distribution: Dict[str, int]
    deviation_classification_distribution: Dict[str, int]
    transition_matrix: Dict[str, Dict[str, int]]


class DecisionTransitionRow(BaseModel):
    baseline_decision: str
    variant_decision: str
    count: int
    percentage: float
    expected_count: int
    unexpected_count: int
    critical_count: int


# ============================================================
# RAG / RETRIEVAL SCHEMAS
# ============================================================
class RetrievalResult(BaseModel):
    document_id: uuid.UUID
    document_code: str
    title: str
    version: str
    effective_date: datetime
    status: str
    is_authoritative: bool
    policy_category: str
    retrieval_rank: int
    retrieval_score: Optional[float] = None
    content: str
    summary: Optional[str] = None


class RetrievalResponse(BaseModel):
    query: str
    policy_scenario: str
    retrieved_documents: List[RetrievalResult]
    total_retrieved: int
    retrieval_metadata: Optional[Dict[str, Any]] = None


# ============================================================
# API RESPONSE WRAPPERS
# ============================================================
class APIResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Optional[Any] = None


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    page_size: int
    pages: int
