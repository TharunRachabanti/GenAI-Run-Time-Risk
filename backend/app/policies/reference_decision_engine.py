"""
Policy Reference Decision Engine
Calculates deterministic, policy-based reference decisions for every applicant.

The reference decision is the benchmark against which GenAI output is evaluated.
It is calculated mechanically from the policy rules — no ML, no GenAI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# DECISION ENUM
# ============================================================
class Decision(str, Enum):
    APPROVE = "APPROVE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DECLINE = "DECLINE"


# ============================================================
# POLICY RULES DATACLASS
# ============================================================
@dataclass
class PolicyRules:
    """
    Represents the active rules for a given policy version.
    All thresholds are stored explicitly for full traceability.
    """
    policy_id: str
    policy_version: str
    effective_date: str

    # PD thresholds
    pd_approve_threshold: float = 0.05    # PD < this → APPROVE
    pd_decline_threshold: float = 0.10    # PD > this → DECLINE
    # Between approve and decline thresholds → MANUAL_REVIEW

    # DTI thresholds
    dti_review_threshold: float = 40.0    # DTI > this → MANUAL_REVIEW (override)
    dti_decline_threshold: Optional[float] = None  # DTI > this → DECLINE

    # Delinquency rules
    delinquency_90day_decline_count: int = 1  # Any 90-day delinquency → DECLINE
    delinquency_30day_review_count: int = 2   # 2+ 30-day delinquencies → MANUAL_REVIEW

    # Override rules
    serious_delinquency_override: bool = True  # Delinquency can override otherwise-OK PD


@dataclass
class RuleResult:
    """Records which specific rule triggered the reference decision."""
    rule_id: str
    rule_description: str
    threshold_type: str
    threshold_value: float
    applicant_value: float
    triggered: bool
    resulting_decision: Optional[Decision] = None


@dataclass
class ReferenceDecisionResult:
    """Complete output of the reference decision calculation."""
    applicant_id: str
    pd_score: float
    dti_ratio: float
    delinquency_90day_count: int
    delinquency_30day_count: int

    reference_decision: Decision
    triggered_rule: str
    rule_details: List[RuleResult]
    decision_explanation: str
    policy_id: str
    policy_version: str


# ============================================================
# REFERENCE DECISION ENGINE
# ============================================================
class ReferenceDecisionEngine:
    """
    Implements the deterministic policy-based reference decision engine.

    Rules are evaluated in priority order:
    1. Delinquency (highest override priority)
    2. PD threshold (primary credit risk rule)
    3. DTI (secondary review trigger)

    The result is always one of: APPROVE / MANUAL_REVIEW / DECLINE

    IMPORTANT: This engine is the gold-standard benchmark.
    Any GenAI deviation from this benchmark is investigated for:
    - expected/unexpected classification
    - materiality
    - severity
    """

    def __init__(self, rules: PolicyRules):
        self.rules = rules

    def calculate(
        self,
        applicant_id: str,
        pd_score: float,
        dti_ratio: float,
        delinquency_90day_count: int = 0,
        delinquency_30day_count: int = 0,
        additional_fields: Optional[Dict[str, Any]] = None,
    ) -> ReferenceDecisionResult:
        """
        Calculate the deterministic policy-based reference decision.

        Args:
            applicant_id: Unique applicant identifier
            pd_score: Fixed, frozen PD score (0.0–1.0)
            dti_ratio: Debt-to-income ratio (%)
            delinquency_90day_count: Count of 90-day delinquencies in last 24 months
            delinquency_30day_count: Count of 30-day delinquencies in last 24 months
            additional_fields: Optional extra applicant fields

        Returns:
            ReferenceDecisionResult with full audit trail
        """
        if not 0 <= pd_score <= 1:
            raise ValueError(f"pd_score must be 0.0–1.0, got {pd_score}")
        if dti_ratio < 0:
            raise ValueError(f"dti_ratio must be >= 0, got {dti_ratio}")

        rules_evaluated: List[RuleResult] = []
        final_decision: Optional[Decision] = None
        triggered_rule: str = ""
        decision_explanation: str = ""

        # ---- RULE 1: SERIOUS DELINQUENCY (90-day) ----
        if (
            self.rules.serious_delinquency_override
            and delinquency_90day_count >= self.rules.delinquency_90day_decline_count
        ):
            rule = RuleResult(
                rule_id="DELINQUENCY_90_DECLINE",
                rule_description=f"90-day delinquency count ({delinquency_90day_count}) >= "
                                  f"threshold ({self.rules.delinquency_90day_decline_count})",
                threshold_type="delinquency_90day",
                threshold_value=self.rules.delinquency_90day_decline_count,
                applicant_value=delinquency_90day_count,
                triggered=True,
                resulting_decision=Decision.DECLINE,
            )
            rules_evaluated.append(rule)
            final_decision = Decision.DECLINE
            triggered_rule = "DELINQUENCY_90_DECLINE"
            decision_explanation = (
                f"DECLINE: Applicant has {delinquency_90day_count} serious 90-day delinquency event(s) "
                f"within the past 24 months. Policy Rule DELINQUENCY_90_DECLINE requires DECLINE "
                f"when 90-day delinquencies >= {self.rules.delinquency_90day_decline_count}. "
                f"This rule takes priority over PD and DTI thresholds."
            )
        else:
            rules_evaluated.append(RuleResult(
                rule_id="DELINQUENCY_90_DECLINE",
                rule_description="90-day delinquency check (not triggered)",
                threshold_type="delinquency_90day",
                threshold_value=self.rules.delinquency_90day_decline_count,
                applicant_value=delinquency_90day_count,
                triggered=False,
            ))

        # ---- RULE 2: PD DECLINE ----
        if final_decision is None:
            if pd_score > self.rules.pd_decline_threshold:
                rule = RuleResult(
                    rule_id="PD_DECLINE",
                    rule_description=f"PD score ({pd_score:.4f}) > decline threshold ({self.rules.pd_decline_threshold:.4f})",
                    threshold_type="pd_decline",
                    threshold_value=self.rules.pd_decline_threshold,
                    applicant_value=pd_score,
                    triggered=True,
                    resulting_decision=Decision.DECLINE,
                )
                rules_evaluated.append(rule)
                final_decision = Decision.DECLINE
                triggered_rule = "PD_DECLINE"
                decision_explanation = (
                    f"DECLINE: PD score ({pd_score:.2%}) exceeds the decline threshold "
                    f"({self.rules.pd_decline_threshold:.2%}). "
                    f"Policy v{self.rules.policy_version} Rule PD_DECLINE: "
                    f"PD > {self.rules.pd_decline_threshold:.2%} → DECLINE."
                )
            else:
                rules_evaluated.append(RuleResult(
                    rule_id="PD_DECLINE",
                    rule_description="PD decline check (not triggered)",
                    threshold_type="pd_decline",
                    threshold_value=self.rules.pd_decline_threshold,
                    applicant_value=pd_score,
                    triggered=False,
                ))

        # ---- RULE 3: DTI DECLINE (if configured) ----
        if final_decision is None and self.rules.dti_decline_threshold is not None:
            if dti_ratio > self.rules.dti_decline_threshold:
                rule = RuleResult(
                    rule_id="DTI_DECLINE",
                    rule_description=f"DTI ({dti_ratio:.1f}%) > DTI decline threshold ({self.rules.dti_decline_threshold:.1f}%)",
                    threshold_type="dti_decline",
                    threshold_value=self.rules.dti_decline_threshold,
                    applicant_value=dti_ratio,
                    triggered=True,
                    resulting_decision=Decision.DECLINE,
                )
                rules_evaluated.append(rule)
                final_decision = Decision.DECLINE
                triggered_rule = "DTI_DECLINE"
                decision_explanation = (
                    f"DECLINE: DTI ratio ({dti_ratio:.1f}%) exceeds the DTI decline threshold "
                    f"({self.rules.dti_decline_threshold:.1f}%)."
                )

        # ---- RULE 4: MODERATE 30-DAY DELINQUENCY → MANUAL REVIEW ----
        if final_decision is None:
            if delinquency_30day_count >= self.rules.delinquency_30day_review_count:
                rule = RuleResult(
                    rule_id="DELINQUENCY_30_REVIEW",
                    rule_description=f"30-day delinquencies ({delinquency_30day_count}) >= "
                                      f"review threshold ({self.rules.delinquency_30day_review_count})",
                    threshold_type="delinquency_30day",
                    threshold_value=self.rules.delinquency_30day_review_count,
                    applicant_value=delinquency_30day_count,
                    triggered=True,
                    resulting_decision=Decision.MANUAL_REVIEW,
                )
                rules_evaluated.append(rule)
                final_decision = Decision.MANUAL_REVIEW
                triggered_rule = "DELINQUENCY_30_REVIEW"
                decision_explanation = (
                    f"MANUAL_REVIEW: Applicant has {delinquency_30day_count} 30-day delinquency event(s), "
                    f"meeting the threshold for manual review ({self.rules.delinquency_30day_review_count}+)."
                )
            else:
                rules_evaluated.append(RuleResult(
                    rule_id="DELINQUENCY_30_REVIEW",
                    rule_description="30-day delinquency review check (not triggered)",
                    threshold_type="delinquency_30day",
                    threshold_value=self.rules.delinquency_30day_review_count,
                    applicant_value=delinquency_30day_count,
                    triggered=False,
                ))

        # ---- RULE 5: PD MANUAL REVIEW ----
        if final_decision is None:
            if pd_score >= self.rules.pd_approve_threshold:
                rule = RuleResult(
                    rule_id="PD_MANUAL_REVIEW",
                    rule_description=(
                        f"PD score ({pd_score:.4f}) > approve threshold ({self.rules.pd_approve_threshold:.4f}) "
                        f"and <= decline threshold ({self.rules.pd_decline_threshold:.4f})"
                    ),
                    threshold_type="pd_manual_review",
                    threshold_value=self.rules.pd_approve_threshold,
                    applicant_value=pd_score,
                    triggered=True,
                    resulting_decision=Decision.MANUAL_REVIEW,
                )
                rules_evaluated.append(rule)
                final_decision = Decision.MANUAL_REVIEW
                triggered_rule = "PD_MANUAL_REVIEW"
                decision_explanation = (
                    f"MANUAL_REVIEW: PD score ({pd_score:.2%}) is between the approve threshold "
                    f"({self.rules.pd_approve_threshold:.2%}) and the decline threshold "
                    f"({self.rules.pd_decline_threshold:.2%}). "
                    f"Policy v{self.rules.policy_version} Rule PD_MANUAL_REVIEW applies."
                )
            else:
                rules_evaluated.append(RuleResult(
                    rule_id="PD_MANUAL_REVIEW",
                    rule_description="PD manual review check (not triggered)",
                    threshold_type="pd_manual_review",
                    threshold_value=self.rules.pd_approve_threshold,
                    applicant_value=pd_score,
                    triggered=False,
                ))

        # ---- RULE 6: DTI MANUAL REVIEW ----
        if final_decision is None or final_decision == Decision.APPROVE:
            if dti_ratio > self.rules.dti_review_threshold:
                rule = RuleResult(
                    rule_id="DTI_MANUAL_REVIEW",
                    rule_description=f"DTI ({dti_ratio:.1f}%) > DTI review threshold ({self.rules.dti_review_threshold:.1f}%)",
                    threshold_type="dti_review",
                    threshold_value=self.rules.dti_review_threshold,
                    applicant_value=dti_ratio,
                    triggered=True,
                    resulting_decision=Decision.MANUAL_REVIEW,
                )
                rules_evaluated.append(rule)
                if final_decision != Decision.DECLINE:  # DTI review cannot override a DECLINE
                    final_decision = Decision.MANUAL_REVIEW
                    triggered_rule = "DTI_MANUAL_REVIEW"
                    decision_explanation = (
                        f"MANUAL_REVIEW: DTI ratio ({dti_ratio:.1f}%) exceeds the policy review threshold "
                        f"({self.rules.dti_review_threshold:.1f}%). "
                        f"Policy v{self.rules.policy_version} Rule DTI_MANUAL_REVIEW: "
                        f"DTI > {self.rules.dti_review_threshold:.1f}% → MANUAL_REVIEW."
                    )
            else:
                rules_evaluated.append(RuleResult(
                    rule_id="DTI_MANUAL_REVIEW",
                    rule_description="DTI review check (not triggered)",
                    threshold_type="dti_review",
                    threshold_value=self.rules.dti_review_threshold,
                    applicant_value=dti_ratio,
                    triggered=False,
                ))

        # ---- RULE 7: APPROVE (default — no rules triggered) ----
        if final_decision is None:
            rule = RuleResult(
                rule_id="PD_APPROVE",
                rule_description=f"PD score ({pd_score:.4f}) <= approve threshold ({self.rules.pd_approve_threshold:.4f}) "
                                   f"and no adverse rules triggered",
                threshold_type="pd_approve",
                threshold_value=self.rules.pd_approve_threshold,
                applicant_value=pd_score,
                triggered=True,
                resulting_decision=Decision.APPROVE,
            )
            rules_evaluated.append(rule)
            final_decision = Decision.APPROVE
            triggered_rule = "PD_APPROVE"
            decision_explanation = (
                f"APPROVE: PD score ({pd_score:.2%}) is at or below the approve threshold "
                f"({self.rules.pd_approve_threshold:.2%}), DTI ({dti_ratio:.1f}%) is within limits, "
                f"and no adverse delinquency rules were triggered. "
                f"Policy v{self.rules.policy_version} Rule PD_APPROVE applies."
            )

        return ReferenceDecisionResult(
            applicant_id=applicant_id,
            pd_score=pd_score,
            dti_ratio=dti_ratio,
            delinquency_90day_count=delinquency_90day_count,
            delinquency_30day_count=delinquency_30day_count,
            reference_decision=final_decision,
            triggered_rule=triggered_rule,
            rule_details=rules_evaluated,
            decision_explanation=decision_explanation,
            policy_id=self.rules.policy_id,
            policy_version=self.rules.policy_version,
        )


# ============================================================
# DEFAULT POLICY RULES (v3.0 — Current)
# ============================================================
CURRENT_POLICY_RULES = PolicyRules(
    policy_id="POLICY-CL-001",
    policy_version="3.0",
    effective_date="2024-01-01",
    pd_approve_threshold=0.05,
    pd_decline_threshold=0.10,
    dti_review_threshold=40.0,
    dti_decline_threshold=None,
    delinquency_90day_decline_count=1,
    delinquency_30day_review_count=2,
    serious_delinquency_override=True,
)

SUPERSEDED_POLICY_RULES = PolicyRules(
    policy_id="POLICY-CL-001",
    policy_version="2.0",
    effective_date="2022-01-01",
    pd_approve_threshold=0.07,   # Higher (more lenient) in old policy
    pd_decline_threshold=0.15,   # Higher (more lenient) in old policy
    dti_review_threshold=45.0,   # Higher (more lenient) in old policy
    dti_decline_threshold=None,
    delinquency_90day_decline_count=2,  # Required 2 serious delinquencies (more lenient)
    delinquency_30day_review_count=3,
    serious_delinquency_override=True,
)


def get_engine_for_policy(policy_version: str) -> ReferenceDecisionEngine:
    """Factory to get the correct engine for a given policy version."""
    if policy_version == "3.0":
        return ReferenceDecisionEngine(CURRENT_POLICY_RULES)
    elif policy_version == "2.0":
        return ReferenceDecisionEngine(SUPERSEDED_POLICY_RULES)
    else:
        raise ValueError(f"Unknown policy version: {policy_version}")
