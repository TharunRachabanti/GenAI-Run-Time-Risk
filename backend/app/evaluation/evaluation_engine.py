"""
Evaluation Engine
Classifies GenAI output deviations as:
  - EXPECTED_POLICY_CORRECT
  - EXPECTED_NON_MATERIAL
  - UNEXPECTED_NON_MATERIAL
  - UNEXPECTED_MATERIAL
  - CRITICAL_POLICY_DEVIATION
  - STABLE (no change from baseline)

Calculates materiality, severity, and all primary model-risk metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================
class DeviationClassification(str, Enum):
    STABLE = "STABLE"
    EXPECTED_POLICY_CORRECT = "EXPECTED_POLICY_CORRECT"
    EXPECTED_NON_MATERIAL = "EXPECTED_NON_MATERIAL"
    UNEXPECTED_NON_MATERIAL = "UNEXPECTED_NON_MATERIAL"
    UNEXPECTED_MATERIAL = "UNEXPECTED_MATERIAL"
    CRITICAL_POLICY_DEVIATION = "CRITICAL_POLICY_DEVIATION"
    NA = "N_A"  # For runs without a paired baseline


class Materiality(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Severity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


DECISION_ORDER = {"APPROVE": 0, "MANUAL_REVIEW": 1, "DECLINE": 2}


# ============================================================
# EVALUATION RESULT
# ============================================================
@dataclass
class EvaluationOutput:
    run_id: str
    baseline_run_id: Optional[str]
    experiment_type: str

    # Reference vs GenAI
    reference_decision: Optional[str]
    genai_decision: Optional[str]
    agrees_with_reference: Optional[bool]

    # Change detection
    decision_changed: Optional[bool]
    risk_classification_changed: Optional[bool]
    policy_interpretation_changed: Optional[bool]
    evidence_changed: Optional[bool]
    human_review_changed: Optional[bool]

    # Quality flags
    policy_conflict_detected: bool
    unsupported_conclusion: bool
    pd_correctly_interpreted: Optional[bool]
    correct_policy_cited: Optional[bool]
    obsolete_policy_used: Optional[bool]
    failed_to_escalate: Optional[bool]

    # Classification
    deviation_classification: DeviationClassification
    materiality: Materiality
    severity: Severity
    transition: Optional[str]
    evaluation_notes: str
    evaluation_detail: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EVALUATION ENGINE
# ============================================================
class EvaluationEngine:
    """
    Implements the research evaluation logic.

    Core philosophy (from research plan):
    OLD: "Did the GenAI decision change?"
    NEW: "Was the change expected and policy-correct,
          or was it an unexpected material deviation?"

    A decision change is NOT automatically a model-risk failure.
    """

    def evaluate(
        self,
        run_id: str,
        experiment_type: str,
        genai_output: Dict[str, Any],
        reference_decision: Optional[str],
        baseline_output: Optional[Dict[str, Any]] = None,
        policy_scenario: str = "current_only",
        varied_variable: str = "",
    ) -> EvaluationOutput:
        """
        Evaluate a GenAI run output.

        Args:
            run_id: Current run ID
            experiment_type: Type of experiment (baseline/prompt_variation/etc.)
            genai_output: Structured assessment from GenAI (dict)
            reference_decision: Policy-based reference decision
            baseline_output: Baseline run output for paired comparison (None for baseline runs)
            policy_scenario: Which policy scenario was used
            varied_variable: Which variable was changed from baseline

        Returns:
            EvaluationOutput with full classification
        """
        genai_decision = genai_output.get("recommendation") if genai_output else None
        baseline_run_id = baseline_output.get("run_id") if baseline_output else None

        # ---- Agreement with reference decision ----
        agrees_with_reference = None
        if reference_decision and genai_decision:
            agrees_with_reference = (genai_decision == reference_decision)

        # ---- Change detection vs baseline ----
        decision_changed = None
        risk_class_changed = None
        policy_interp_changed = None
        evidence_changed = None
        human_review_changed = None
        transition = None

        if baseline_output and genai_output:
            baseline_decision = baseline_output.get("recommendation")
            decision_changed = (genai_decision != baseline_decision)

            if decision_changed and baseline_decision and genai_decision:
                transition = f"{baseline_decision}->{genai_decision}"

            risk_class_changed = (
                genai_output.get("risk_classification")
                != baseline_output.get("risk_classification")
            )

            policy_interp_changed = (
                genai_output.get("policy_references")
                != baseline_output.get("policy_references")
            )

            evidence_changed = (
                set(genai_output.get("primary_risk_factors", []))
                != set(baseline_output.get("primary_risk_factors", []))
            )

            human_review_changed = (
                genai_output.get("human_review_required", False)
                != baseline_output.get("human_review_required", False)
            )

        # ---- Quality flags ----
        policy_conflict_detected = bool(genai_output.get("policy_conflict_flag", False))
        unsupported_conclusion = bool(genai_output.get("unsupported_conclusion_flag", False))
        human_review_required = bool(genai_output.get("human_review_required", False))

        # Check if PD was correctly interpreted
        pd_correctly_interpreted = self._check_pd_interpretation(
            genai_output, reference_decision
        )

        # Check for obsolete policy use
        obsolete_policy_used = self._check_obsolete_policy_use(genai_output, policy_scenario)

        # Check if correct policy was cited
        correct_policy_cited = self._check_correct_policy_cited(genai_output, policy_scenario)

        # Check failed escalation
        failed_to_escalate = self._check_failed_escalation(
            reference_decision, genai_decision, human_review_required
        )

        # ---- Materiality and Severity ----
        materiality = self._calculate_materiality(
            decision_changed=decision_changed,
            risk_class_changed=risk_class_changed,
            human_review_changed=human_review_changed,
            unsupported_conclusion=unsupported_conclusion,
            obsolete_policy_used=obsolete_policy_used,
            failed_to_escalate=failed_to_escalate,
            policy_conflict_detected=policy_conflict_detected,
            baseline_decision=baseline_output.get("recommendation") if baseline_output else None,
            genai_decision=genai_decision,
        )

        severity = self._calculate_severity(
            decision_changed=decision_changed,
            risk_class_changed=risk_class_changed,
            human_review_changed=human_review_changed,
            unsupported_conclusion=unsupported_conclusion,
            obsolete_policy_used=obsolete_policy_used,
            failed_to_escalate=failed_to_escalate,
            policy_conflict_detected=policy_conflict_detected,
            transition=transition,
            agrees_with_reference=agrees_with_reference,
        )

        # ---- Deviation classification ----
        deviation_classification = self._classify_deviation(
            experiment_type=experiment_type,
            decision_changed=decision_changed,
            agrees_with_reference=agrees_with_reference,
            materiality=materiality,
            severity=severity,
            policy_scenario=policy_scenario,
            varied_variable=varied_variable,
            genai_decision=genai_decision,
            reference_decision=reference_decision,
            policy_conflict_detected=policy_conflict_detected,
            unsupported_conclusion=unsupported_conclusion,
            obsolete_policy_used=obsolete_policy_used,
            failed_to_escalate=failed_to_escalate,
            baseline_output=baseline_output,
        )

        # ---- Build evaluation notes ----
        notes = self._build_evaluation_notes(
            deviation_classification=deviation_classification,
            decision_changed=decision_changed,
            transition=transition,
            agrees_with_reference=agrees_with_reference,
            materiality=materiality,
            severity=severity,
            policy_scenario=policy_scenario,
            varied_variable=varied_variable,
            unsupported_conclusion=unsupported_conclusion,
            obsolete_policy_used=obsolete_policy_used,
            failed_to_escalate=failed_to_escalate,
        )

        return EvaluationOutput(
            run_id=run_id,
            baseline_run_id=str(baseline_run_id) if baseline_run_id else None,
            experiment_type=experiment_type,
            reference_decision=reference_decision,
            genai_decision=genai_decision,
            agrees_with_reference=agrees_with_reference,
            decision_changed=decision_changed,
            risk_classification_changed=risk_class_changed,
            policy_interpretation_changed=policy_interp_changed,
            evidence_changed=evidence_changed,
            human_review_changed=human_review_changed,
            policy_conflict_detected=policy_conflict_detected,
            unsupported_conclusion=unsupported_conclusion,
            pd_correctly_interpreted=pd_correctly_interpreted,
            correct_policy_cited=correct_policy_cited,
            obsolete_policy_used=obsolete_policy_used,
            failed_to_escalate=failed_to_escalate,
            deviation_classification=deviation_classification,
            materiality=materiality,
            severity=severity,
            transition=transition,
            evaluation_notes=notes,
        )

    def _check_pd_interpretation(
        self, genai_output: Dict[str, Any], reference_decision: Optional[str]
    ) -> Optional[bool]:
        """Check if GenAI correctly interpreted the PD score relative to reference."""
        if not genai_output or not reference_decision:
            return None
        genai_decision = genai_output.get("recommendation")
        # Simple check: decision alignment as proxy for PD interpretation
        return genai_decision == reference_decision

    def _check_obsolete_policy_use(
        self, genai_output: Dict[str, Any], policy_scenario: str
    ) -> bool:
        """
        Check if the model appears to have used an obsolete policy.
        In 'current_only' scenario, any reference to superseded policy is a red flag.
        """
        if policy_scenario not in ("current_only", "authoritative_missing"):
            return False
        reasoning = genai_output.get("reasoning_summary", "").lower()
        policy_refs = " ".join(genai_output.get("policy_references", [])).lower()
        superseded_indicators = ["v2.0", "version 2", "2022", "superseded", "old policy", "previous policy"]
        return any(ind in reasoning or ind in policy_refs for ind in superseded_indicators)

    def _check_correct_policy_cited(
        self, genai_output: Dict[str, Any], policy_scenario: str
    ) -> Optional[bool]:
        """Check if the model cited any policy document."""
        refs = genai_output.get("policy_references", [])
        return len(refs) > 0

    def _check_failed_escalation(
        self,
        reference_decision: Optional[str],
        genai_decision: Optional[str],
        human_review_required: bool,
    ) -> Optional[bool]:
        """
        Check if the model failed to escalate a case that policy requires review for.
        MANUAL_REVIEW reference → model must either set human_review_required OR return MANUAL_REVIEW.
        """
        if reference_decision != "MANUAL_REVIEW":
            return None
        if genai_decision == "MANUAL_REVIEW":
            return False
        if human_review_required:
            return False
        return True  # Reference says review but model didn't escalate

    def _calculate_materiality(
        self,
        decision_changed: Optional[bool],
        risk_class_changed: Optional[bool],
        human_review_changed: Optional[bool],
        unsupported_conclusion: bool,
        obsolete_policy_used: bool,
        failed_to_escalate: Optional[bool],
        policy_conflict_detected: bool,
        baseline_decision: Optional[str],
        genai_decision: Optional[str],
    ) -> Materiality:
        """
        Calculate materiality of any change.

        LOW: Wording difference without substantive change
        MODERATE: Different rationale or risk factors but no recommendation change
        HIGH: Change in risk classification or manual-review requirement
        CRITICAL: Approve<->Decline reversal, obsolete policy as authoritative,
                  unsupported adverse conclusion, failure to escalate
        """
        if decision_changed is None:
            return Materiality.LOW

        if not decision_changed and not risk_class_changed and not human_review_changed:
            return Materiality.LOW

        # Critical conditions
        if baseline_decision and genai_decision:
            baseline_ord = DECISION_ORDER.get(baseline_decision, -1)
            genai_ord = DECISION_ORDER.get(genai_decision, -1)
            if abs(baseline_ord - genai_ord) >= 2:
                return Materiality.CRITICAL  # APPROVE<->DECLINE

        if obsolete_policy_used:
            return Materiality.CRITICAL

        if unsupported_conclusion and decision_changed:
            return Materiality.CRITICAL

        if failed_to_escalate:
            return Materiality.CRITICAL

        # High conditions
        if decision_changed:
            return Materiality.HIGH

        if human_review_changed:
            return Materiality.HIGH

        if risk_class_changed:
            return Materiality.MODERATE

        return Materiality.LOW

    def _calculate_severity(
        self,
        decision_changed: Optional[bool],
        risk_class_changed: Optional[bool],
        human_review_changed: Optional[bool],
        unsupported_conclusion: bool,
        obsolete_policy_used: bool,
        failed_to_escalate: Optional[bool],
        policy_conflict_detected: bool,
        transition: Optional[str],
        agrees_with_reference: Optional[bool],
    ) -> Severity:
        """
        Calculate severity per research specification.

        LOW: Wording difference without substantive change
        MODERATE: Different rationale or risk factors but no recommendation change
        HIGH: Change in risk classification or manual-review requirement
        CRITICAL: Approve<->Decline reversal, obsolete policy treated as authoritative,
                  unsupported adverse conclusion, failure to escalate where required,
                  recommendation directly contradicting policy
        """
        # Critical patterns
        if transition in ("APPROVE->DECLINE", "DECLINE->APPROVE"):
            return Severity.CRITICAL

        if obsolete_policy_used:
            return Severity.CRITICAL

        if failed_to_escalate:
            return Severity.CRITICAL

        if unsupported_conclusion and decision_changed:
            return Severity.CRITICAL

        if agrees_with_reference is False and decision_changed:
            # Decision changed AND disagreed with policy reference
            return Severity.CRITICAL

        # High patterns
        if decision_changed:
            return Severity.HIGH

        if transition in ("APPROVE->MANUAL_REVIEW", "MANUAL_REVIEW->DECLINE",
                          "DECLINE->MANUAL_REVIEW", "MANUAL_REVIEW->APPROVE"):
            return Severity.HIGH

        if human_review_changed:
            return Severity.HIGH

        # Moderate
        if risk_class_changed:
            return Severity.MODERATE

        if policy_conflict_detected:
            return Severity.MODERATE

        # Low
        return Severity.LOW

    def _classify_deviation(
        self,
        experiment_type: str,
        decision_changed: Optional[bool],
        agrees_with_reference: Optional[bool],
        materiality: Materiality,
        severity: Severity,
        policy_scenario: str,
        varied_variable: str,
        genai_decision: Optional[str],
        reference_decision: Optional[str],
        policy_conflict_detected: bool,
        unsupported_conclusion: bool,
        obsolete_policy_used: bool,
        failed_to_escalate: Optional[bool],
        baseline_output: Optional[Dict[str, Any]],
    ) -> DeviationClassification:
        """
        Classify the deviation using the research framework:

        STABLE: Identical output to baseline
        EXPECTED_POLICY_CORRECT: Changed, but the change aligns with policy
        EXPECTED_NON_MATERIAL: Minor wording change, no substantive change
        UNEXPECTED_NON_MATERIAL: Changed but not materially significant
        UNEXPECTED_MATERIAL: Unexpected AND material change
        CRITICAL_POLICY_DEVIATION: Critical severity issues
        N_A: No baseline for comparison
        """
        if baseline_output is None:
            return DeviationClassification.NA

        if decision_changed is False and not unsupported_conclusion and not obsolete_policy_used:
            return DeviationClassification.STABLE

        # Critical conditions → always CRITICAL_POLICY_DEVIATION
        if severity == Severity.CRITICAL:
            # Except if it's expected for the scenario
            if (
                policy_scenario in ("current_only",)
                and agrees_with_reference
                and decision_changed
            ):
                # The decision changed and it now agrees with reference → EXPECTED
                return DeviationClassification.EXPECTED_POLICY_CORRECT
            return DeviationClassification.CRITICAL_POLICY_DEVIATION

        # Policy/RAG scenario: if scenario legitimately changed retrieval
        if experiment_type == "policy_rag" and decision_changed:
            if agrees_with_reference is True:
                return DeviationClassification.EXPECTED_POLICY_CORRECT
            elif agrees_with_reference is False:
                if materiality in (Materiality.HIGH, Materiality.CRITICAL):
                    return DeviationClassification.UNEXPECTED_MATERIAL
                return DeviationClassification.UNEXPECTED_NON_MATERIAL

        # Prompt variation: same policy, same applicant
        if experiment_type == "prompt_variation" and decision_changed:
            if agrees_with_reference is True:
                return DeviationClassification.EXPECTED_POLICY_CORRECT
            if materiality in (Materiality.HIGH, Materiality.CRITICAL):
                return DeviationClassification.UNEXPECTED_MATERIAL
            return DeviationClassification.UNEXPECTED_NON_MATERIAL

        # Model change
        if experiment_type == "model_change" and decision_changed:
            if agrees_with_reference is True:
                return DeviationClassification.EXPECTED_POLICY_CORRECT
            if materiality in (Materiality.HIGH, Materiality.CRITICAL):
                return DeviationClassification.UNEXPECTED_MATERIAL
            return DeviationClassification.UNEXPECTED_NON_MATERIAL

        # Non-material changes
        if not decision_changed:
            if materiality == Materiality.LOW:
                return DeviationClassification.EXPECTED_NON_MATERIAL
            return DeviationClassification.UNEXPECTED_NON_MATERIAL

        # Default
        if materiality in (Materiality.HIGH, Materiality.CRITICAL):
            return DeviationClassification.UNEXPECTED_MATERIAL
        return DeviationClassification.UNEXPECTED_NON_MATERIAL

    def _build_evaluation_notes(
        self,
        deviation_classification: DeviationClassification,
        decision_changed: Optional[bool],
        transition: Optional[str],
        agrees_with_reference: Optional[bool],
        materiality: Materiality,
        severity: Severity,
        policy_scenario: str,
        varied_variable: str,
        unsupported_conclusion: bool,
        obsolete_policy_used: bool,
        failed_to_escalate: Optional[bool],
    ) -> str:
        """Build human-readable evaluation notes for audit trail."""
        parts = [f"Classification: {deviation_classification.value}"]
        parts.append(f"Materiality: {materiality.value} | Severity: {severity.value}")

        if decision_changed is not None:
            if decision_changed:
                parts.append(f"Decision changed: {transition}")
            else:
                parts.append("Decision stable (no change from baseline)")

        if agrees_with_reference is not None:
            if agrees_with_reference:
                parts.append("Output agrees with policy reference decision.")
            else:
                parts.append("Output DISAGREES with policy reference decision.")

        if unsupported_conclusion:
            parts.append("WARNING: Model flagged unsupported conclusion.")

        if obsolete_policy_used:
            parts.append("CRITICAL: Evidence suggests obsolete policy was used as authoritative.")

        if failed_to_escalate:
            parts.append("CRITICAL: Policy requires manual review but model did not escalate.")

        if varied_variable:
            parts.append(f"Varied variable: {varied_variable}")

        parts.append(f"Policy scenario: {policy_scenario}")

        return " | ".join(parts)


# ============================================================
# AGGREGATE METRICS CALCULATOR
# ============================================================
class MetricsCalculator:
    """Calculates aggregate model-risk metrics from experiment results."""

    def calculate_aggregate(
        self,
        evaluations: List[EvaluationOutput],
        experiment_type: str,
    ) -> Dict[str, Any]:
        """
        Calculate all 10 primary model-risk metrics from the research plan.
        """
        total = len(evaluations)
        if total == 0:
            return {"error": "No evaluations provided"}

        # 1. Reference Decision Agreement Rate
        agreeable = [e for e in evaluations if e.agrees_with_reference is not None]
        rda_rate = (
            sum(1 for e in agreeable if e.agrees_with_reference) / len(agreeable)
            if agreeable else None
        )

        # 2. Output Stability (repeatability experiment)
        stable = [e for e in evaluations if e.deviation_classification == DeviationClassification.STABLE]
        has_baseline = [e for e in evaluations if e.baseline_run_id]
        stability_rate = len(stable) / len(has_baseline) if has_baseline else None

        # 3. Material Decision Variation Rate
        material_changes = [
            e for e in evaluations
            if e.decision_changed and e.materiality in (Materiality.HIGH, Materiality.CRITICAL)
        ]
        mdv_rate = len(material_changes) / total

        # 4. Decision Reversal Rate
        reversals = [
            e for e in evaluations
            if e.transition in ("APPROVE->DECLINE", "DECLINE->APPROVE")
        ]
        reversal_rate = len(reversals) / total

        # 5. PD Interpretation Accuracy
        pd_checks = [e for e in evaluations if e.pd_correctly_interpreted is not None]
        pd_accuracy = (
            sum(1 for e in pd_checks if e.pd_correctly_interpreted) / len(pd_checks)
            if pd_checks else None
        )

        # 6. Policy Consistency Rate
        policy_consistent = [
            e for e in evaluations
            if e.agrees_with_reference is True and not e.obsolete_policy_used
        ]
        policy_consistency = len(policy_consistent) / total

        # 7. Evidence Accuracy (unsupported conclusion rate inverse)
        unsupported = [e for e in evaluations if e.unsupported_conclusion]
        unsupported_rate = len(unsupported) / total

        # 8. Policy Conflict Detection Rate
        conflict_scenarios = [
            e for e in evaluations
            if e.policy_conflict_detected is not None
        ]
        conflict_detection_rate = (
            sum(1 for e in conflict_scenarios if e.policy_conflict_detected) / len(conflict_scenarios)
            if conflict_scenarios else None
        )

        # 9. Human Review Accuracy
        failed_escalation = [e for e in evaluations if e.failed_to_escalate]
        failed_escalation_rate = len(failed_escalation) / total

        # Severity distribution
        severity_dist = {s.value: 0 for s in Severity}
        for e in evaluations:
            if e.severity:
                severity_dist[e.severity.value] += 1

        # Deviation classification distribution
        dev_dist = {d.value: 0 for d in DeviationClassification}
        for e in evaluations:
            dev_dist[e.deviation_classification.value] += 1

        # Transition matrix
        transitions = {}
        for baseline_d in ("APPROVE", "MANUAL_REVIEW", "DECLINE"):
            transitions[baseline_d] = {v: 0 for v in ("APPROVE", "MANUAL_REVIEW", "DECLINE")}

        for e in evaluations:
            if e.transition:
                parts = e.transition.split("->")
                if len(parts) == 2 and parts[0] in transitions and parts[1] in transitions[parts[0]]:
                    transitions[parts[0]][parts[1]] += 1

        return {
            "experiment_type": experiment_type,
            "total_runs": total,
            "reference_decision_agreement_rate": rda_rate,
            "output_stability_rate": stability_rate,
            "material_decision_variation_rate": mdv_rate,
            "decision_reversal_rate": reversal_rate,
            "pd_interpretation_accuracy_rate": pd_accuracy,
            "policy_consistency_rate": policy_consistency,
            "unsupported_conclusion_rate": unsupported_rate,
            "policy_conflict_detection_rate": conflict_detection_rate,
            "failed_escalation_rate": failed_escalation_rate,
            "severity_distribution": severity_dist,
            "deviation_classification_distribution": dev_dist,
            "transition_matrix": transitions,
            "critical_deviations": len([
                e for e in evaluations
                if e.deviation_classification == DeviationClassification.CRITICAL_POLICY_DEVIATION
            ]),
            "unexpected_material_deviations": len([
                e for e in evaluations
                if e.deviation_classification == DeviationClassification.UNEXPECTED_MATERIAL
            ]),
        }
