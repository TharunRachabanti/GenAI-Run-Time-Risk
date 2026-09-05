"""
Unit Tests — Evaluation Engine
Tests the deviation classification, materiality, and severity calculations.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.evaluation.evaluation_engine import (
    EvaluationEngine,
    DeviationClassification,
    Materiality,
    Severity,
)


@pytest.fixture
def engine():
    return EvaluationEngine()


def make_output(recommendation="APPROVE", risk_class="LOW_RISK", conflict=False, unsupported=False, review=False):
    return {
        "recommendation": recommendation,
        "risk_classification": risk_class,
        "primary_risk_factors": ["factor1"],
        "policy_references": ["CLP-v3.0"],
        "policy_conflict_flag": conflict,
        "unsupported_conclusion_flag": unsupported,
        "human_review_required": review,
        "reasoning_summary": "Test reasoning",
    }


def make_baseline(recommendation="APPROVE", risk_class="LOW_RISK"):
    return {
        "run_id": "baseline-001",
        "recommendation": recommendation,
        "risk_classification": risk_class,
        "primary_risk_factors": ["factor1"],
        "policy_references": ["CLP-v3.0"],
    }


class TestStableClassification:
    def test_stable_when_identical(self, engine):
        result = engine.evaluate(
            run_id="r1",
            experiment_type="prompt_variation",
            genai_output=make_output("APPROVE"),
            reference_decision="APPROVE",
            baseline_output=make_baseline("APPROVE"),
            policy_scenario="current_only",
            varied_variable="prompt_version",
        )
        assert result.deviation_classification == DeviationClassification.STABLE
        assert result.severity == Severity.LOW

    def test_stable_with_reference_agreement(self, engine):
        result = engine.evaluate(
            run_id="r2",
            experiment_type="baseline",
            genai_output=make_output("MANUAL_REVIEW"),
            reference_decision="MANUAL_REVIEW",
            baseline_output=make_baseline("MANUAL_REVIEW"),
            policy_scenario="current_only",
            varied_variable="none",
        )
        assert result.deviation_classification == DeviationClassification.STABLE
        assert result.agrees_with_reference is True


class TestCriticalDeviations:
    def test_approve_to_decline_is_critical(self, engine):
        result = engine.evaluate(
            run_id="r3",
            experiment_type="prompt_variation",
            genai_output=make_output("DECLINE"),
            reference_decision="APPROVE",
            baseline_output=make_baseline("APPROVE"),
            policy_scenario="current_only",
            varied_variable="prompt_version",
        )
        assert result.severity == Severity.CRITICAL
        assert result.transition == "APPROVE->DECLINE"
        assert result.materiality == Materiality.CRITICAL

    def test_decline_to_approve_is_critical(self, engine):
        result = engine.evaluate(
            run_id="r4",
            experiment_type="model_change",
            genai_output=make_output("APPROVE"),
            reference_decision="DECLINE",
            baseline_output=make_baseline("DECLINE"),
            policy_scenario="current_only",
            varied_variable="genai_model",
        )
        assert result.severity == Severity.CRITICAL
        assert result.transition == "DECLINE->APPROVE"


class TestExpectedPolicyCorrect:
    def test_expected_when_change_aligns_with_reference(self, engine):
        # Baseline was APPROVE (wrong), variant is MANUAL_REVIEW which matches reference
        result = engine.evaluate(
            run_id="r5",
            experiment_type="policy_rag",
            genai_output=make_output("MANUAL_REVIEW"),
            reference_decision="MANUAL_REVIEW",
            baseline_output=make_baseline("APPROVE"),
            policy_scenario="current_only",
            varied_variable="policy_scenario",
        )
        assert result.agrees_with_reference is True
        assert result.deviation_classification == DeviationClassification.EXPECTED_POLICY_CORRECT


class TestFailedEscalation:
    def test_failed_escalation_detected(self, engine):
        result = engine.evaluate(
            run_id="r6",
            experiment_type="prompt_variation",
            genai_output=make_output("APPROVE", review=False),  # Should have been MANUAL_REVIEW
            reference_decision="MANUAL_REVIEW",
            baseline_output=make_baseline("MANUAL_REVIEW"),
            policy_scenario="current_only",
            varied_variable="prompt_version",
        )
        assert result.failed_to_escalate is True
        assert result.severity == Severity.CRITICAL

    def test_no_failed_escalation_when_human_review_set(self, engine):
        result = engine.evaluate(
            run_id="r7",
            experiment_type="prompt_variation",
            genai_output=make_output("APPROVE", review=True),  # APPROVE but human_review set
            reference_decision="MANUAL_REVIEW",
            baseline_output=make_baseline("MANUAL_REVIEW"),
            policy_scenario="current_only",
            varied_variable="prompt_version",
        )
        assert result.failed_to_escalate is False


class TestMaterialityLevels:
    def test_low_materiality_when_no_change(self, engine):
        result = engine.evaluate(
            run_id="r8",
            experiment_type="prompt_variation",
            genai_output=make_output("APPROVE"),
            reference_decision="APPROVE",
            baseline_output=make_baseline("APPROVE"),
            policy_scenario="current_only",
            varied_variable="prompt_version",
        )
        assert result.materiality == Materiality.LOW

    def test_high_materiality_when_decision_changes(self, engine):
        result = engine.evaluate(
            run_id="r9",
            experiment_type="prompt_variation",
            genai_output=make_output("MANUAL_REVIEW"),
            reference_decision="APPROVE",
            baseline_output=make_baseline("APPROVE"),
            policy_scenario="current_only",
            varied_variable="prompt_version",
        )
        assert result.materiality == Materiality.HIGH


class TestNoBaselineNA:
    def test_na_classification_without_baseline(self, engine):
        result = engine.evaluate(
            run_id="r10",
            experiment_type="baseline",
            genai_output=make_output("APPROVE"),
            reference_decision="APPROVE",
            baseline_output=None,  # No baseline
            policy_scenario="current_only",
            varied_variable="none",
        )
        assert result.deviation_classification == DeviationClassification.NA
        assert result.decision_changed is None


class TestMetricsCalculator:
    def test_aggregate_metrics_with_empty_list(self):
        from app.evaluation.evaluation_engine import MetricsCalculator, EvaluationOutput
        calc = MetricsCalculator()
        result = calc.calculate_aggregate([], "test")
        assert "error" in result

    def test_transition_matrix_construction(self, engine):
        outputs = [
            engine.evaluate(
                run_id=f"r{i}",
                experiment_type="prompt_variation",
                genai_output=make_output(d1),
                reference_decision=d2,
                baseline_output=make_baseline(d2),
                policy_scenario="current_only",
                varied_variable="prompt_version",
            )
            for i, (d1, d2) in enumerate([
                ("APPROVE", "APPROVE"),
                ("MANUAL_REVIEW", "APPROVE"),
                ("DECLINE", "APPROVE"),
            ])
        ]
        from app.evaluation.evaluation_engine import MetricsCalculator
        calc = MetricsCalculator()
        metrics = calc.calculate_aggregate(outputs, "prompt_variation")
        assert metrics["total_runs"] == 3
        assert "transition_matrix" in metrics
