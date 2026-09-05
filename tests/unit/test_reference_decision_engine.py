"""
Unit Tests — Policy Reference Decision Engine
Tests that the deterministic policy engine produces correct decisions
for all rule combinations.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.policies.reference_decision_engine import (
    Decision,
    PolicyRules,
    ReferenceDecisionEngine,
    CURRENT_POLICY_RULES,
    SUPERSEDED_POLICY_RULES,
)


@pytest.fixture
def engine():
    """Default engine with current policy rules."""
    return ReferenceDecisionEngine(CURRENT_POLICY_RULES)


@pytest.fixture
def superseded_engine():
    """Engine with superseded policy rules."""
    return ReferenceDecisionEngine(SUPERSEDED_POLICY_RULES)


class TestCurrentPolicyApprove:
    def test_clear_approve(self, engine):
        result = engine.calculate("A001", pd_score=0.030, dti_ratio=20.0)
        assert result.reference_decision == Decision.APPROVE
        assert result.triggered_rule == "PD_APPROVE"

    def test_boundary_just_below_approve_threshold(self, engine):
        result = engine.calculate("A002", pd_score=0.0499, dti_ratio=25.0)
        assert result.reference_decision == Decision.APPROVE

    def test_zero_pd_approve(self, engine):
        result = engine.calculate("A003", pd_score=0.001, dti_ratio=10.0)
        assert result.reference_decision == Decision.APPROVE


class TestCurrentPolicyManualReview:
    def test_pd_in_review_zone(self, engine):
        result = engine.calculate("B001", pd_score=0.07, dti_ratio=30.0)
        assert result.reference_decision == Decision.MANUAL_REVIEW
        assert result.triggered_rule == "PD_MANUAL_REVIEW"

    def test_boundary_just_above_approve_threshold(self, engine):
        result = engine.calculate("B002", pd_score=0.0501, dti_ratio=30.0)
        assert result.reference_decision == Decision.MANUAL_REVIEW

    def test_boundary_at_decline_threshold(self, engine):
        result = engine.calculate("B003", pd_score=0.1000, dti_ratio=30.0)
        assert result.reference_decision == Decision.MANUAL_REVIEW

    def test_dti_triggers_review_despite_low_pd(self, engine):
        result = engine.calculate("B004", pd_score=0.03, dti_ratio=42.0)
        assert result.reference_decision == Decision.MANUAL_REVIEW
        assert result.triggered_rule == "DTI_MANUAL_REVIEW"

    def test_two_thirty_day_delinquencies_triggers_review(self, engine):
        result = engine.calculate("B005", pd_score=0.03, dti_ratio=25.0, delinquency_30day_count=2)
        assert result.reference_decision == Decision.MANUAL_REVIEW
        assert result.triggered_rule == "DELINQUENCY_30_REVIEW"


class TestCurrentPolicyDecline:
    def test_pd_above_decline_threshold(self, engine):
        result = engine.calculate("C001", pd_score=0.12, dti_ratio=30.0)
        assert result.reference_decision == Decision.DECLINE
        assert result.triggered_rule == "PD_DECLINE"

    def test_boundary_just_above_decline_threshold(self, engine):
        result = engine.calculate("C002", pd_score=0.1001, dti_ratio=30.0)
        assert result.reference_decision == Decision.DECLINE

    def test_ninety_day_delinquency_overrides_low_pd(self, engine):
        result = engine.calculate("C003", pd_score=0.03, dti_ratio=25.0, delinquency_90day_count=1)
        assert result.reference_decision == Decision.DECLINE
        assert result.triggered_rule == "DELINQUENCY_90_DECLINE"

    def test_ninety_day_delinquency_overrides_review_zone(self, engine):
        result = engine.calculate("C004", pd_score=0.07, dti_ratio=35.0, delinquency_90day_count=1)
        assert result.reference_decision == Decision.DECLINE

    def test_extreme_high_pd_decline(self, engine):
        result = engine.calculate("C005", pd_score=0.45, dti_ratio=60.0)
        assert result.reference_decision == Decision.DECLINE


class TestRulePriority:
    def test_delinquency_overrides_all(self, engine):
        """90-day delinquency should override even a very low PD."""
        result = engine.calculate("D001", pd_score=0.01, dti_ratio=15.0, delinquency_90day_count=1)
        assert result.reference_decision == Decision.DECLINE
        assert result.triggered_rule == "DELINQUENCY_90_DECLINE"

    def test_pd_decline_overrides_dti_review(self, engine):
        """PD decline should take precedence over DTI review trigger."""
        result = engine.calculate("D002", pd_score=0.15, dti_ratio=42.0)
        assert result.reference_decision == Decision.DECLINE
        assert result.triggered_rule == "PD_DECLINE"

    def test_single_30day_delinquency_alone_not_trigger(self, engine):
        """Single 30-day delinquency should not trigger review by itself."""
        result = engine.calculate("D003", pd_score=0.03, dti_ratio=25.0, delinquency_30day_count=1)
        assert result.reference_decision == Decision.APPROVE


class TestSupersededPolicy:
    def test_superseded_has_different_thresholds(self, superseded_engine):
        """Old policy had PD 7%/15% thresholds (more lenient)."""
        # PD=6% would be APPROVE under v2.0 but MANUAL_REVIEW under v3.0
        result = superseded_engine.calculate("E001", pd_score=0.06, dti_ratio=30.0)
        assert result.reference_decision == Decision.APPROVE  # Under v2.0, 6% is below 7% approve threshold

    def test_current_policy_stricter_than_superseded(self):
        """Demonstrate that v3.0 is stricter than v2.0."""
        current_engine = ReferenceDecisionEngine(CURRENT_POLICY_RULES)
        superseded_eng = ReferenceDecisionEngine(SUPERSEDED_POLICY_RULES)
        
        # PD=6% applicant
        current_result = current_engine.calculate("E002", pd_score=0.06, dti_ratio=30.0)
        superseded_result = superseded_eng.calculate("E002", pd_score=0.06, dti_ratio=30.0)
        
        assert current_result.reference_decision == Decision.MANUAL_REVIEW
        assert superseded_result.reference_decision == Decision.APPROVE


class TestBoundaryCases:
    def test_exactly_at_approve_threshold(self, engine):
        """PD exactly at 5% — should be MANUAL_REVIEW (rule: PD < 5% → APPROVE)."""
        result = engine.calculate("F001", pd_score=0.05, dti_ratio=30.0)
        assert result.reference_decision == Decision.MANUAL_REVIEW

    def test_exactly_at_decline_threshold(self, engine):
        """PD exactly at 10% — should be MANUAL_REVIEW (rule: PD > 10% → DECLINE)."""
        result = engine.calculate("F002", pd_score=0.10, dti_ratio=30.0)
        assert result.reference_decision == Decision.MANUAL_REVIEW

    def test_dti_exactly_at_threshold(self, engine):
        """DTI exactly at 40% — should NOT trigger review."""
        result = engine.calculate("F003", pd_score=0.03, dti_ratio=40.0)
        assert result.reference_decision == Decision.APPROVE  # DTI == 40 is not > 40

    def test_dti_just_above_threshold(self, engine):
        """DTI at 40.1% — should trigger review."""
        result = engine.calculate("F004", pd_score=0.03, dti_ratio=40.1)
        assert result.reference_decision == Decision.MANUAL_REVIEW


class TestRuleAuditTrail:
    def test_result_contains_explanation(self, engine):
        result = engine.calculate("G001", pd_score=0.04, dti_ratio=25.0)
        assert len(result.decision_explanation) > 20
        assert result.policy_version == "3.0"
        assert result.policy_id == "POLICY-CL-001"

    def test_result_contains_rule_details(self, engine):
        result = engine.calculate("G002", pd_score=0.07, dti_ratio=30.0)
        assert len(result.rule_details) > 0
        triggered = [r for r in result.rule_details if r.triggered]
        assert len(triggered) == 1
        assert triggered[0].rule_id == "PD_MANUAL_REVIEW"

    def test_invalid_pd_raises(self, engine):
        with pytest.raises(ValueError):
            engine.calculate("G003", pd_score=1.5, dti_ratio=25.0)

    def test_negative_dti_raises(self, engine):
        with pytest.raises(ValueError):
            engine.calculate("G004", pd_score=0.04, dti_ratio=-5.0)
