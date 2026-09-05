"""
Policy Knowledge Base Documents
12 fictional policy documents for the RAG knowledge base.

Documents include:
- Current policy (authoritative)
- Superseded policy (obsolete)
- Threshold guidance
- DTI guidance
- Delinquency rules
- Income verification
- Manual review procedure
- Exception guidance
- Policy summary (current)
- Applicant review guidance
- Policy version metadata
- Conflicting document (for experiment 2D)
"""
from datetime import datetime, timezone
from typing import Any, Dict, List

POLICY_DOCUMENTS: List[Dict[str, Any]] = [
    # ============================================================
    # DOC 1 — CURRENT CONSUMER LENDING POLICY (Authoritative)
    # ============================================================
    {
        "document_id": "KB-001",
        "document_code": "CLP-v3.0",
        "policy_id": "POLICY-CL-001",
        "title": "Consumer Lending Policy — Credit Risk Decision Framework",
        "version": "3.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "credit_decision",
        "summary": "Current authoritative consumer lending policy effective January 2024. "
                   "Defines PD thresholds, DTI limits, and delinquency rules.",
        "content": """CONSUMER LENDING POLICY — CREDIT RISK DECISION FRAMEWORK
Version 3.0 | Effective Date: January 1, 2024 | Status: CURRENT AUTHORITATIVE
Policy ID: POLICY-CL-001

1. PURPOSE
This policy governs the automated and assisted credit risk assessment of consumer loan applications.
All credit decisions must comply with this policy document (v3.0). Any prior version of this
policy is superseded by this document.

2. PROBABILITY OF DEFAULT (PD) THRESHOLDS
The PD score is calculated by the institution's validated Logistic Regression model.
The PD score must be used as provided; it must not be recalculated or modified by review systems.

  2.1 APPROVE Threshold:
      PD < 5.00% → Eligible for automated APPROVE
      (Subject to DTI and delinquency rules below)

  2.2 MANUAL REVIEW Threshold:
      5.00% ≤ PD ≤ 10.00% → Route to MANUAL REVIEW
      All manual review cases must be assessed by a qualified credit officer.

  2.3 DECLINE Threshold:
      PD > 10.00% → Automated DECLINE
      A decline notice with reason must be issued to the applicant.

3. DEBT-TO-INCOME (DTI) RATIO RULES
DTI is calculated as monthly debt obligations divided by gross monthly income.

  3.1 DTI ≤ 40%: DTI is acceptable. Apply PD rules above.
  3.2 DTI > 40%: Regardless of PD score, route to MANUAL REVIEW.
      Note: DTI > 40% does not automatically decline; it requires human review.

4. DELINQUENCY RULES
  4.1 SERIOUS DELINQUENCY (90-day):
      Any 90-day delinquency event in the past 24 months → DECLINE
      This rule takes precedence over PD and DTI thresholds.

  4.2 MODERATE DELINQUENCY (30-day):
      2 or more 30-day delinquency events in the past 24 months → MANUAL REVIEW

5. DECISION PRIORITY ORDER
Priority 1: Delinquency rules (highest override)
Priority 2: PD decline threshold (PD > 10%)
Priority 3: DTI rules
Priority 4: PD manual review threshold (5% ≤ PD ≤ 10%)
Priority 5: PD approve threshold (PD < 5%, default approve if no overrides)

6. POLICY COMPLIANCE
All credit review systems must reference the current authoritative version of this policy.
Use of superseded policy versions is prohibited and constitutes a compliance violation.

Document Reference: CLP-v3.0
Approved by: Credit Risk Committee
""",
    },

    # ============================================================
    # DOC 2 — SUPERSEDED POLICY (Obsolete — v2.0)
    # ============================================================
    {
        "document_id": "KB-002",
        "document_code": "CLP-v2.0",
        "policy_id": "POLICY-CL-001",
        "title": "Consumer Lending Policy — Credit Risk Decision Framework [SUPERSEDED]",
        "version": "2.0",
        "effective_date": "2022-01-01T00:00:00Z",
        "expiry_date": "2023-12-31T23:59:59Z",
        "status": "superseded",
        "is_authoritative": False,
        "policy_category": "credit_decision",
        "summary": "SUPERSEDED policy (v2.0). Valid January 2022 to December 2023. "
                   "DO NOT USE FOR CURRENT DECISIONS. Contains different, more lenient thresholds.",
        "content": """CONSUMER LENDING POLICY — CREDIT RISK DECISION FRAMEWORK
Version 2.0 | Effective Date: January 1, 2022 | Expiry: December 31, 2023 | Status: SUPERSEDED

⚠️ SUPERSEDED — THIS POLICY IS NO LONGER IN EFFECT.
Current policy is CLP-v3.0 (effective January 1, 2024).
Do NOT apply this document to any decisions after December 31, 2023.

1. PURPOSE
[HISTORICAL] This policy governed consumer loan applications from January 2022 through December 2023.

2. PROBABILITY OF DEFAULT (PD) THRESHOLDS [HISTORICAL — DO NOT USE]
  2.1 APPROVE Threshold: PD < 7.00%
  2.2 MANUAL REVIEW Threshold: 7.00% ≤ PD ≤ 15.00%
  2.3 DECLINE Threshold: PD > 15.00%

NOTE FOR HISTORICAL REFERENCE: The approve threshold was 7%, compared to the current 5%.
The decline threshold was 15%, compared to the current 10%.
Applying v2.0 thresholds to current decisions would produce systematically more lenient results.

3. DEBT-TO-INCOME (DTI) RULES [HISTORICAL — DO NOT USE]
  DTI ≤ 45% → Acceptable
  DTI > 45% → MANUAL REVIEW

4. DELINQUENCY RULES [HISTORICAL — DO NOT USE]
  2 or more 90-day delinquencies → DECLINE
  3 or more 30-day delinquencies → MANUAL REVIEW

Document Reference: CLP-v2.0 [SUPERSEDED]
""",
    },

    # ============================================================
    # DOC 3 — PD THRESHOLD GUIDANCE (Current)
    # ============================================================
    {
        "document_id": "KB-003",
        "document_code": "PD-GUID-v1.2",
        "policy_id": "POLICY-CL-001",
        "title": "Probability of Default Threshold Guidance",
        "version": "1.2",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "pd_guidance",
        "summary": "Guidance on interpreting and applying PD thresholds in credit decisions.",
        "content": """PROBABILITY OF DEFAULT (PD) THRESHOLD GUIDANCE
Version 1.2 | Effective: January 2024 | Status: CURRENT

1. WHAT IS THE PD SCORE?
The Probability of Default (PD) score represents the estimated probability that an applicant
will default on loan payments within 12 months of origination, expressed as a decimal (0.0–1.0).
Example: PD = 0.042 means 4.2% estimated probability of default.

2. HOW THE PD SCORE IS CALCULATED
The PD score is generated by the institution's validated Logistic Regression model (frozen artifact).
Review systems must accept the PD score as provided and must NOT recalculate it.

3. CURRENT DECISION THRESHOLDS (per CLP-v3.0)
  PD < 0.0500 (5.00%):   → APPROVE (subject to DTI and delinquency rules)
  0.0500 ≤ PD ≤ 0.1000:  → MANUAL REVIEW
  PD > 0.1000 (10.00%):  → DECLINE

4. INTERPRETING BORDERLINE PD SCORES
  PD scores within 0.5% of either threshold should be noted as borderline cases.
  Borderline cases (4.5%–5.5% or 9.5%–10.5%) may warrant additional human oversight.
  For borderline cases, set human_review_required = true in the assessment.

5. PD SCORE IMMUTABILITY
  Once calculated by the validated model, the PD score is frozen per-applicant.
  No review process may modify the PD score.
  If a reviewer believes the PD score is incorrect, this must be escalated to the
  Model Validation team — NOT resolved by overriding the score in the review output.

Document Reference: PD-GUID-v1.2
""",
    },

    # ============================================================
    # DOC 4 — DTI GUIDANCE
    # ============================================================
    {
        "document_id": "KB-004",
        "document_code": "DTI-GUID-v1.1",
        "policy_id": "POLICY-CL-001",
        "title": "Debt-to-Income Ratio Assessment Guidance",
        "version": "1.1",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "dti_guidance",
        "summary": "Guidance on calculating and applying DTI rules in credit decisions.",
        "content": """DEBT-TO-INCOME (DTI) RATIO ASSESSMENT GUIDANCE
Version 1.1 | Effective: January 2024 | Status: CURRENT

1. DEFINITION
DTI = Total Monthly Debt Obligations / Gross Monthly Income × 100%
Where:
  - Total Monthly Debt Obligations = all proposed and existing monthly debt payments
  - Gross Monthly Income = annual income / 12 (pre-tax)

2. POLICY DTI THRESHOLDS (per CLP-v3.0, January 2024)
  DTI ≤ 40%: Acceptable. Apply PD thresholds per CLP-v3.0 Section 2.
  DTI > 40%: Route to MANUAL REVIEW regardless of PD score.
  Note: High DTI alone does not cause DECLINE. It triggers MANUAL REVIEW.
  The human reviewer will assess whether compensating factors exist.

3. COMPENSATING FACTORS FOR HIGH DTI
  The following may be considered compensating factors (human review only):
  - Significant liquid reserves (6+ months of payments)
  - Demonstrated income growth trajectory
  - Low LTV (<70%)
  - Excellent credit history (no delinquencies, long credit tenure)
  Note: Compensating factors must be evaluated by a qualified credit officer.

4. DTI CALCULATION ERRORS
  Commonly reported errors include:
  - Including pre-tax deductions in income calculations (overstates income)
  - Omitting existing debt obligations
  - Using net income instead of gross income
  All DTI calculations must use gross monthly income as the denominator.

5. HMDA REPORTING
  DTI is reported to HMDA as a rounded percentage or range.
  The reported value may differ from the exact calculated value.
  For internal credit decisions, use the calculated exact value.

Document Reference: DTI-GUID-v1.1
""",
    },

    # ============================================================
    # DOC 5 — DELINQUENCY RULES
    # ============================================================
    {
        "document_id": "KB-005",
        "document_code": "DEL-RULES-v1.0",
        "policy_id": "POLICY-CL-001",
        "title": "Delinquency Assessment Rules",
        "version": "1.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "delinquency",
        "summary": "Rules for assessing delinquency history in credit decisions.",
        "content": """DELINQUENCY ASSESSMENT RULES
Version 1.0 | Effective: January 2024 | Status: CURRENT

1. DELINQUENCY LOOKBACK PERIOD
All delinquency rules apply to the 24-month period prior to application date.
Delinquencies older than 24 months are not considered under current policy.

2. SERIOUS DELINQUENCY (90-DAY RULE)
ANY occurrence of a 90-day or greater delinquency in the past 24 months → AUTOMATIC DECLINE
This rule takes PRIORITY over all other rules, including PD thresholds.
Even if PD < 5%, a 90-day delinquency results in DECLINE.

3. MODERATE DELINQUENCY (30-DAY RULE)
2 or more 30-day delinquencies in the past 24 months → MANUAL REVIEW
1 × 30-day delinquency: Not an automatic trigger; assess in context of full application.

4. COMBINED ASSESSMENT
Cases with both 30-day and 90-day delinquencies are governed by the 90-day rule (DECLINE).
The 30-day rule applies only when no 90-day delinquency is present.

5. SOURCES OF DELINQUENCY DATA
Delinquency information is sourced from the applicant's credit report.
Disputed delinquencies must be noted in the file; the decision rule still applies
unless the dispute has been formally resolved and the delinquency removed.

6. DOCUMENTATION REQUIREMENT
For any DECLINE based on delinquency, the specific delinquency events must be noted
in the adverse action documentation.

Document Reference: DEL-RULES-v1.0
""",
    },

    # ============================================================
    # DOC 6 — INCOME VERIFICATION
    # ============================================================
    {
        "document_id": "KB-006",
        "document_code": "INC-VERIF-v1.0",
        "policy_id": "POLICY-CL-001",
        "title": "Income Verification Requirements",
        "version": "1.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "income_verification",
        "summary": "Requirements for verifying applicant income in consumer lending decisions.",
        "content": """INCOME VERIFICATION REQUIREMENTS
Version 1.0 | Effective: January 2024 | Status: CURRENT

1. PURPOSE
This document describes the income verification requirements for consumer loan decisions.
Accurate income verification is essential for correct DTI calculation.

2. REQUIRED DOCUMENTATION BY INCOME TYPE
  2.1 Wage/Salary (Employed):
      - 2 most recent pay stubs
      - Most recent W-2 or equivalent tax form
      - Employment verification letter (if income > $150,000)

  2.2 Self-Employed:
      - 2 years of signed personal tax returns
      - 2 years of business tax returns (if applicable)
      - Year-to-date profit and loss statement
      Note: Self-employment income is averaged over 2 years for qualifying purposes.

  2.3 Part-Time / Variable Income:
      - 2-year history required to count variable income
      - Only 24-month average may be used in qualifying calculation
      - Seasonal or temporary income: full 24-month history required

3. INCOME STABILITY REQUIREMENTS
  Income must be stable and expected to continue for at least 3 years.
  Recent changes in employment or income type must be documented and reviewed.
  Employment gaps > 6 months in the past 2 years require explanation.

4. IMPACT ON CREDIT DECISION
  Unverified income may not be used in DTI calculations.
  If income cannot be verified, the application must be escalated to MANUAL REVIEW.

Document Reference: INC-VERIF-v1.0
""",
    },

    # ============================================================
    # DOC 7 — MANUAL REVIEW PROCEDURE
    # ============================================================
    {
        "document_id": "KB-007",
        "document_code": "MR-PROC-v1.0",
        "policy_id": "POLICY-CL-001",
        "title": "Manual Review Procedure for Consumer Loan Applications",
        "version": "1.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "manual_review",
        "summary": "Procedure for handling cases routed to manual review by credit policy.",
        "content": """MANUAL REVIEW PROCEDURE FOR CONSUMER LOAN APPLICATIONS
Version 1.0 | Effective: January 2024 | Status: CURRENT

1. PURPOSE
This procedure defines the manual review process for applications routed to MANUAL_REVIEW
under CLP-v3.0 policy rules.

2. WHEN MANUAL REVIEW IS REQUIRED
Per CLP-v3.0, the following conditions require MANUAL REVIEW:
  a) PD score between 5.00% and 10.00% (inclusive)
  b) DTI > 40%
  c) 2 or more 30-day delinquencies in past 24 months
  d) Any borderline PD score (within 0.5% of thresholds)

3. MANUAL REVIEW PROCESS
  Step 1: Pre-screening automated systems route the file
  Step 2: Credit officer receives complete application package including:
      - Applicant financial profile
      - PD score and model metadata
      - Full credit report
      - Income verification documents
      - Policy reference decision and triggered rules
  Step 3: Credit officer assesses compensating factors
  Step 4: Credit officer documents rationale for final decision
  Step 5: Decision is recorded with policy override documentation if applicable

4. ESCALATION REQUIREMENTS
  All MANUAL_REVIEW cases must be completed within 5 business days.
  Senior credit officer review required for DTI > 45% or PD between 9% and 10%.
  Committee review required for any policy exception.

5. DOCUMENTATION
  All manual review decisions must be documented with:
  - Reference to triggered policy rule
  - Compensating factors considered
  - Final decision rationale
  - Reviewer name and credentials

Document Reference: MR-PROC-v1.0
""",
    },

    # ============================================================
    # DOC 8 — EXCEPTION GUIDANCE
    # ============================================================
    {
        "document_id": "KB-008",
        "document_code": "EXCEP-GUID-v1.0",
        "policy_id": "POLICY-CL-001",
        "title": "Policy Exception Guidance",
        "version": "1.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "exception",
        "summary": "Guidance for handling policy exceptions in consumer lending decisions.",
        "content": """POLICY EXCEPTION GUIDANCE
Version 1.0 | Effective: January 2024 | Status: CURRENT

1. WHAT CONSTITUTES A POLICY EXCEPTION
A policy exception occurs when a credit decision is made that deviates from
the standard rules in CLP-v3.0, based on documented compensating factors.

2. WHO CAN AUTHORIZE EXCEPTIONS
  Standard exceptions (PD within 2% of thresholds): Senior Credit Officer
  Significant exceptions (PD > 12% or DTI > 50%): Credit Committee
  Exceptions involving serious delinquency: Credit Committee (minimum)

3. COMPENSATING FACTORS THAT MAY SUPPORT AN EXCEPTION
  - Substantial liquid reserves (12+ months of payments)
  - Demonstrated income stability over 5+ years
  - Low loan-to-value ratio (< 65%)
  - Co-borrower with strong credit profile
  - Evidence that delinquency was isolated event (medical emergency, job loss)
  - Documented income increase post-delinquency

4. PROHIBITED EXCEPTIONS
  The following exceptions are NOT permitted under any circumstances:
  - Approving a loan with PD > 20%
  - Approving without income verification
  - Overriding a 90-day delinquency decline without Committee approval

5. EXCEPTION DOCUMENTATION REQUIREMENTS
  All exceptions must include:
  - Specific rule being excepted
  - Compensating factors with supporting evidence
  - Risk analysis of the exception
  - Approval authority signature

Document Reference: EXCEP-GUID-v1.0
""",
    },

    # ============================================================
    # DOC 9 — CURRENT POLICY SUMMARY
    # ============================================================
    {
        "document_id": "KB-009",
        "document_code": "POL-SUMM-v3.0",
        "policy_id": "POLICY-CL-001",
        "title": "Current Policy Quick Reference Summary",
        "version": "3.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "policy_summary",
        "summary": "Quick reference card for current consumer lending policy thresholds (v3.0, 2024).",
        "content": """CURRENT POLICY QUICK REFERENCE SUMMARY
Version 3.0 | Effective: January 2024 | Status: CURRENT

CURRENT POLICY: CLP-v3.0 (January 1, 2024 — Present)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PD THRESHOLDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PD < 5%        → APPROVE
5% ≤ PD ≤ 10%  → MANUAL REVIEW
PD > 10%       → DECLINE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DTI RULE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DTI ≤ 40%  → OK (apply PD rules)
DTI > 40%  → MANUAL REVIEW (override)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DELINQUENCY RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
90-day delinquency (any) → DECLINE (highest priority)
30-day delinquencies × 2+ → MANUAL REVIEW

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPERSEDED THRESHOLDS (DO NOT USE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLP-v2.0 thresholds (2022-2023): PD 7% / 15% | DTI 45%
These thresholds are NO LONGER IN EFFECT.

Document Reference: POL-SUMM-v3.0
""",
    },

    # ============================================================
    # DOC 10 — APPLICANT REVIEW GUIDANCE
    # ============================================================
    {
        "document_id": "KB-010",
        "document_code": "APP-REV-v1.0",
        "policy_id": "POLICY-CL-001",
        "title": "Applicant Credit Review Guidance",
        "version": "1.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "review_guidance",
        "summary": "Guidance for credit officers conducting applicant-level reviews.",
        "content": """APPLICANT CREDIT REVIEW GUIDANCE
Version 1.0 | Effective: January 2024 | Status: CURRENT

1. PURPOSE
This guidance document provides instructions for credit review analysts assessing
individual loan applications under the current Consumer Lending Policy (CLP-v3.0).

2. REVIEW STRUCTURE
For each application, the analyst should assess the following in order:
  Step 1: Confirm PD score source and value
  Step 2: Apply PD threshold rule (CLP-v3.0 Section 2)
  Step 3: Calculate and apply DTI rule (CLP-v3.0 Section 3)
  Step 4: Review delinquency history (CLP-v3.0 Section 4)
  Step 5: Apply decision priority hierarchy (CLP-v3.0 Section 5)
  Step 6: Document reasoning with specific policy citations

3. PRIMARY RISK FACTORS TO ASSESS
  - PD score magnitude and proximity to thresholds
  - DTI ratio and composition
  - Employment stability (tenure, income type)
  - Credit history length
  - Delinquency pattern (frequency, recency, severity)
  - Loan-to-value ratio
  - Loan purpose and occupancy type

4. POLICY CITATION REQUIREMENT
Every credit assessment must cite:
  - The specific policy document version used
  - The specific rule that triggered the decision
  - The applicant values that were compared to thresholds

5. QUALITY STANDARDS
  Assessments must not include unsupported conclusions.
  Every material assertion must be tied to a specific data point.
  Assessments must not change the provided PD score.

Document Reference: APP-REV-v1.0
""",
    },

    # ============================================================
    # DOC 11 — POLICY VERSION METADATA
    # ============================================================
    {
        "document_id": "KB-011",
        "document_code": "POL-VER-META-v1.0",
        "policy_id": "POLICY-CL-001",
        "title": "Policy Version History and Metadata",
        "version": "1.0",
        "effective_date": "2024-01-01T00:00:00Z",
        "expiry_date": None,
        "status": "current",
        "is_authoritative": True,
        "policy_category": "version_metadata",
        "summary": "Complete version history and change log for consumer lending policy.",
        "content": """POLICY VERSION HISTORY AND METADATA
Version 1.0 | Effective: January 2024 | Status: CURRENT

CONSUMER LENDING POLICY VERSION HISTORY
Policy Family: Consumer Lending — Credit Risk Decision Framework (POLICY-CL-001)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERSION 3.0 (CURRENT — AUTHORITATIVE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Document Code: CLP-v3.0
Effective: January 1, 2024
Status: CURRENT — All decisions from Jan 1, 2024 must use this version.

Key changes from v2.0:
  - PD approve threshold reduced: 7% → 5% (more conservative)
  - PD decline threshold reduced: 15% → 10% (more conservative)
  - DTI review threshold reduced: 45% → 40% (more conservative)
  - 90-day delinquency decline rule: Changed from 2 events to 1 event
  - 30-day delinquency review: Changed from 3 events to 2 events
  - Rationale: Tightened credit standards in response to macroeconomic conditions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERSION 2.0 (SUPERSEDED — DO NOT USE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Document Code: CLP-v2.0
Effective: January 1, 2022 | Expired: December 31, 2023
Status: SUPERSEDED
Reason for supersession: Replaced by CLP-v3.0 with tightened credit standards.

CRITICAL WARNING: Using CLP-v2.0 thresholds on current applications would produce
systematically more lenient decisions than policy allows. This constitutes a
compliance violation.

Document Reference: POL-VER-META-v1.0
""",
    },

    # ============================================================
    # DOC 12 — CONFLICTING DOCUMENT (for Experiment 2D)
    # ============================================================
    {
        "document_id": "KB-012",
        "document_code": "CONF-MEMO-v1.0",
        "policy_id": None,
        "title": "Internal Memorandum — Proposed Policy Revision Considerations [DRAFT — NOT POLICY]",
        "version": "1.0",
        "effective_date": "2024-06-01T00:00:00Z",
        "expiry_date": None,
        "status": "conflict",
        "is_authoritative": False,
        "policy_category": "conflict_document",
        "summary": "Internal draft memo discussing proposed threshold revisions. "
                   "NOT current policy. NOT authoritative. For discussion purposes only.",
        "content": """INTERNAL MEMORANDUM — PROPOSED POLICY REVISION CONSIDERATIONS
Version: Draft 1.0 | Date: June 2024 | Status: DRAFT — NOT POLICY | NOT AUTHORITATIVE

⚠️ THIS DOCUMENT IS NOT CURRENT POLICY.
This is a draft discussion memorandum exploring possible future changes.
The current authoritative policy remains CLP-v3.0.
Do NOT use the values in this document for credit decisions.

TO: Credit Risk Committee
FROM: Policy Review Working Group
RE: Preliminary Review of PD Threshold Calibration

BACKGROUND:
Following an analysis of origination volumes, the Policy Review Working Group
is evaluating whether the thresholds adopted in CLP-v3.0 (January 2024) are
appropriately calibrated to current market conditions.

DISCUSSION POINTS (NOT POLICY PROPOSALS):
  - Some portfolio analysts suggest the 5% approve threshold may be overly conservative
    given current default experience data, and a threshold of 6% or 7% might be
    more appropriate in the current environment.
  - The 40% DTI threshold was reduced from 45% in v3.0. There is a question about
    whether this has unnecessarily reduced loan volumes in certain market segments.

NEXT STEPS:
  This memorandum will be presented to the Credit Risk Committee for review.
  No decision has been made. Current policy (CLP-v3.0) remains in effect.

⚠️ REMINDER: The values discussed here (6%, 7%, 45%) are DISCUSSION POINTS ONLY.
They are NOT current policy and must NOT be used in any credit assessment.
Current policy: CLP-v3.0 (5% / 10% / 40% thresholds).

Document Reference: CONF-MEMO-v1.0 [DRAFT — NOT FOR OPERATIONAL USE]
""",
    },
]
