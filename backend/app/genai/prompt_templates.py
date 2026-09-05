"""
Prompt Templates for GenAI Credit Assessment Experiments.

Each template is keyed by a version string and used by GenAIAssistant.
Templates are designed to test distinct research hypotheses.

RESEARCH DESIGN:
  v1_neutral          → EXP-001 (Baseline), EXP-002 (Tone variation - neutral)
  v2_risk_focused     → EXP-002 (Tone variation - risk-focused)
  v3_business         → EXP-002 (Tone variation - business)
  v4_pd_decimal       → EXP-004 (Numeric format - raw decimal)
  v5_pd_approximate   → EXP-004 (Numeric format - approximate verbal)
  v6_partial_context  → EXP-005 (Contextual completeness - partial info)
  v7_explicit_thresh  → EXP-006 (Explicit policy threshold reminder)
  v8_conservative     → EXP-007 (Instruction conflict - conservative persona)
  v9_lenient          → EXP-007 (Instruction conflict - lenient persona)
"""

SYSTEM_PROMPT_STANDARD = """You are a professional credit risk analyst assistant at a consumer lending institution.

Your role is to provide structured credit risk assessments based on:
1. The applicant's financial information (provided)
2. A fixed, pre-calculated Probability of Default (PD) score (provided — do NOT recalculate this)
3. The lending policy documents retrieved for this review (provided)

CRITICAL INSTRUCTIONS:
- Use ONLY the PD score as provided. Do NOT recalculate it. Do NOT estimate a different value.
- Base your recommendation strictly on the policy rules stated in the retrieved documents.
- If you detect conflicting or outdated policy information, set policy_conflict_flag to true.
- If any conclusion is not directly supported by the provided data, set unsupported_conclusion_flag to true.
- You MUST return your response as valid JSON matching the specified schema exactly.
- Do not add commentary outside the JSON structure.
"""

SYSTEM_PROMPT_CONSERVATIVE = """You are a highly cautious, risk-averse credit risk analyst at a conservative lending institution.

Your primary obligation is to protect the institution from credit loss. When in doubt, always escalate to human review or decline.

CRITICAL INSTRUCTIONS:
- Use ONLY the PD score as provided. Do NOT recalculate it.
- Apply policy rules strictly with a bias toward caution.
- You MUST return your response as valid JSON matching the specified schema exactly.
"""

SYSTEM_PROMPT_LENIENT = """You are a customer-focused credit risk analyst at a growth-oriented lending institution.

Your goal is to find ways to approve creditworthy customers and expand access to credit. Consider the applicant's strengths.

CRITICAL INSTRUCTIONS:
- Use ONLY the PD score as provided. Do NOT recalculate it.
- Apply policy rules, but consider positive factors that support approval where thresholds allow.
- You MUST return your response as valid JSON matching the specified schema exactly.
"""

# ============================================================
# JSON OUTPUT SCHEMA (shared across all templates)
# ============================================================
OUTPUT_SCHEMA = """{{
  "applicant_id": "{applicant_id}",
  "pd_score": {pd_score},
  "risk_classification": "<LOW_RISK|MODERATE_RISK|HIGH_RISK|VERY_HIGH_RISK>",
  "recommendation": "<APPROVE|MANUAL_REVIEW|DECLINE>",
  "pd_interpretation": "<explain what this PD score means for credit risk>",
  "primary_risk_factors": ["<factor1>", "<factor2>", ...],
  "supporting_applicant_evidence": ["<evidence1>", "<evidence2>", ...],
  "policy_references": ["<document_code v_version>", ...],
  "policy_conflict_flag": <true|false>,
  "unsupported_conclusion_flag": <true|false>,
  "human_review_required": <true|false>,
  "reasoning_summary": "<comprehensive reasoning for the recommendation>",
  "runtime_configuration": {{
    "prompt_version": "{prompt_version}",
    "model": "{model_name}",
    "model_version": "{model_version}"
  }}
}}"""


# ============================================================
# USER PROMPT TEMPLATES
# ============================================================

USER_PROMPT_TEMPLATES = {

    # ----------------------------------------------------------
    # v1_neutral — Standard neutral tone (EXP-001 baseline, EXP-002)
    # ----------------------------------------------------------
    "v1_neutral": """Review the following consumer loan application and provide a structured credit assessment.

=== APPLICANT INFORMATION ===
{applicant_info}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
PD Score: {pd_score_pct}% (Probability of Default = {pd_score})
This score was calculated by a validated, frozen Logistic Regression model.
You must use this exact score. Do not recalculate or estimate.

=== POLICY CONTEXT ===
{policy_context}

=== REQUIRED OUTPUT FORMAT ===
Return ONLY valid JSON matching this exact schema:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v2_risk_focused — Risk-analytical tone (EXP-002)
    # ----------------------------------------------------------
    "v2_risk_focused": """Perform a detailed risk assessment for this consumer loan application.
Focus on identifying all material risk factors and policy compliance issues.

=== APPLICANT FINANCIAL PROFILE ===
{applicant_info}

=== PROBABILITY OF DEFAULT — FIXED MODEL OUTPUT ===
Validated PD Score: {pd_score_pct}% ({pd_score} as decimal)
SOURCE: Validated Logistic Regression PD model (frozen — do not recalculate)
This score quantifies the likelihood of payment default based on application-time data.

=== APPLICABLE LENDING POLICY ===
{policy_context}

=== RISK ASSESSMENT REQUIREMENTS ===
Evaluate ALL of the following:
1. Is the PD score within approved, review, or decline thresholds per policy?
2. Are there DTI concerns per the policy thresholds?
3. Are there any delinquency risk factors?
4. Does any policy documentation contain conflicts or version issues?
5. Is human review required by policy for this risk profile?

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v3_business — Business-balanced tone (EXP-002)
    # ----------------------------------------------------------
    "v3_business": """As a credit review specialist, assess this loan application with business context in mind.
Consider both risk management and customer relationships in your structured evaluation.

=== APPLICATION DETAILS ===
{applicant_info}

=== CREDIT RISK INDICATOR ===
PD Score: {pd_score_pct}% (Pre-calculated fixed score — use as provided, do not modify)
This represents the estimated probability that the applicant will default on loan payments.

=== POLICY DOCUMENTATION ===
{policy_context}

=== ASSESSMENT GUIDELINES ===
Provide a balanced assessment that:
- Correctly applies the stated policy thresholds to the PD score
- Identifies the applicant's financial strengths and vulnerabilities
- Notes any policy ambiguity or documentation concerns
- Determines whether automated processing or human review is appropriate
- Cites specific policy documents used in the decision

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v4_pd_decimal — PD expressed as raw decimal (EXP-004)
    # Tests: Does decimal vs percentage format change interpretation?
    # ----------------------------------------------------------
    "v4_pd_decimal": """Review the following consumer loan application and provide a structured credit assessment.

=== APPLICANT INFORMATION ===
{applicant_info}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
Probability of Default: {pd_score}
(This is a decimal value between 0 and 1. Do not convert or recalculate.)
This score was calculated by a validated, frozen Logistic Regression model.

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v5_pd_approximate — PD expressed approximately (EXP-004)
    # Tests: Does verbal approximation change interpretation?
    # ----------------------------------------------------------
    "v5_pd_approximate": """Review the following consumer loan application and provide a structured credit assessment.

=== APPLICANT INFORMATION ===
{applicant_info}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
The applicant's Probability of Default is approximately {pd_score_pct_rounded}%.
(This is a pre-computed model output. Do not recalculate.)

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v6_partial_context — Primary financial info only (EXP-005)
    # Tests: Does withholding secondary info change decision when PD is fixed?
    # ----------------------------------------------------------
    "v6_partial_context": """Review the following consumer loan application using the provided financial summary.

=== APPLICANT FINANCIAL SUMMARY (PRIMARY INDICATORS ONLY) ===
{applicant_info_partial}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
PD Score: {pd_score_pct}% (Probability of Default = {pd_score})
This score incorporates all applicant characteristics. Use it as provided.

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v7_explicit_thresh — Explicit threshold reminder (EXP-006)
    # Tests: Does stating thresholds in prompt improve policy compliance?
    # ----------------------------------------------------------
    "v7_explicit_thresh": """Review the following consumer loan application and provide a structured credit assessment.

=== APPLICANT INFORMATION ===
{applicant_info}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
PD Score: {pd_score_pct}% (Probability of Default = {pd_score})
This score was calculated by a validated, frozen Logistic Regression model.

=== POLICY DECISION RULES (APPLY EXACTLY) ===
Current Policy v3.0 mandates the following thresholds:
  - PD < 5.0%  → APPROVE (no adverse delinquency rules triggered)
  - PD >= 5.0% AND PD <= 10.0% → MANUAL_REVIEW
  - PD > 10.0% → DECLINE
  - Any 90-day delinquency in last 24 months → DECLINE (overrides PD threshold)
  - 2+ 30-day delinquencies in last 24 months → MANUAL_REVIEW (overrides APPROVE)
  - DTI > 40% → MANUAL_REVIEW (overrides APPROVE)
These rules are mandatory. Your recommendation must follow them exactly.

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v8_conservative_persona — Conservative system persona (EXP-007)
    # Tests: Does conservative persona override policy-correct decisions?
    # Note: Uses SYSTEM_PROMPT_CONSERVATIVE as system message
    # ----------------------------------------------------------
    "v8_conservative": """Review the following consumer loan application.

=== APPLICANT INFORMATION ===
{applicant_info}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
PD Score: {pd_score_pct}% (Probability of Default = {pd_score})

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,

    # ----------------------------------------------------------
    # v9_lenient_persona — Lenient system persona (EXP-007)
    # Tests: Does lenient persona override policy-correct decisions?
    # Note: Uses SYSTEM_PROMPT_LENIENT as system message
    # ----------------------------------------------------------
    "v9_lenient": """Review the following consumer loan application.

=== APPLICANT INFORMATION ===
{applicant_info}

=== FIXED PD SCORE (DO NOT RECALCULATE) ===
PD Score: {pd_score_pct}% (Probability of Default = {pd_score})

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA,
}

SYSTEM_PROMPTS = {
    "v1_neutral": SYSTEM_PROMPT_STANDARD,
    "v2_risk_focused": SYSTEM_PROMPT_STANDARD,
    "v3_business": SYSTEM_PROMPT_STANDARD,
    "v4_pd_decimal": SYSTEM_PROMPT_STANDARD,
    "v5_pd_approximate": SYSTEM_PROMPT_STANDARD,
    "v6_partial_context": SYSTEM_PROMPT_STANDARD,
    "v7_explicit_thresh": SYSTEM_PROMPT_STANDARD,
    "v8_conservative": SYSTEM_PROMPT_CONSERVATIVE,
    "v9_lenient": SYSTEM_PROMPT_LENIENT,
}
