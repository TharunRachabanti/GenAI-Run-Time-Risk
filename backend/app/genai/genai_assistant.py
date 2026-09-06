"""
GenAI Credit Review Assistant
Single-provider architecture: uses GENAI_PROVIDER from config/env.

CRITICAL DESIGN PRINCIPLE:
- The PD score passed to this assistant is FIXED and FROZEN.
- The assistant MUST NOT recalculate or modify the PD score.
- Structured output is validated against GenAIStructuredAssessment schema.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.schemas.schemas import GenAIStructuredAssessment
from app.genai.prompt_templates import (
    USER_PROMPT_TEMPLATES, SYSTEM_PROMPTS
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================
# LLM CONFIG VALIDATOR
# ============================================================
def check_llm_configured() -> Tuple[bool, str]:
    """
    Check if the LLM is properly configured in .env.

    Returns:
        (is_configured: bool, error_message: str)
    """
    if settings.is_llm_configured():
        return True, ""

    provider = settings.genai_provider
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    key_name = key_map.get(provider, "API_KEY")
    msg = (
        f"\n[bold red]FAIL LLM Not Configured[/bold red]\n\n"
        f"  Provider: [bold]{provider}[/bold]\n"
        f"  Required key: [bold]{key_name}[/bold] — currently a placeholder or missing.\n\n"
        f"  To fix:\n"
        f"  1. Open the [bold].env[/bold] file in the project root.\n"
        f"  2. Set  GENAI_PROVIDER={provider}\n"
        f"  3. Set  {key_name}=<your-real-api-key>\n"
        f"  4. Save the file and run the command again.\n"
    )
    return False, msg


# ============================================================
# PROVIDER ADAPTERS
# ============================================================
class OpenAIAdapter:
    def __init__(self, api_key: str, model_name: str):
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=api_key)
            self._model = model_name
        except ImportError:
            raise ImportError("openai package required: pip install openai")

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        metadata = {
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "finish_reason": response.choices[0].finish_reason,
        }
        return content, metadata


class AnthropicAdapter:
    def __init__(self, api_key: str, model_name: str):
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._model = model_name
        except ImportError:
            raise ImportError("anthropic package required: pip install anthropic")

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        response = await self._client.messages.create(
            model=self._model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.content[0].text
        metadata = {
            "model": response.model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "stop_reason": response.stop_reason,
        }
        return content, metadata


class GoogleAdapter:
    def __init__(self, api_key: str, model_name: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model_name = model_name
        except ImportError:
            raise ImportError("google-generativeai package required")

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        import asyncio
        gemini_model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
        )
        generation_config = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )
        full_prompt = user_prompt + "\n\nRespond with ONLY valid JSON."
        
        max_retries = 6
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    gemini_model.generate_content, full_prompt,
                    generation_config=generation_config,
                )
                content = response.text
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = 15 * (2 ** attempt)
                    logger.warning(f"Google API rate limit hit (429). Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                else:
                    raise e

        metadata = {"model": self._model_name, "candidates": len(response.candidates)}
        return content, metadata


class MockAdapter:
    """Mock adapter for testing without a real LLM key."""

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> Tuple[str, Dict]:
        pd_score_match = re.search(
            r"PD Score: (\d+\.?\d*)%?|Probability of Default[:\s=]+(\d+\.?\d*)", user_prompt
        )
        pd_score = 0.05
        if pd_score_match:
            pd_str = pd_score_match.group(1) or pd_score_match.group(2)
            try:
                val = float(pd_str)
                pd_score = val / 100.0 if val > 1.0 else val
            except ValueError:
                pass

        if pd_score > 0.10:
            rec, risk = "DECLINE", "HIGH_RISK"
        elif pd_score >= 0.05:
            rec, risk = "MANUAL_REVIEW", "MODERATE_RISK"
        else:
            rec, risk = "APPROVE", "LOW_RISK"

        if "90-day Delinquencies (last 24 months): 1" in user_prompt:
            rec, risk = "DECLINE", "HIGH_RISK"

        app_id_match = re.search(r'"applicant_id":\s*"([^"]+)"', user_prompt)
        app_id = app_id_match.group(1) if app_id_match else "mock-id"

        prompt_ver_match = re.search(r'"prompt_version":\s*"([^"]+)"', user_prompt)
        prompt_version = prompt_ver_match.group(1) if prompt_ver_match else "v1_neutral"

        mock_response = {
            "applicant_id": app_id,
            "pd_score": pd_score,
            "risk_classification": risk,
            "recommendation": rec,
            "pd_interpretation": f"PD score of {pd_score:.4f} ({pd_score*100:.2f}%) classifies as {risk}.",
            "primary_risk_factors": ["PD Score above threshold"] if risk != "LOW_RISK" else [],
            "supporting_applicant_evidence": ["Applicant financial profile reviewed."],
            "policy_references": ["POLICY-CL-001 v3.0"],
            "policy_conflict_flag": False,
            "unsupported_conclusion_flag": False,
            "human_review_required": rec == "MANUAL_REVIEW",
            "reasoning_summary": (
                f"Based on the fixed PD score of {pd_score:.4f} and policy v3.0 thresholds, "
                f"the recommendation is {rec}."
            ),
            "runtime_configuration": {
                "prompt_version": prompt_version,
                "model": "mock-model",
                "model_version": "1.0",
            },
        }
        import asyncio
        await asyncio.sleep(0.1)
        return json.dumps(mock_response), {"model": "mock-model", "usage": {"total_tokens": 100}}


# ============================================================
# GENAI ASSISTANT (single-provider)
# ============================================================
class GenAIAssistant:
    """
    GenAI Credit Review Assistant.
    Provider is determined by settings.genai_provider (from .env).
    """

    def __init__(self):
        self.provider = settings.genai_provider.lower()
        self.model_name = settings.genai_model_name
        self.model_version = settings.genai_model_version
        self.temperature = settings.genai_temperature
        self.max_tokens = settings.genai_max_tokens

        # Use mock if LLM is not configured (graceful fallback for testing)
        if not settings.is_llm_configured():
            self._adapter = MockAdapter()
            self.provider = "mock"
            logger.warning("LLM not configured — using MockAdapter.")
        elif self.provider == "openai":
            self._adapter = OpenAIAdapter(settings.openai_api_key, self.model_name)
        elif self.provider == "anthropic":
            self._adapter = AnthropicAdapter(settings.anthropic_api_key, self.model_name)
        elif self.provider == "google":
            self._adapter = GoogleAdapter(settings.google_api_key, self.model_name)
        elif self.provider == "mock":
            self._adapter = MockAdapter()
        else:
            raise ValueError(f"Unsupported GENAI_PROVIDER: {self.provider}")

    async def assess(
        self,
        applicant_id: str,
        applicant_data: Dict[str, Any],
        fixed_pd_score: float,
        retrieved_docs: List[Dict[str, Any]],
        prompt_version: str = "v1_neutral",
        system_prompt_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a GenAI credit assessment.
        The fixed_pd_score is NEVER recalculated here.
        """
        system_prompt = system_prompt_override or SYSTEM_PROMPTS.get(
            prompt_version, SYSTEM_PROMPTS["v1_neutral"]
        )
        user_template = USER_PROMPT_TEMPLATES.get(prompt_version, USER_PROMPT_TEMPLATES["v1_neutral"])

        applicant_info = self._format_applicant_info(applicant_data)
        applicant_info_partial = self._format_applicant_info_partial(applicant_data)
        policy_context = self._build_policy_context(retrieved_docs)
        pd_score_pct = round(fixed_pd_score * 100, 4)
        pd_score_pct_rounded = round(fixed_pd_score * 100)

        try:
            user_prompt = user_template.format(
                applicant_id=applicant_id,
                applicant_info=applicant_info,
                applicant_info_partial=applicant_info_partial,
                pd_score=fixed_pd_score,
                pd_score_pct=pd_score_pct,
                pd_score_pct_rounded=pd_score_pct_rounded,
                policy_context=policy_context,
                prompt_version=prompt_version,
                model_name=self.model_name,
                model_version=self.model_version,
            )
        except KeyError as e:
            logger.error(f"Template format error for {prompt_version}: {e}")
            user_prompt = USER_PROMPT_TEMPLATES["v1_neutral"].format(
                applicant_id=applicant_id,
                applicant_info=applicant_info,
                applicant_info_partial=applicant_info_partial,
                pd_score=fixed_pd_score,
                pd_score_pct=pd_score_pct,
                pd_score_pct_rounded=pd_score_pct_rounded,
                policy_context=policy_context,
                prompt_version=prompt_version,
                model_name=self.model_name,
                model_version=self.model_version,
            )

        try:
            raw_response, response_metadata = await self._adapter.complete(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"GenAI API call failed: {e}")
            return {
                "raw_response": "",
                "structured_assessment": None,
                "parse_success": False,
                "parse_error": str(e),
                "response_metadata": {},
                "prompt_version": prompt_version,
                "prompt_hash": "",
            }

        structured_assessment, parse_success, parse_error = self._parse_and_validate(
            raw_response, applicant_id, fixed_pd_score
        )

        return {
            "raw_response": raw_response,
            "structured_assessment": structured_assessment,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "response_metadata": response_metadata,
            "prompt_version": prompt_version,
            "prompt_hash": hashlib.sha256(
                (system_prompt + user_prompt).encode()
            ).hexdigest(),
        }

    def _format_applicant_info(self, data: Dict[str, Any]) -> str:
        """Full applicant info string for prompts."""
        field_labels = {
            "applicant_name": "Applicant Name",
            "annual_income": "Annual Income",
            "loan_amount_requested": "Loan Amount Requested",
            "loan_purpose": "Loan Purpose",
            "occupancy_type": "Occupancy Type",
            "loan_type": "Loan Type",
            "property_value": "Property Value",
            "loan_term_months": "Loan Term (months)",
            "interest_rate": "Interest Rate (%)",
            "debt_to_income_ratio": "Debt-to-Income Ratio (%)",
            "combined_loan_to_value_ratio": "Combined LTV Ratio (%)",
            "employment_status": "Employment Status",
            "employment_years": "Employment Duration (years)",
            "credit_history_years": "Credit History Length (years)",
            "delinquency_90day_count": "90-day Delinquencies (last 24 months)",
            "delinquency_30day_count": "30-day Delinquencies (last 24 months)",
            "monthly_annuity": "Monthly Payment (annuity)",
            "credit_to_income_ratio": "Credit-to-Income Ratio",
            "external_credit_score_proxy": "Credit Score Indicator (0=high risk, 1=low risk)",
        }
        lines = []
        for field, label in field_labels.items():
            val = data.get(field)
            if val is not None:
                if field in ("annual_income", "loan_amount_requested", "property_value", "monthly_annuity"):
                    lines.append(f"  {label}: ${val:,.2f}")
                elif field in ("debt_to_income_ratio", "combined_loan_to_value_ratio", "interest_rate"):
                    lines.append(f"  {label}: {val:.1f}%")
                elif field == "external_credit_score_proxy":
                    lines.append(f"  {label}: {val:.3f}")
                else:
                    lines.append(f"  {label}: {val}")
        return "\n".join(lines)

    def _format_applicant_info_partial(self, data: Dict[str, Any]) -> str:
        """Primary financial indicators only (for EXP-005 contextual completeness test)."""
        primary_fields = {
            "annual_income": "Annual Income",
            "loan_amount_requested": "Loan Amount Requested",
            "debt_to_income_ratio": "Debt-to-Income Ratio (%)",
            "delinquency_90day_count": "90-day Delinquencies (last 24 months)",
            "delinquency_30day_count": "30-day Delinquencies (last 24 months)",
        }
        lines = []
        for field, label in primary_fields.items():
            val = data.get(field)
            if val is not None:
                if field in ("annual_income", "loan_amount_requested"):
                    lines.append(f"  {label}: ${val:,.2f}")
                elif field == "debt_to_income_ratio":
                    lines.append(f"  {label}: {val:.1f}%")
                else:
                    lines.append(f"  {label}: {val}")
        return "\n".join(lines)

    def _build_policy_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build policy context string from retrieved documents."""
        if not retrieved_docs:
            return "No policy documents were retrieved for this assessment."
        sections = []
        for doc in retrieved_docs:
            is_auth = doc.get("is_authoritative", True)
            status = doc.get("status", "unknown")
            status_label = (
                "[CURRENT AUTHORITATIVE POLICY]"
                if is_auth and status == "current"
                else f"[{status.upper()} — {'AUTHORITATIVE' if is_auth else 'NOT AUTHORITATIVE'}]"
            )
            sections.append(
                f"--- POLICY DOCUMENT ---\n"
                f"Code: {doc.get('document_code', 'N/A')}\n"
                f"Title: {doc.get('title', 'N/A')}\n"
                f"Version: {doc.get('version', 'N/A')}\n"
                f"Effective Date: {doc.get('effective_date', 'N/A')}\n"
                f"Status: {status_label}\n\n"
                f"{doc.get('content', '')}"
            )
        return "\n\n".join(sections)

    def _parse_and_validate(self, raw_response: str, applicant_id: str,
                            fixed_pd_score: float):
        """Parse raw model output into validated GenAIStructuredAssessment."""
        json_str = self._extract_json(raw_response)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return None, False, f"JSON parse error: {e}"

        # Enforce fixed PD score
        pd_returned = data.get("pd_score", fixed_pd_score)
        if abs(float(pd_returned) - fixed_pd_score) > 0.001:
            logger.warning(
                f"Model returned different PD {pd_returned} vs fixed {fixed_pd_score}. Overriding."
            )
            data["pd_score"] = fixed_pd_score

        data["applicant_id"] = applicant_id

        try:
            assessment = GenAIStructuredAssessment(**data)
            return assessment, True, None
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            return None, False, f"Schema validation error: {e}"

    def _extract_json(self, text: str) -> str:
        """Extract JSON from model output that may contain markdown."""
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block_match:
            return code_block_match.group(1).strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            return json_match.group(0)
        return text.strip()
