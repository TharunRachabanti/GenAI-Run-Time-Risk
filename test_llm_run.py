print("Initializing Test LLM Run (All Experiments, 5 Records)... (Loading libraries)")
import pandas as pd
import os
import zipfile
import xml.etree.ElementTree as ET
import asyncio
import json
import logging

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleAdapter:
    """Handles communication with the Google Gemini API."""
    def __init__(self, api_key: str, model_name: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model_name = model_name
        except ImportError:
            raise ImportError("google-generativeai package required")

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> tuple[str, dict]:
        gemini_model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
        )
        generation_config = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
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
                return response.text, {"model": self._model_name}
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    delay = 15 * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                elif "404" in err and "gemini-1.5-pro" in self._model_name:
                    # Free tier does not support gemini-1.5-pro — fall back to Flash
                    logger.warning(f"Model {self._model_name} not available on free tier. Falling back to gemini-1.5-flash.")
                    self._model_name = "gemini-1.5-flash"
                    gemini_model = self._genai.GenerativeModel(
                        model_name=self._model_name,
                        system_instruction=system_prompt,
                    )
                else:
                    raise e

OUTPUT_SCHEMA = """{{
  "applicant_id": "{applicant_id}",
  "reasoning_summary": "<MANDATORY: 1-sentence explanation citing the key policy measure(s)>",
  "recommendation": "<APPROVE|APPROVE_WITH_CONDITIONS|DECLINE>",
  "material_exceptions_count": <integer>
}}"""

SYSTEM_PROMPT_STANDARD = """You are a professional credit risk analyst evaluating consumer loan applications.
Your role is to apply the provided consumer lending policy rules to each application.

CRITICAL INSTRUCTIONS:
- Apply the provided policy rules using the exact measures: PTI, CTI, LGV, PD, and Borrower Stability.
- The Predicted PD is fixed and frozen. Do NOT recalculate it.
- Classify each policy dimension as Standard, Review/Elevated, or High/High Concern.
- Count High findings and Review findings, then apply the three-tier decision table.
- Return your assessment exactly matching the JSON schema.
"""

USER_PROMPT_BASELINE = """Review the following consumer loan application and provide a structured credit assessment.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code: {applicant_id}
Annual Income: {annual_income}
Loan Requested: {loan_requested}
Goods Price: {goods_price}
Monthly Annuity: {monthly_annuity}

=== DERIVED POLICY MEASURES ===
PTI (Payment-to-Income): {pti}%
CTI (Credit-to-Income): {cti}x
LGV (Loan-to-Goods-Value): {lgv}%
Employment Tenure: {employment_years} years
Monthly Inquiries: {inq_mon} | Quarterly: {inq_qrt} | Annual: {inq_year}

=== FROZEN PD SCORE ===
Predicted PD: {pd_score}

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON matching this exact schema:
""" + OUTPUT_SCHEMA

SYSTEM_PROMPT_CONSERVATIVE = """You are a highly cautious, risk-averse credit risk analyst evaluating consumer loan applications.
Your primary obligation is to protect the institution from credit loss. When in doubt, prioritize declining or escalating exceptions.
Apply the consumer lending policy measures (PTI, CTI, LGV, PD, Borrower Stability) strictly.
You must return your response as valid JSON exactly matching the schema.
"""

USER_PROMPT_CONSERVATIVE = """Perform a strict, downside-risk focused assessment for this consumer loan application.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code: {applicant_id}
Annual Income: {annual_income}
Loan Requested: {loan_requested}
Goods Price: {goods_price}
Monthly Annuity: {monthly_annuity}

=== DERIVED POLICY MEASURES ===
PTI (Payment-to-Income): {pti}%
CTI (Credit-to-Income): {cti}x
LGV (Loan-to-Goods-Value): {lgv}%
Employment Tenure: {employment_years} years
Monthly Inquiries: {inq_mon} | Quarterly: {inq_qrt} | Annual: {inq_year}

=== FROZEN PD SCORE ===
Predicted PD: {pd_score}

=== POLICY CONTEXT ===
{policy_context}

Evaluate strictly and return ONLY valid JSON matching this schema:
""" + OUTPUT_SCHEMA


def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text_nodes = [p.text.strip() for p in tree.findall('.//w:t', ns) if p.text and p.text.strip()]
            raw_text = ' '.join(text_nodes)
            # Aggressively minify whitespace to save tokens
            import re
            minified_text = re.sub(r'\s+', ' ', raw_text)
            return minified_text
    except Exception as e:
        print(f"Failed to read {docx_path}: {e}")
        return ""

def build_knowledge_bases():
    base_dir = "policy_documents"
    kb_2026_files = [
        "POL-01_2026-1_Affordability_Policy.docx",
        "POL-02_2026-1_Credit_Exposure_Policy.docx",
        "POL-03_2026-1_Financing_to_Value_Policy.docx",
        "POL-04_2026-1_Credit_Risk_Rating_Policy.docx",
        "POL-05_2026-1_Borrower_Stability_Policy.docx",
        "Rule_Based_Decision_Engine_Frozen_Ground_Truth.docx",
    ]
    kb_2025_files = [
        "POL-01_2025-1_Affordability_Policy.docx",
        "POL-02_2025-1_Credit_Exposure_Policy.docx",
        "POL-03_2026-1_Financing_to_Value_Policy.docx",
        "POL-04_2026-1_Credit_Risk_Rating_Policy.docx",
        "POL-05_2026-1_Borrower_Stability_Policy.docx",
        "Rule_Based_Decision_Engine_Frozen_Ground_Truth.docx",
    ]
    kb_2026 = "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in kb_2026_files])
    kb_2025 = "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in kb_2025_files])
    kb_alt  = "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in [
        "POL-01_2026-1_Affordability_Policy.docx",
        "POL-02_2026-1_Credit_Exposure_Policy.docx",
    ]])
    return kb_2026, kb_2025, kb_alt

async def run_single_experiment(adapter, exp_code, prompt_template, sys_template, kb_text, row):
    import re
    user_prompt = prompt_template.format(
        applicant_id=row['Applicant Code'],
        annual_income=row['Annual Income'],
        loan_requested=row['Loan Requested'],
        goods_price=row['Goods Price'],
        monthly_annuity=row['Monthly Annuity'],
        pti=row['PTI (%)'],
        cti=row['CTI (x)'],
        lgv=row['LGV (%)'],
        employment_years=row['Employment Tenure (Years)'],
        inq_mon=row['Inquiries (Month)'],
        inq_qrt=row['Inquiries (Quarter)'],
        inq_year=row['Inquiries (Year)'],
        pd_score=row['Predicted PD'],
        policy_context=kb_text,
        prompt_version=exp_code,
        model_name=adapter._model_name
    )
    max_retries = 3
    for attempt in range(max_retries):
        try:
            content, _ = await adapter.complete(sys_template, user_prompt, temperature=0.0, max_tokens=800)
            content_clean = content.replace("```json", "").replace("```", "").strip()
            
            try:
                result = json.loads(content_clean)
                rec = result.get("recommendation", "ERROR")
                rsn = result.get("reasoning_summary", "")
                if rec != "ERROR" and rsn.strip() != "":
                    return exp_code, rec, rsn
            except json.JSONDecodeError:
                # Fallback: robust regex to extract from broken JSON
                rec_match = re.search(r'"recommendation"\s*:\s*"([^"]+)"', content_clean)
                rsn_match = re.search(r'"reasoning_summary"\s*:\s*"([\s\S]*?)(?:",|"\s*}|$)', content_clean)
                
                rec = rec_match.group(1) if rec_match else "ERROR"
                rsn = rsn_match.group(1).strip() if rsn_match else ""
                
                # If the string was truncated, rsn might have captured up to the truncation point.
                # But since recommendation is AFTER reasoning now, if recommendation is found, reasoning MUST be complete.
                rsn = re.sub(r'\\?["\\]*$', '', rsn)
                
                if rec != "ERROR" and rsn.strip() != "":
                    return exp_code, rec, rsn
                    
            # If we reached here, recommendation is still ERROR (e.g. garbled response)
            if attempt < max_retries - 1:
                print(f"    [WARN] {exp_code} returned garbled output. Retrying in 10s...")
                await asyncio.sleep(10)
            else:
                return exp_code, "ERROR", ""
                
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(10)
            else:
                print(f"  ERROR on {exp_code} for {row['Applicant Code']}: {e}")
                return exp_code, "ERROR", ""

async def async_main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_google_api_key"):
        print("CRITICAL ERROR: Valid GOOGLE_API_KEY not found in .env!")
        return

    primary_model   = os.getenv("PRIMARY_MODEL", "gemini-2.0-flash")
    # In test mode, use the same model for all experiments to avoid 404 on free tier
    # The real model variation (EXP-005/007) will use SECONDARY_MODEL in the full run
    adapter_A = GoogleAdapter(api_key, primary_model)
    adapter_B = adapter_A  # Same adapter for test — avoids Pro 404 on free tier

    kb_2026, kb_2025, kb_alt = build_knowledge_bases()

    input_path     = "data/processed/03_master_fixed_dataset.csv"
    benchmark_path = "data/processed/04_benchmark_results.csv"

    df        = pd.read_csv(input_path)
    bench_df  = pd.read_csv(benchmark_path)

    # ---------------------------------------------------------
    # TEST: Run all 7 experiments for 1 applicant only
    # Once confirmed error-free, add more IDs here
    # ---------------------------------------------------------
    test_ids = ["P001"]
    test_df  = df[df['Applicant Code'].isin(test_ids)]

    print(f"\n--- RUNNING ALL 7 EXPERIMENTS FOR 1 APPLICANT: {test_ids} ---")
    print(f"    Total API calls: {len(test_df) * 7}\n")

    all_rows = []
    reasoning_rows = []  # Separate list to capture detailed reasoning
    for _, row in test_df.iterrows():
        print(f"  Evaluating {row['Applicant Code']}...")
        results = {"Applicant Code": row['Applicant Code']}
        
        # Run experiments SEQUENTIALLY to avoid flooding free-tier rate limits
        exp_configs = [
            (adapter_A, "EXP-001", USER_PROMPT_BASELINE,     SYSTEM_PROMPT_STANDARD,      kb_2026),
            (adapter_A, "EXP-002", USER_PROMPT_CONSERVATIVE, SYSTEM_PROMPT_CONSERVATIVE,  kb_2026),
            (adapter_A, "EXP-003", USER_PROMPT_BASELINE,     SYSTEM_PROMPT_STANDARD,      kb_alt),
            (adapter_A, "EXP-004", USER_PROMPT_BASELINE,     SYSTEM_PROMPT_STANDARD,      kb_2025),
            (adapter_B, "EXP-005", USER_PROMPT_BASELINE,     SYSTEM_PROMPT_STANDARD,      kb_2026),
            (adapter_A, "EXP-006", USER_PROMPT_BASELINE,     SYSTEM_PROMPT_STANDARD,      kb_2026),
            (adapter_B, "EXP-007", USER_PROMPT_CONSERVATIVE, SYSTEM_PROMPT_CONSERVATIVE,  kb_alt),
        ]
        for adp, exp_code, prompt_t, sys_t, kb in exp_configs:
            exp_result_code, recommendation, reasoning = await run_single_experiment(adp, exp_code, prompt_t, sys_t, kb, row)
            results[exp_result_code] = recommendation
            print(f"    {exp_code}: {recommendation}")
            reasoning_rows.append({
                "Borrower":        row['Applicant Code'],
                "Experiment":      exp_code,
                "Recommendation":  recommendation,
                "Reasoning":       reasoning,
            })
            # 15-second pause to prevent Google free-tier 32,000 TPM limit truncation
            await asyncio.sleep(15)
        
        all_rows.append(results)

    # Build the results dataframe
    results_df = pd.DataFrame(all_rows)

    # Merge with Ground Truth (Rule Based Decision) from benchmark file
    bench_subset = bench_df[['Applicant Code', 'Rule_Based_Decision']].rename(
        columns={'Rule_Based_Decision': 'Rule Based'}
    )
    final_df = pd.merge(bench_subset, results_df, on='Applicant Code', how='right')
    final_df = final_df.rename(columns={'Applicant Code': 'Borrower'})

    # Reorder columns to match client format
    cols = ['Borrower', 'Rule Based', 'EXP-001', 'EXP-002', 'EXP-003', 'EXP-004', 'EXP-005', 'EXP-006', 'EXP-007']
    final_df = final_df[[c for c in cols if c in final_df.columns]]

    # Write TWO sheets into one single Excel file
    reasoning_df = pd.DataFrame(reasoning_rows)
    output_path = "data/processed/test_5records_all_experiments.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="Decision Matrix", index=False)
        reasoning_df.to_excel(writer, sheet_name="Reasoning", index=False)

    print(f"\n--- TEST COMPLETE ---")
    print(f"Results saved to: {output_path}")
    print(final_df.to_string(index=False))

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
