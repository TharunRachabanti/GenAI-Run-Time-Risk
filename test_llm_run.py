print("Initializing Test LLM Run... (Loading libraries)")
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
        
        response = await asyncio.to_thread(
            gemini_model.generate_content, full_prompt,
            generation_config=generation_config,
        )
        content = response.text
        return content, {"model": self._model_name}

OUTPUT_SCHEMA = """{{
  "applicant_id": "{applicant_id}",
  "pd_score": {pd_score},
  "recommendation": "<APPROVE|APPROVE_WITH_CONDITIONS|DECLINE|INCOMPLETE>",
  "reasoning_summary": "<comprehensive reasoning for the recommendation>",
  "material_exceptions_count": <integer>,
  "runtime_configuration": {{
    "prompt_version": "{prompt_version}",
    "model": "{model_name}"
  }}
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

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as z:
            xml_content = z.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text = '\n'.join([p.text for p in tree.findall('.//w:t', ns) if p.text])
            return text
    except Exception as e:
        print(f"Failed to read {docx_path}: {e}")
        return ""

def build_knowledge_base():
    """Loads 2026.1 consumer policies."""
    base_dir = "policy_documents"
    kb_2026_files = [
        "POL-01_2026-1_Affordability_Policy.docx",
        "POL-02_2026-1_Credit_Exposure_Policy.docx",
        "POL-03_2026-1_Financing_to_Value_Policy.docx",
        "POL-04_2026-1_Credit_Risk_Rating_Policy.docx",
        "POL-05_2026-1_Borrower_Stability_Policy.docx",
        "Rule_Based_Decision_Engine_Frozen_Ground_Truth.docx",
    ]
    return "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in kb_2026_files])

async def run_single_experiment(adapter, exp_code, prompt_template, sys_template, kb_text, row):
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
        policy_context=kb_text
    )
    
    try:
        content, _ = await adapter.complete(sys_template, user_prompt, temperature=0.0, max_tokens=1000)
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return exp_code, result
    except Exception as e:
        print(f"Error on {exp_code} for {row['Applicant Code']}: {e}")
        return exp_code, {"error": str(e)}

async def async_main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_google_api_key"):
        print("CRITICAL ERROR: Valid GOOGLE_API_KEY not found in .env!")
        return

    primary_model = os.getenv("PRIMARY_MODEL", "gemini-1.5-flash")
    adapter = GoogleAdapter(api_key, primary_model)
    kb_2026 = build_knowledge_base()
    
    input_path = "data/processed/03_master_fixed_dataset.csv"
    df = pd.read_csv(input_path)
    
    # ---------------------------------------------------------
    # TEST ONLY: Filter to ONE perfect test record
    # ---------------------------------------------------------
    test_ids = ["P001"]
    test_df = df[df['Applicant Code'].isin(test_ids)]
    
    print(f"\n--- RUNNING TEST FOR 1 APPLICANT: {test_ids} ---")
    
    all_results = []
    for _, row in test_df.iterrows():
        print(f"\nEvaluating {row['Applicant Code']}...")
        _, result = await run_single_experiment(
            adapter, "EXP-001", USER_PROMPT_BASELINE, SYSTEM_PROMPT_STANDARD, kb_2026, row
        )
        print(json.dumps(result, indent=2))
        all_results.append(result)
        
    # Save the output to a CSV file (can be opened in Excel)
    output_path = "data/processed/test_single_record_output.csv"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- TEST COMPLETE: Saved to {output_path} ---")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
