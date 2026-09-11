import pandas as pd
import os
import zipfile
import xml.etree.ElementTree as ET
import asyncio
import json
import logging

# We need to manually load env vars for standalone script
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleAdapter:
    """Handles communication with the Google Gemini API with built-in rate limit handling."""
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
                content = response.text
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = 15 * (2 ** attempt)
                    logger.warning(f"Google API rate limit hit (429). Retrying in {delay}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                else:
                    raise e

        metadata = {"model": self._model_name}
        return content, metadata

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

# ---------------------------------------------------------
# Baseline Prompt
# ---------------------------------------------------------
SYSTEM_PROMPT_STANDARD = """You are a professional credit risk analyst.
Your role is to evaluate individual loan applications using the provided financial metrics and the strict synthetic lending policy.

CRITICAL INSTRUCTIONS:
- You must strictly apply the provided policy rules regarding Leverage, DSCR, LTV, Current Ratio, and PD.
- The Predicted PD is fixed and provided to you. Do NOT recalculate it.
- Count the number of Material Exceptions.
- Return your assessment exactly matching the JSON schema.
"""

USER_PROMPT_BASELINE = """Review the following application and provide a structured credit assessment.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code: {applicant_id}
Annual Income: {annual_income}
Loan Requested: {loan_requested}
Leverage: {leverage}x
DSCR: {dscr}x
LTV: {ltv}%
Current Ratio: {current_ratio}x
Documentation Complete: {documentation}

=== FIXED PD SCORE ===
Predicted PD: {pd_score}

=== POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON matching this exact schema:
""" + OUTPUT_SCHEMA

# ---------------------------------------------------------
# Conservative Prompt
# ---------------------------------------------------------
SYSTEM_PROMPT_CONSERVATIVE = """You are a highly cautious, risk-averse credit risk analyst.
Your primary obligation is to protect the institution from credit loss. When in doubt, prioritize declining or escalating exceptions.
You must return your response as valid JSON exactly matching the schema.
"""

USER_PROMPT_CONSERVATIVE = """Perform a strict, downside-risk focused assessment for this application.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code: {applicant_id}
Annual Income: {annual_income}
Loan Requested: {loan_requested}
Leverage: {leverage}x
DSCR: {dscr}x
LTV: {ltv}%
Current Ratio: {current_ratio}x
Documentation Complete: {documentation}

=== FIXED PD SCORE ===
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
            text = '\n'.join([p.text for p in tree.findall('.//w:t', ns) if p.text])
            return text
    except Exception as e:
        print(f"Failed to read {docx_path}: {e}")
        return ""

def build_knowledge_bases():
    """Loads and compiles policy documents into respective knowledge bases."""
    base_dir = "policy_documents"
    
    kb_2026_files = [
        "POL-01_2026-1.docx", "POL-02_2026-1.docx", "POL-03_2026-1.docx",
        "POL-04_2026-1.docx", "POL-05_2026-1.docx", "POL-06_2026-1.docx"
    ]
    
    kb_2025_files = [
        "POL-01_2025-1.docx", "POL-02_2025-1.docx", "POL-03_2025-1.docx",
        "POL-04_2026-1.docx", "POL-05_2026-1.docx", "POL-06_2026-1.docx" # using 2026 for 4,5,6 as 2025 doesn't exist
    ]
    
    kb_2026 = "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in kb_2026_files])
    kb_2025 = "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in kb_2025_files])
    
    # Alternate context for EXP-003 (just POL-01 and POL-02)
    kb_alt = "\n\n".join([extract_text_from_docx(os.path.join(base_dir, f)) for f in ["POL-01_2026-1.docx", "POL-02_2026-1.docx"]])
    
    return kb_2026, kb_2025, kb_alt

async def run_single_experiment(adapter, exp_code, prompt_template, sys_template, kb_text, row):
    user_prompt = prompt_template.format(
        applicant_id=row['Applicant Code'],
        annual_income=row['Annual Income'],
        loan_requested=row['Loan Requested'],
        leverage=row['Leverage'],
        dscr=row['DSCR'],
        ltv=row['LTV'],
        current_ratio=row['Current Ratio'],
        documentation=row['Documentation Complete'],
        pd_score=row['Predicted PD'],
        policy_context=kb_text
    )
    
    try:
        content, _ = await adapter.complete(sys_template, user_prompt, temperature=0.0, max_tokens=1000)
        
        # Clean markdown formatting if present
        content = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        return exp_code, result.get("recommendation", "ERROR")
    except Exception as e:
        print(f"Error on {exp_code} for {row['Applicant Code']}: {e}")
        return exp_code, "ERROR"

async def process_borrower(row, kb_2026, kb_2025, kb_alt, adapter_A, adapter_B, semaphore):
    async with semaphore:
        results = {"Applicant Code": row['Applicant Code']}
        
        # Execute 7 defined experiments
        tasks = [
            # EXP-001: Baseline Run
            run_single_experiment(adapter_A, "EXP-001", USER_PROMPT_BASELINE, SYSTEM_PROMPT_STANDARD, kb_2026, row),
            
            # EXP-002: Prompt Variation
            run_single_experiment(adapter_A, "EXP-002", USER_PROMPT_CONSERVATIVE, SYSTEM_PROMPT_CONSERVATIVE, kb_2026, row),
            
            # EXP-003: Retrieved Context Variation 
            run_single_experiment(adapter_A, "EXP-003", USER_PROMPT_BASELINE, SYSTEM_PROMPT_STANDARD, kb_alt, row),
            
            # EXP-004: Knowledge Version Variation
            run_single_experiment(adapter_A, "EXP-004", USER_PROMPT_BASELINE, SYSTEM_PROMPT_STANDARD, kb_2025, row),
            
            # EXP-005: Model Variation
            run_single_experiment(adapter_B, "EXP-005", USER_PROMPT_BASELINE, SYSTEM_PROMPT_STANDARD, kb_2026, row),
            
            # EXP-006: Repeatability 
            run_single_experiment(adapter_A, "EXP-006", USER_PROMPT_BASELINE, SYSTEM_PROMPT_STANDARD, kb_2026, row),
            
            # EXP-007: Combined Runtime Change 
            run_single_experiment(adapter_B, "EXP-007", USER_PROMPT_CONSERVATIVE, SYSTEM_PROMPT_CONSERVATIVE, kb_alt, row)
        ]
        
        exp_results = await asyncio.gather(*tasks)
        for exp_code, rec in exp_results:
            results[exp_code] = rec
            
        return results

async def async_main():
    """Main execution block for GenAI experiments."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.startswith("your_google_api_key"):
        print("CRITICAL ERROR: Valid GOOGLE_API_KEY not found in .env!")
        return

    # Initialize Models from .env
    primary_model = os.getenv("PRIMARY_MODEL", "gemini-1.5-flash")
    secondary_model = os.getenv("SECONDARY_MODEL", "gemini-1.5-pro")
    
    adapter_A = GoogleAdapter(api_key, primary_model)
    adapter_B = GoogleAdapter(api_key, secondary_model)
    
    kb_2026, kb_2025, kb_alt = build_knowledge_bases()
    
    input_path = "data/processed/03_master_fixed_dataset.csv"
    output_path = "data/processed/05_experiment_results.csv"
    
    df = pd.read_csv(input_path)
    # We will process 500 borrowers, but cap concurrency to avoid 429 rate limits
    semaphore = asyncio.Semaphore(5) 
    
    print(f"Executing 7 experiments for {len(df)} borrowers (Total calls: {len(df) * 7})...")
    print("This may take several minutes. Rate limits will be handled automatically with backoff.")
    
    tasks = [process_borrower(row, kb_2026, kb_2025, kb_alt, adapter_A, adapter_B, semaphore) for _, row in df.iterrows()]
    all_results = await asyncio.gather(*tasks)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)
    print(f"\nAll experiments complete! Saved to {output_path}")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
