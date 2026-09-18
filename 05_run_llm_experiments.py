"""
05_run_llm_experiments.py  (v2 — full RAG + state cache)
=========================================================
Runs all LLM experiments over the master dataset.

Components added in v2:
  - RAGRetriever class         (loads FAISS once, deterministic query translation)
  - StateCache class           (append-only JSONL, fully resumable)
  - SYSTEM_PROMPT_COT          (Chain-of-Thought + mandatory citation — EXP-002-P3)
  - Master Decision Matrix     hardcoded into ALL system prompts (not via RAG)
  - Experiment registry        6 canonical experiments aligned with client spec
  - Retrieved context logging  every result row records which policies were fetched

Run 05b_build_rag_index.py first to generate data/rag/*.
"""

print("Initializing LLM Experiments v2... (Loading libraries)")
import pandas as pd
import os
import asyncio
import json
import logging
import re
import time
import numpy as np
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. GOOGLE GEMINI ADAPTER
# ==============================================================================

class GoogleAdapter:
    """Handles Gemini API calls with rate-limit back-off and free-tier 404 fallback."""

    def __init__(self, api_key: str, model_name: str):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model_name = model_name
        except ImportError:
            raise ImportError("Run: pip install google-generativeai")

    async def complete(self, system_prompt: str, user_prompt: str,
                       temperature: float, max_tokens: int) -> tuple[str, dict]:
        gemini_model = self._genai.GenerativeModel(
            model_name=self._model_name,
            system_instruction=system_prompt,
        )
        gen_cfg = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
        )
        full_prompt = user_prompt + "\n\nRespond with ONLY valid JSON."

        max_retries = 6
        content = ""
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    gemini_model.generate_content, full_prompt,
                    generation_config=gen_cfg,
                )
                content = response.text
                break
            except Exception as e:
                err = str(e)
                if "429" in err and attempt < max_retries - 1:
                    delay = 15 * (2 ** attempt)
                    logger.warning(f"Rate limit. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                elif "404" in err and "gemini-1.5-pro" in self._model_name:
                    logger.warning("Pro model unavailable on free tier. Falling back to Flash.")
                    self._model_name = "gemini-2.0-flash"
                    gemini_model = self._genai.GenerativeModel(
                        model_name=self._model_name,
                        system_instruction=system_prompt,
                    )
                else:
                    raise e

        return content, {"model": self._model_name}


# ==============================================================================
# 2. RAG RETRIEVER
# ==============================================================================

class RAGRetriever:
    """
    Loads the FAISS index once and provides semantic retrieval.
    Uses deterministic query translation so numerical borrower data
    is never used directly as a FAISS query vector.
    """

    # Fixed natural-language queries — one per policy dimension.
    # These are semantically rich enough to retrieve the correct threshold tables.
    FIXED_QUERIES = [
        "What are the policy rules and thresholds for Payment-to-Income (PTI) ratio affordability?",
        "What are the policy rules and thresholds for Credit-to-Income (CTI) ratio credit exposure?",
        "What are the policy rules and thresholds for Loan-to-Goods-Value (LGV) financing to value ratio?",
        "What are the policy rules and thresholds for Probability of Default (PD) credit risk rating?",
        "What are the policy rules for borrower stability, employment tenure, and credit bureau inquiries?",
    ]

    INDEX_PATH  = "data/rag/policy_faiss.index"
    CHUNKS_PATH = "data/rag/policy_chunks.json"
    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "RAG dependencies missing. Run:\n"
                "  pip install faiss-cpu sentence-transformers\n"
                "  python 05b_build_rag_index.py"
            )

        if not os.path.exists(self.INDEX_PATH):
            raise FileNotFoundError(
                f"{self.INDEX_PATH} not found. Run: python 05b_build_rag_index.py"
            )

        print("  [RAG] Loading FAISS index and embedding model...")
        self._index  = faiss.read_index(self.INDEX_PATH)
        self._model  = SentenceTransformer(self.EMBED_MODEL)
        with open(self.CHUNKS_PATH, encoding="utf-8") as f:
            self._chunks = json.load(f)
        print(f"  [RAG] Index loaded: {self._index.ntotal} vectors | "
              f"{len(self._chunks)} chunks")

    def retrieve(self, top_k: int = 5,
                 year_filter: int | None = None) -> tuple[str, list[str]]:
        """
        Run all 5 deterministic queries, collect top_k results each,
        deduplicate, optionally filter by policy year, and return:
          - context_text  : assembled prompt context (with citation headers)
          - policy_codes  : unique list of retrieved policy codes (e.g. ["POL-01", "POL-04"])
        """
        query_embeddings = self._model.encode(
            self.FIXED_QUERIES, normalize_embeddings=True
        ).astype(np.float32)

        seen_ids  = set()
        retrieved = []

        for q_emb in query_embeddings:
            D, I = self._index.search(q_emb.reshape(1, -1), top_k)
            for idx in I[0]:
                if idx < 0 or idx >= len(self._chunks):
                    continue
                chunk = self._chunks[idx]
                # Year filter: None = no filter (mixed); integer = exact year match
                if year_filter is not None and chunk["year"] != year_filter:
                    continue
                if chunk["chunk_id"] not in seen_ids:
                    seen_ids.add(chunk["chunk_id"])
                    retrieved.append(chunk)

        # Assemble context with section separators
        context_text = "\n\n---\n\n".join(c["text"] for c in retrieved)

        # Unique, sorted policy codes for reporting
        policy_codes = sorted(set(c["policy_code"] for c in retrieved))

        return context_text, policy_codes


# ==============================================================================
# 3. STATE CACHE  (resumable, append-only JSONL)
# ==============================================================================

class StateCache:
    """
    Append-only JSON Lines cache at data/experiments_cache.jsonl.
    Keyed by (applicant_code, exp_id) — skips already-completed calls.
    Safe to interrupt at any point; resume picks up from last written record.
    """

    CACHE_PATH = "data/experiments_cache.jsonl"

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._done: dict[tuple, dict] = {}
        if os.path.exists(self.CACHE_PATH):
            with open(self.CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = (rec["applicant_code"], rec["exp_id"])
                        self._done[key] = rec
                    except json.JSONDecodeError:
                        pass
        print(f"  [Cache] {len(self._done)} completed records loaded from cache.")

    def is_done(self, applicant_code: str, exp_id: str) -> bool:
        return (applicant_code, exp_id) in self._done

    def get(self, applicant_code: str, exp_id: str) -> dict | None:
        return self._done.get((applicant_code, exp_id))

    def append(self, record: dict):
        """Write a single result to disk immediately (crash-safe)."""
        key = (record["applicant_code"], record["exp_id"])
        self._done[key] = record
        with open(self.CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==============================================================================
# 4. PROMPTS
# ==============================================================================

# Master Decision Matrix — hardcoded into every system prompt (Requirement #4)
# RAG is ONLY used to retrieve specific policy thresholds, NOT this rubric.
MASTER_DECISION_MATRIX = """
MASTER DECISION MATRIX (FROZEN — do not modify):
  | Findings                                  | Decision                  |
  |-------------------------------------------|---------------------------|
  | 2 or more HIGH / HIGH CONCERN findings    | DECLINE                   |
  | Exactly 1 HIGH  OR  any REVIEW findings   | APPROVE WITH CONDITIONS   |
  | All dimensions STANDARD                   | APPROVE                   |
"""

OUTPUT_SCHEMA = """{
  "applicant_id": "<applicant_id>",
  "reasoning_summary": "<MANDATORY: 1-sentence explanation citing the key policy measure(s)>",
  "recommendation": "<APPROVE|APPROVE_WITH_CONDITIONS|DECLINE>",
  "material_exceptions_count": <integer>
}"""

# ── Prompt A: Standard ────────────────────────────────────────────────────────
SYSTEM_PROMPT_STANDARD = f"""You are a professional credit risk analyst evaluating consumer loan applications.
Apply the provided policy thresholds to each of the 5 policy dimensions and reach a decision.
{MASTER_DECISION_MATRIX}
Return your assessment as valid JSON matching the schema exactly."""

USER_PROMPT_BASELINE = """Review the following consumer loan application and provide a structured credit assessment.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code:   {applicant_id}
Annual Income:    {annual_income}
Loan Requested:   {loan_requested}
Goods Price:      {goods_price}
Monthly Annuity:  {monthly_annuity}

=== DERIVED POLICY MEASURES ===
PTI (Payment-to-Income):   {pti}%
CTI (Credit-to-Income):    {cti}x
LGV (Loan-to-Goods-Value): {lgv}%
Employment Tenure:         {employment_years} years
Monthly Inquiries: {inq_mon} | Quarterly: {inq_qrt} | Annual: {inq_year}

=== FROZEN PD SCORE ===
Predicted PD: {pd_score}

=== RETRIEVED POLICY CONTEXT ===
{policy_context}

Return ONLY valid JSON:
""" + OUTPUT_SCHEMA

# ── Prompt B: Conservative ────────────────────────────────────────────────────
SYSTEM_PROMPT_CONSERVATIVE = f"""You are a highly cautious, risk-averse credit risk analyst.
Your primary obligation is to protect the institution from credit loss. When in doubt, decline or escalate.
Apply the provided policy thresholds strictly.
{MASTER_DECISION_MATRIX}
Return your assessment as valid JSON matching the schema exactly."""

USER_PROMPT_CONSERVATIVE = """Perform a strict, downside-risk focused assessment for this consumer loan application.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code:   {applicant_id}
Annual Income:    {annual_income}
Loan Requested:   {loan_requested}
Goods Price:      {goods_price}
Monthly Annuity:  {monthly_annuity}

=== DERIVED POLICY MEASURES ===
PTI (Payment-to-Income):   {pti}%
CTI (Credit-to-Income):    {cti}x
LGV (Loan-to-Goods-Value): {lgv}%
Employment Tenure:         {employment_years} years
Monthly Inquiries: {inq_mon} | Quarterly: {inq_qrt} | Annual: {inq_year}

=== FROZEN PD SCORE ===
Predicted PD: {pd_score}

=== RETRIEVED POLICY CONTEXT ===
{policy_context}

Evaluate strictly. Return ONLY valid JSON:
""" + OUTPUT_SCHEMA

# ── Prompt C: Chain-of-Thought + Mandatory Citation (EXP-002-P3) ──────────────
SYSTEM_PROMPT_COT = f"""You are a meticulous credit risk analyst. You MUST reason step-by-step.

Follow this exact evaluation procedure:
  STEP 1 — For each of the 5 policy measures (PTI, CTI, LGV, PD, Borrower Stability),
            find the relevant threshold in the Retrieved Policy Context below.
            Quote the exact threshold and cite the policy source (e.g. "Per POL-01 2026: PTI > 30% = High").
  STEP 2 — Classify each measure: Standard / Review (Elevated) / High (High Concern).
  STEP 3 — Count: High findings = X, Review findings = Y.
  STEP 4 — Apply the Master Decision Matrix below.
  STEP 5 — State your final recommendation.

{MASTER_DECISION_MATRIX}
Return your assessment as valid JSON matching the schema exactly."""

USER_PROMPT_COT = """Perform a step-by-step, citation-backed credit risk assessment.

=== APPLICANT FINANCIAL SUMMARY ===
Applicant Code:   {applicant_id}
Annual Income:    {annual_income}
Loan Requested:   {loan_requested}
Goods Price:      {goods_price}
Monthly Annuity:  {monthly_annuity}

=== DERIVED POLICY MEASURES ===
PTI (Payment-to-Income):   {pti}%
CTI (Credit-to-Income):    {cti}x
LGV (Loan-to-Goods-Value): {lgv}%
Employment Tenure:         {employment_years} years
Monthly Inquiries: {inq_mon} | Quarterly: {inq_qrt} | Annual: {inq_year}

=== FROZEN PD SCORE ===
Predicted PD: {pd_score}

=== RETRIEVED POLICY CONTEXT ===
{policy_context}

Follow the 5-step procedure in your system instructions. Return ONLY valid JSON:
""" + OUTPUT_SCHEMA


# ==============================================================================
# 5. EXPERIMENT REGISTRY
# ==============================================================================
# Each entry defines one experimental condition.
# top_k      : number of policy chunks to retrieve per FAISS query
# year_filter: None = all years (mixed KB for EXP-004); 2026 = current policies only
# adapter    : "A" = primary model, "B" = secondary model

EXPERIMENTS = [
    {
        "exp_id":       "EXP-001",
        "what_changed": "Baseline",
        "sys_prompt":   SYSTEM_PROMPT_STANDARD,
        "user_prompt":  USER_PROMPT_BASELINE,
        "top_k":        5,
        "year_filter":  2026,
        "adapter":      "A",
    },
    {
        "exp_id":       "EXP-002-P2",
        "what_changed": "Prompt: Conservative",
        "sys_prompt":   SYSTEM_PROMPT_CONSERVATIVE,
        "user_prompt":  USER_PROMPT_CONSERVATIVE,
        "top_k":        5,
        "year_filter":  2026,
        "adapter":      "A",
    },
    {
        "exp_id":       "EXP-002-P3",
        "what_changed": "Prompt: Chain-of-Thought",
        "sys_prompt":   SYSTEM_PROMPT_COT,
        "user_prompt":  USER_PROMPT_COT,
        "top_k":        5,
        "year_filter":  2026,
        "adapter":      "A",
    },
    {
        "exp_id":       "EXP-003-K3",
        "what_changed": "Top-K: 3",
        "sys_prompt":   SYSTEM_PROMPT_STANDARD,
        "user_prompt":  USER_PROMPT_BASELINE,
        "top_k":        3,
        "year_filter":  2026,
        "adapter":      "A",
    },
    {
        "exp_id":       "EXP-004",
        "what_changed": "Policy Year: Mixed (2025+2026)",
        "sys_prompt":   SYSTEM_PROMPT_STANDARD,
        "user_prompt":  USER_PROMPT_BASELINE,
        "top_k":        5,
        "year_filter":  None,   # No filter — let RAG retrieve from 2025 or 2026
        "adapter":      "A",
    },
    {
        "exp_id":       "EXP-005-B",
        "what_changed": "LLM: Secondary Model",
        "sys_prompt":   SYSTEM_PROMPT_STANDARD,
        "user_prompt":  USER_PROMPT_BASELINE,
        "top_k":        5,
        "year_filter":  2026,
        "adapter":      "B",
    },
]


# ==============================================================================
# 6. JSON EXTRACTION (robust — handles truncated / broken JSON)
# ==============================================================================

def extract_result(content: str) -> tuple[str, str]:
    """
    Returns (recommendation, reasoning_summary).
    Tries json.loads first; falls back to regex on broken JSON.
    Recommendation comes AFTER reasoning in the schema so truncation
    that cuts off recommendation means the whole response is rejected.
    """
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(content)
        return (
            data.get("recommendation", "ERROR"),
            data.get("reasoning_summary", ""),
        )
    except json.JSONDecodeError:
        rec_m = re.search(r'"recommendation"\s*:\s*"([^"]+)"', content)
        rsn_m = re.search(r'"reasoning_summary"\s*:\s*"([\s\S]*?)(?:",|"\s*}|$)', content)
        rec = rec_m.group(1) if rec_m else "ERROR"
        rsn = re.sub(r'\\?["\\]*$', '', rsn_m.group(1).strip()) if rsn_m else ""
        return rec, rsn


# ==============================================================================
# 7. SINGLE EXPERIMENT RUNNER
# ==============================================================================

async def run_experiment(exp: dict, row: pd.Series,
                         adapter, retriever: RAGRetriever,
                         cache: StateCache) -> dict:
    """
    Runs one (borrower × experiment) cell.
    Returns a flat result dict ready for the final report.
    Skips if already in cache.
    """
    applicant_code = row["Applicant Code"]
    exp_id         = exp["exp_id"]

    # ── Cache check ────────────────────────────────────────────────────────────
    if cache.is_done(applicant_code, exp_id):
        return cache.get(applicant_code, exp_id)

    # ── RAG retrieval ──────────────────────────────────────────────────────────
    context_text, policy_codes = retriever.retrieve(
        top_k=exp["top_k"],
        year_filter=exp["year_filter"],
    )
    retrieved_context_str = ",".join(policy_codes)  # e.g. "POL-01,POL-02,POL-04"

    # ── Format prompt ──────────────────────────────────────────────────────────
    user_prompt = exp["user_prompt"].format(
        applicant_id    = applicant_code,
        annual_income   = row["Annual Income"],
        loan_requested  = row["Loan Requested"],
        goods_price     = row["Goods Price"],
        monthly_annuity = row["Monthly Annuity"],
        pti             = row["PTI (%)"],
        cti             = row["CTI (x)"],
        lgv             = row["LGV (%)"],
        employment_years= row["Employment Tenure (Years)"],
        inq_mon         = row["Inquiries (Month)"],
        inq_qrt         = row["Inquiries (Quarter)"],
        inq_year        = row["Inquiries (Year)"],
        pd_score        = row["Predicted PD"],
        policy_context  = context_text,
    )

    # ── LLM call with retry ────────────────────────────────────────────────────
    recommendation = "ERROR"
    reasoning      = ""

    for attempt in range(3):
        try:
            content, _ = await adapter.complete(
                exp["sys_prompt"], user_prompt,
                temperature=0.0, max_tokens=800,
            )
            rec, rsn = extract_result(content)
            if rec != "ERROR" and rsn.strip():
                recommendation = rec
                reasoning      = rsn
                break
        except Exception as e:
            logger.warning(f"{exp_id} attempt {attempt+1} failed: {e}")
        if attempt < 2:
            await asyncio.sleep(10)

    # ── Build result record ────────────────────────────────────────────────────
    result = {
        "applicant_code":     applicant_code,
        "exp_id":             exp_id,
        "what_changed":       exp["what_changed"],
        "retrieved_context":  retrieved_context_str,
        "recommendation":     recommendation,
        "reasoning_summary":  reasoning,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }

    cache.append(result)
    return result


# ==============================================================================
# 8. MAIN
# ==============================================================================

async def async_main():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or "your_google" in api_key:
        print("ERROR: Set GOOGLE_API_KEY in .env")
        return

    primary_model   = os.getenv("PRIMARY_MODEL",   "gemini-2.0-flash")
    secondary_model = os.getenv("SECONDARY_MODEL",  "gemini-2.0-flash")

    adapter_A = GoogleAdapter(api_key, primary_model)
    adapter_B = GoogleAdapter(api_key, secondary_model)
    adapters  = {"A": adapter_A, "B": adapter_B}

    # Load RAG retriever (loads FAISS + embedding model once)
    retriever = RAGRetriever()

    # Load state cache (skips already-done cells on resume)
    cache = StateCache()

    df = pd.read_csv("data/processed/03_master_fixed_dataset.csv")

    total_cells = len(df) * len(EXPERIMENTS)
    done_cells  = sum(
        1 for _, row in df.iterrows()
        for exp in EXPERIMENTS
        if cache.is_done(row["Applicant Code"], exp["exp_id"])
    )
    print(f"\nExperiment grid: {len(df)} borrowers × {len(EXPERIMENTS)} experiments = {total_cells} cells")
    print(f"Already cached:  {done_cells} | Remaining: {total_cells - done_cells}")
    print("Starting... (15s pause between calls to respect free-tier TPM limits)\n")

    all_results = []

    for idx, (_, row) in enumerate(df.iterrows()):
        applicant_code = row["Applicant Code"]
        print(f"  [{idx+1}/{len(df)}] {applicant_code}")

        for exp in EXPERIMENTS:
            result = await run_experiment(
                exp, row,
                adapter  = adapters[exp["adapter"]],
                retriever= retriever,
                cache    = cache,
            )
            all_results.append(result)
            print(f"    {exp['exp_id']:15s}: {result['recommendation']:25s} | Context: {result['retrieved_context']}")

            # Rate-limit pause (only when we actually called the API)
            if result["timestamp"]:   # fresh result, not cache hit
                await asyncio.sleep(15)

    # ── Save flat results CSV ──────────────────────────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    output_path = "data/processed/05_experiment_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nAll experiments complete! Saved to {output_path}")
    print(f"Resume cache: {StateCache.CACHE_PATH}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
