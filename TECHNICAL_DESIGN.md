# GenAI Runtime Risk — Technical Design

This document provides a high-level technical overview of the system architecture, the evaluation engine, and the data models. It serves as the primary technical reference for the repository.

## 1. System Architecture

The software is designed as a **CLI-driven data pipeline**. It does not use a web server (no FastAPI, no Flask) because its purpose is to run bulk research experiments in a headless environment.

### Core Components
1. **SQLite Database (`genai_runtime_risk.db`)**: 
   - Serves as the single source of truth. 
   - Stores applicants, PD scores, policy documents, prompts, and the final experiment results.
   - Built using SQLAlchemy Async ORM.
2. **GenAI Assistant (`genai_assistant.py`)**:
   - Abstraction layer that routes requests to OpenAI, Anthropic, or Google based on `.env` configuration.
   - Implements structured JSON output parsing and retry logic.
   - Includes a `MockAdapter` for testing without API costs.
3. **Data Pipeline (`data_pipeline.py`)**:
   - Trains the Logistic Regression Probability of Default (PD) model using `scikit-learn`.
   - Scores applicants and determines the *Reference Decision* (the deterministic, policy-correct outcome).
4. **Experiment Runner (`cli_experiment_runner.py`)**:
   - Iterates through defined experiments.
   - Feeds applicant data and policy context into the LLM.
   - Passes the output to the Evaluation Engine.
5. **Evaluation Engine (`evaluation_engine.py`)**:
   - Compares the LLM's JSON decision against the Reference Decision.
   - Automatically calculates Deviation Classification, Materiality, and Severity.

---

## 2. The 7 Research Experiments

The platform programmatically executes the following experiments by injecting different parameters into the `genai_assistant`:

| Code | Name | Description | Applicant Scope |
|---|---|---|---|
| **EXP-001** | Baseline Repeatability | Runs the neutral prompt 3 times per applicant to measure intra-run stability. | All (N=35) |
| **EXP-002** | Prompt Tone Variation | Swaps the prompt framing between Neutral, Risk-Analytical, and Business/Growth. | All (N=35) |
| **EXP-003** | Decision Boundary | Tests applicants who sit exactly on the 5% or 10% PD thresholds. | Boundary cases only |
| **EXP-004** | Numeric PD Format | Represents the PD score as a percentage (5.2%), decimal (0.052), or approximation (~5%). | All (N=35) |
| **EXP-005** | Contextual Completeness | Withholds the applicant's raw financial data (income/DTI), forcing reliance purely on the PD score. | All (N=35) |
| **EXP-006** | Explicit Thresholds | Explicitly reminds the LLM in the prompt that "PD > 10% MUST be declined". | All (N=35) |
| **EXP-007** | Instruction Conflict | Uses system instructions to adopt a "conservative risk officer" or "lenient sales rep" persona. | Pilot group only |

**EXP-008 (Cross-Cutting Metric):** Across all experiments, the evaluation engine checks if the LLM attempted to recalculate or override the mathematically fixed PD score provided to it.

---

## 3. Evaluation Engine Metrics

Every LLM decision is passed through the `EvaluationEngine`, which calculates the following:

### Agreement
*   `agrees_with_reference (bool)`: Does the LLM Decision exactly match the deterministic Reference Decision?
*   `decision_changed (bool)`: Did the LLM decision differ from its own EXP-001 Baseline decision?

### Severity Grading
If `agrees_with_reference` is `False`, the engine assigns a severity:
*   **CRITICAL**: The LLM APPROVED an applicant that the policy mandated must be DECLINED (massive risk exposure).
*   **HIGH**: The LLM DECLINED an applicant that the policy mandated must be APPROVED (severe customer friction / fair lending risk).
*   **MODERATE**: Policy mandated MANUAL_REVIEW, but LLM made a definitive APPROVE/DECLINE (process circumvention).
*   **LOW**: Policy mandated definitive APPROVE/DECLINE, but LLM returned MANUAL_REVIEW (inefficiency).

### LLM Parsing
*   `parse_success`: Did the LLM return valid JSON matching the exact required schema?

---

## 4. Export Mechanism

The project dumps all research data into the `results/` folder.
To keep the directory clean, **every run of the pipeline deletes the old results and generates a fresh set.**

*   `pd_results_[TIMESTAMP].csv`: The ground truth PD scores.
*   `experiment_results_[TIMESTAMP].csv`: The raw LLM responses and evaluations.
*   `research_output_[TIMESTAMP].csv`: The finalized combined dataset.
*   `research_summary_[TIMESTAMP].json`: The aggregated metrics (agreement rates, severity distribution) needed to write the final research paper.
