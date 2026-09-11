# GenAI Runtime Risk - Research Platform

## 1. What Exactly Is This Project?

This is an empirical research platform designed to measure **Generative AI credit decision drift** (also known as "Runtime Risk"). 

When financial institutions use GenAI (like OpenAI, Anthropic, or Google models) to assist in risk assessment, they expose themselves to runtime risk. This means the LLM might produce unexpected or wildly different credit decisions based purely on subtle changes in prompt tone, retrieved documents, or model versions—even when the underlying financial math (the borrower's actual data) remains completely identical.

**What We Are Doing Here:**
This software freezes the financial math for 500 real-world borrowers (extracted from the Kaggle Home Credit Default Risk dataset). It then programmatically feeds this frozen data into an LLM across 7 controlled runtime experiments, swapping out variables like the prompt wording and the policy documents to see if the AI flips its decision.

The goal is to prove whether GenAI introduces a dangerous new type of non-deterministic risk into automated credit systems.

---

## 2. Setup Guide

If you are cloning this repository to your local system, follow these steps exactly to run the 6-script pipeline.

### Step 1: Set Up the Environment

> ⚠️ **CRITICAL WARNING:** Do **NOT** use experimental Python versions like 3.13 or 3.14! Core AI and data science libraries (`numpy`, `pandas`) rely on C-extensions that frequently fail to compile on bleeding-edge Python versions (causing "DLL load failed" errors). You **must** stick to stable Python 3.11 or 3.12.

To prevent conflicts with your system Python packages, it is highly recommended to create a virtual environment specifically using Python 3.12 (or 3.11):

```bash
# Windows
py -3.12 -m venv venv
venv\Scripts\activate

# macOS/Linux
python3.12 -m venv venv
source venv/bin/activate
```

Once your virtual environment is active, install the required packages:
```bash
python -m pip install -r requirements.txt
```

### Step 2: Configure the API Key
You must provide your own Google Gemini API key to run the experiments.
1. You will see a file named `.env.example` in the root directory. Rename it to `.env`:
   ```bash
   # Windows
   ren .env.example .env
   
   # macOS/Linux
   mv .env.example .env
   ```
2. Open the new `.env` file and add your Google API key to it like this:
```text
GOOGLE_API_KEY=your_actual_api_key_here
```
*(You can get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

### Step 3: Add the Dataset
The dataset is too large to store in this repository. You must download it manually:
1. Create the raw data directory: `mkdir -p data/raw`
2. Download the `application_train.csv` file from the [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) competition.
3. Place `application_train.csv` exactly here:
   `GEN-AI-RUNTIME-RISK/data/raw/application_train.csv`

---

## 3. Running the Pipeline (Step-by-Step)

Because the architecture is strictly modular, you simply run the 6 Python scripts in numerical order.

**Command 1: Clean the Raw Data**
```bash
python 01_clean_raw_data.py
```
*(What it does: Extracts and cleans the top 500 borrowers from the Kaggle dataset.)*

**Command 2: Train and Score the AI**
```bash
python 02_train_and_score_pd.py
```
*(What it does: Trains a Logistic Regression model on the dataset and appends a deterministic `pd_score` to all 500 borrowers.)*

**Command 3: Freeze the Master Dataset**
```bash
python 03_build_master_dataset.py
```
*(What it does: Formats the financial math (Leverage, DSCR, LTV, etc.) and freezes it so it never changes during the GenAI experiments.)*

**Command 4: Generate Mathematical Ground Truth**
```bash
python 04_run_rule_based_benchmark.py
```
*(What it does: Runs traditional, deterministic Python `if/else` logic to decide if the borrower should be Approved or Declined. This is the mathematical baseline we compare the AI against.)*

**Command 5: Execute the GenAI LLM Experiments**
```bash
python 05_run_llm_experiments.py
```
*(What it does: The core of the project. It makes ~3,500 asynchronous API calls to Google Gemini, running the 7 controlled runtime experiments by swapping out prompts, policies, and models.)*

**Command 6: Generate Final Analysis Report**
```bash
python 06_generate_final_report.py
```
*(What it does: Calculates the "Decision Change %" and "Material Reversal %" for the experiments and generates the final summary tables.)*
