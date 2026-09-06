# GenAI Runtime Risk Research Platform

## 1. What Exactly Is This Project?

This is an empirical, command-line research platform designed to measure **Generative AI credit decision drift**. 

When lenders use GenAI systems (like OpenAI, Anthropic, or Google models) to assist in underwriting or risk assessment, they expose themselves to "runtime risk." This means the LLM might produce unexpected, non-compliant, or wildly different credit decisions based on subtle changes in prompt tone, number formatting, or system persona.

**What We Are Doing Here:**
This software programmatically feeds applicant data into an LLM across 7 controlled experiments. It then evaluates the LLM's decisions against a mathematically validated, frozen "ground truth" (a Logistic Regression model trained on real Kaggle data + strict policy rules).

**Expected Output & Why It Is Helpful:**
The platform outputs a structured dataset (`CSV`) and a summary report (`JSON`) that quantifies:
- How often the LLM agrees with the strict policy baseline.
- How often the LLM changes its decision based purely on prompt phrasing.
- If the LLM ever attempts to improperly override the fixed numerical risk score.
- The severity of any non-compliant decisions.

These exact metrics are used to author a professional-grade AI safety research paper, demonstrating the exact failure modes of LLMs in regulated financial environments.

---

## 2. Setup Guide (For New Users)

If you are setting this up on your local system, follow these steps exactly.

### Prerequisites
Before cloning, ensure you have installed:
1. **VS Code** (or your preferred code editor).
2. **Python 3.11 or 3.12** (Run `python --version` in your terminal to check). 
   > ⚠️ **CRITICAL WARNING:** Do **NOT** use experimental Python versions like 3.13 or 3.14! Core AI libraries (`pydantic-core`, `tokenizers`) rely on Rust bindings (PyO3) that will fail to compile on bleeding-edge Python versions, causing massive installation errors. Stick to stable Python 3.12.

> **Note on `.gitignore`:**
> To protect sensitive data and keep the repository clean, the `.gitignore` specifically excludes:
> *   Your `.env` file (API keys).
> *   The SQLite database (`genai_runtime_risk.db`).
> *   The Kaggle dataset CSVs (`data/raw/*.csv`).
> *   The `results/` folder and any local output files.
> You will need to recreate the `.env` and download the dataset manually after cloning, as detailed in the steps below.

### Step 1: Clone the Repository
Open your terminal and run:
```bash
git clone <your-repository-url>
cd GEN-AI-RUNTIME-RISK
```

### Step 2: Set Up a Virtual Environment (Highly Recommended)
To prevent conflicts with your system Python packages, create a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Requirements
Once your virtual environment is active, install all the necessary Python libraries:
```bash
pip install -r requirements.txt
```

### Step 4: Configure the LLM (.env)
Since the `.env` file containing API keys is ignored by Git, you must create one. We have provided a template for you.
1. In the root directory, you will see a file named `.env.example`.
2. Rename this file to `.env`:
   ```bash
   # Windows
   ren .env.example .env
   
   # macOS/Linux
   mv .env.example .env
   ```
3. **Get a FREE API Key:**
   We recommend using Google Gemini as they provide a highly capable free tier for developers:
   - Go to **[Google AI Studio](https://aistudio.google.com/app/apikey)** and sign in with your Google account.
   - Click **"Get API key"** on the left menu, then click **"Create API key in new project"**.
   - Copy the long string of letters and numbers that appears.
4. Open the new `.env` file in VS Code.
   - Ensure the provider is set to `google` and the model to `gemini-1.5-flash`.
   - Paste your copied key onto the `GOOGLE_API_KEY` line.
   ```env
   GENAI_PROVIDER=google
   GENAI_MODEL_NAME=gemini-1.5-flash
   GOOGLE_API_KEY=AIzaSyPasteYourLongGoogleKeyRightHere
   ```
5. Save the file (`Ctrl + S`).

### Step 5: Add the Dataset
Because the dataset is too large for GitHub, you must download it manually.
1. Create a `data/raw/` directory if it doesn't exist:
   ```bash
   mkdir -p data/raw
   ```
2. Go to Kaggle (you will need a free account): [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
3. Click "Download All" (or just download `application_train.csv`).
4. Extract the `.zip` file and place the `application_train.csv` file exactly here:
   `GEN-AI-RUNTIME-RISK/data/raw/application_train.csv`

---

## 3. How to Run the Pipeline

You can run the pipeline step-by-step to see exactly what is happening, or you can run it all at once.

### The Step-By-Step Commands

1. **Initialize the System**
   ```bash
   python cli.py setup
   ```
   *What this does:* Initializes the SQLite database and seeds the system with the 7 experiment definitions, prompts, and applicant data.

2. **Train the Baseline Model**
   ```bash
   python cli.py train-pd
   ```
   *What this does:* Trains the Logistic Regression model on the Kaggle data, scores the 35 applicants, and saves the baseline results to the `results/` folder.

3. **Verify the Baseline**
   ```bash
   python cli.py show-pd-results
   ```
   *What this does:* Displays a table of the baseline PD scores so you can verify the ground truth before the LLM gets involved.

4. **Build the Knowledge Base**
   ```bash
   python cli.py build-rag
   ```
   *What this does:* Chunks the lending policy documents and indexes them into the ChromaDB vector store.

5. **Run the AI Experiments**
   ```bash
   python cli.py run-experiments
   ```
   *What this does:* This is the core task. It connects to the LLM via your API key, runs all 7 experiments across the applicants, evaluates the decisions, and saves the raw data to the `results/` folder.

6. **View the AI Results**
   ```bash
   python cli.py show-results
   ```
   *What this does:* Prints a summary table of the LLM's performance and any policy deviations to your terminal.

7. **Export Final Research**
   ```bash
   python cli.py export
   ```
   *What this does:* Packages the final results into a clean JSON summary format ready for your research paper.

---

### The Final "Auto" Command (Recommended)

If everything is configured correctly in your `.env` file and you just want to generate the final research data without typing 7 commands, you can run the entire pipeline from start to finish with zero manual intervention:

```bash
python cli.py run-all
```

*Note: Every time you run the pipeline, the old files in the `results/` folder are completely overwritten with the newest run, ensuring you always have a clean, single source of truth timestamped to your latest execution.*
