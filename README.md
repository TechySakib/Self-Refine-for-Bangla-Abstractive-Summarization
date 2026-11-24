# Bangla Self-Refine Pipeline V5.1.1

A no-GPU, API-based research pipeline for Bengali abstractive summarization using XL-Sum Bengali. V5.1.1 provides a controlled, paired experimental framework comparing single-pass generation against conventional and evidence-linked iterative refinement, augmented with deterministic qualifier-preservation checks, model-based source-support screening, and conservative source preprocessing.

---

## Table of Contents

- [Overview & Research Systems](#overview--research-systems)
- [How to Run](#how-to-run)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone and Setup Environment](#2-clone-and-setup-environment)
  - [3. Configure Environment Variables](#3-configure-environment-variables)
  - [4. Run Local Unit Tests (No API Calls)](#4-run-local-unit-tests-no-api-calls)
  - [5. Dataset Preparation](#5-dataset-preparation)
  - [6. Article Preprocessing & Cleaning (V5.1.1 Workflow)](#6-article-preprocessing--cleaning-v511-workflow)
  - [7. Smoke Testing (Recommended Before Full Runs)](#7-smoke-testing-recommended-before-full-runs)
  - [8. Full Experiment Execution](#8-full-experiment-execution)
  - [9. Automated Evaluation](#9-automated-evaluation)
  - [10. Human Evaluation & Error Audit](#10-human-evaluation--error-audit)
  - [11. Running on Google Colab](#11-running-on-google-colab)
  - [Quick Command Reference](#quick-command-reference)
- [Key Features & Methodology](#key-features--methodology)
- [Project Structure & Output Map](#project-structure--output-map)
- [Important Limitations](#important-limitations)
- [Citation](#citation)

---

## Overview & Research Systems

All three systems operate on the same source articles and share the identical initial summary to enable rigorous paired comparisons. Reference summaries are kept strictly local for offline evaluation and are **never** exposed to the LLM.

- **System A — Single-pass Baseline:** Generates one initial zero-shot summary using the configured model.
- **System B — Conventional Self-Refine:** Shares the initial summary from System A, receives free-form feedback across iterative rounds, and performs targeted revisions. Deterministic numerical-qualifier flags are supplied as explicit review cues.
- **System C — Evidence-linked Self-Refine:** Shares the initial summary, generates structured Bengali-aware feedback with quotation and sentence-ID tracking, performs two-stage evidence screening (deterministic exact quote verification + second-pass semantic support assessment at temperature 0.0), evaluates semantic equivalence, and only authorizes factual edits when strictly supported.

---

## How to Run

### 1. Prerequisites

- **Python:** 3.10 or newer (tested on 3.10, 3.11, 3.12, 3.14)
- **Google Gemini API Key:** Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
- **Hardware:** Standard CPU / consumer laptop; no local GPU or model fine-tuning required.

---

### 2. Clone and Setup Environment

Clone the repository and set up a Python virtual environment:

#### macOS / Linux

```bash
# Clone the repository
git clone https://github.com/TechySakib/Self-Refine-for-Bangla-Abstractive-Summarization.git
cd Self-Refine-for-Bangla-Abstractive-Summarization

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
# Clone the repository
git clone https://github.com/TechySakib/Self-Refine-for-Bangla-Abstractive-Summarization.git
cd Self-Refine-for-Bangla-Abstractive-Summarization

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(If PowerShell execution policies block script activation, use Command Prompt: `.venv\Scripts\activate.bat`)*

---

### 3. Configure Environment Variables

Copy the example `.env` template and add your API key:

#### macOS / Linux:
```bash
cp .env.example .env
```

#### Windows:
```powershell
Copy-Item .env.example .env
```

Edit `.env` with your preferred editor and configure the settings:

```env
# Required: Your Gemini API Key (never commit this file)
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Model selection and parameters
GEMINI_MODEL=gemini-3.1-flash-lite
TEMPERATURE=0.2
MAX_ROUNDS=3
REQUEST_PAUSE_SECONDS=1.0

# Optional: Pricing per 1M tokens (for experiment cost tracking)
INPUT_USD_PER_1M=0.075
OUTPUT_USD_PER_1M=0.30
```

> **Security Note:** `.env` is ignored by `.gitignore`. Never commit or share API keys publicly.

---

### 4. Run Local Unit Tests (No API Calls)

Before executing any API requests, run the offline test suite to verify that length calibration, qualifier tracking, semantic verification blocking, and preprocessing logic work correctly on your platform:

```bash
python -m unittest discover -s tests -v
```

All unit tests run locally without consuming API quota or tokens.

---

### 5. Dataset Preparation

Download and extract the official XL-Sum Bengali dataset automatically:

```bash
python scripts/prepare_data.py --pilot-size 20 --main-size 150 --seed 42
```

This creates:
- `data/processed/pilot.jsonl`: 20 development examples sampled from the Bengali validation split.
- `data/processed/main.jsonl`: 150 evaluation examples sampled from the official Bengali test split.

*(If automatic download is blocked by network policies, download `bengali_XLSum_v2.0.tar.bz2` manually from [HuggingFace](https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/bengali_XLSum_v2.0.tar.bz2), place it in `data/raw/`, and re-run the script.)*

---

### 6. Article Preprocessing & Cleaning (V5.1.1 Workflow)

V5.1.1 implements conservative article preprocessing to normalize Bengali text and remove noisy boilerplate (e.g. `আরও পড়ুন:` related-article blocks) with full audit trails.

#### Step 6.1 — First pass: generate cleaned candidate and review sheet
```bash
python scripts/clean_dataset.py --split pilot
```
This generates:
- `data/processed/pilot_cleaned.jsonl`
- `data/processed/pilot_cleaning_review.csv` (contains ambiguous inline navigation markers)
- `data/processed/pilot_cleaning_audit.csv`

#### Step 6.2 — Resolve review items
Open `data/processed/pilot_cleaning_review.csv`. For each row requiring review, choose one decision:
- `remove_inline_block`: provide the exact `resume_at` text from the resuming article paragraph.
- `remove_trailing_block`: remove trailing text when only related links remain at the end.
- `keep`: retain text with a brief `review_note`.

*(Refer to `PREPROCESSING_V5_1_1.md` for detailed rules and examples.)*

#### Step 6.3 — Apply overrides
```bash
python scripts/clean_dataset.py --split pilot --apply-overrides
```
Ensure the terminal reports: `Records still requiring manual review: 0`.

#### Step 6.4 — Optional reference alignment audit
```bash
python scripts/audit_reference_alignment.py --split pilot_cleaned
```

---

### 7. Smoke Testing (Recommended Before Full Runs)

Run a single-article smoke test to verify API communication, pipeline flow, and cost tracking:

```bash
python scripts/run_experiment.py --split pilot_cleaned --limit 1 --max-rounds 3
```

Inspect the smoke test output:
```bash
python scripts/inspect_v51_results.py --split pilot_cleaned
```

Next, test resumability and two-article execution:
```bash
python scripts/run_experiment.py --split pilot_cleaned --limit 2 --max-rounds 3
python scripts/evaluate.py --split pilot_cleaned
```

> **Note on Resumption:** The pipeline automatically skips article IDs already recorded in `results/`. To start a clean run, remove or rename `results/pilot_cleaned_results.jsonl`.

---

### 8. Full Experiment Execution

#### Run Pilot Experiment (20 articles)
```bash
python scripts/run_experiment.py --split pilot_cleaned --max-rounds 3
python scripts/evaluate.py --split pilot_cleaned
python scripts/inspect_v51_results.py --split pilot_cleaned
```

#### Run Main Experiment (150 articles)
1. Preprocess and resolve main split:
   ```bash
   python scripts/clean_dataset.py --split main
   # Resolve any rows in data/processed/main_cleaning_review.csv if flagged
   python scripts/clean_dataset.py --split main --apply-overrides
   python scripts/audit_reference_alignment.py --split main_cleaned
   ```
2. Run experiment and evaluation:
   ```bash
   python scripts/run_experiment.py --split main_cleaned --max-rounds 3
   python scripts/evaluate.py --split main_cleaned
   python scripts/inspect_v51_results.py --split main_cleaned
   ```

---

### 9. Automated Evaluation

Evaluate Bengali summaries using Unicode-aware ROUGE:

```bash
python scripts/evaluate.py --split main_cleaned
```

Generated outputs in `results/`:
- `main_cleaned_automatic_metrics_by_article.csv`: Per-article ROUGE-1, ROUGE-2, ROUGE-L.
- `main_cleaned_automatic_metrics_summary.csv`: Aggregated means and standard deviations across Systems A, B, and C.
- `main_cleaned_paired_tests.json`: Paired statistical tests (A vs B, A vs C, B vs C).

#### Optional: Multilingual BERTScore
For semantic similarity via multilingual transformer embeddings (`xlm-roberta-base`):

```bash
pip install -r requirements-bertscore.txt
python scripts/evaluate.py --split main_cleaned --bertscore --bertscore-model xlm-roberta-base
```

---

### 10. Human Evaluation & Error Audit

#### Blinded Human Evaluation
Generate randomized, blinded scoring sheets for 3 annotators across 60 articles:

```bash
python scripts/prepare_human_eval.py --split main_cleaned --n 60 --annotators 3 --seed 42
```
- Distribute `annotations/annotator_1.csv`, `annotator_2.csv`, `annotator_3.csv`.
- Rate criteria (1 to 5): Factual Consistency, Informativeness, Bangla Fluency, Conciseness.
- **Admin Warning:** Keep `annotations/admin_only/annotation_key.csv` confidential.

Once annotators submit their completed CSV files into `annotations/`, calculate inter-annotator agreement (Krippendorff's alpha) and statistical significance:

```bash
python scripts/analyze_human_eval.py
```

#### Round-by-Round Error Audit
Prepare a structured template to inspect iterative refinement transitions:

```bash
python scripts/prepare_error_audit.py --split main_cleaned --n 30 --seed 42
```
Outputs `results/error_audit_template.csv` to record whether errors were corrected, preserved, or introduced in each round.

---

### 11. Running on Google Colab

If you prefer to run experiments in Google Colab:
- Open `colab/Bangla_Self_Refine_Pipeline_V5_1.ipynb` in Google Colab.
- Add `GEMINI_API_KEY` to Colab Secrets (`userdata.get('GEMINI_API_KEY')`).
- Follow the notebook cells for dataset downloading, execution, and evaluation.

---

### Quick Command Reference

| Step | Goal | Command |
|---|---|---|
| **Setup** | Run Unit Tests | `python -m unittest discover -s tests -v` |
| **Data** | Download & Sample XL-Sum | `python scripts/prepare_data.py --pilot-size 20 --main-size 150` |
| **Clean** | Preprocess Bengali Text | `python scripts/clean_dataset.py --split pilot` |
| **Clean** | Apply Manual Overrides | `python scripts/clean_dataset.py --split pilot --apply-overrides` |
| **Audit** | Reference Alignment Audit | `python scripts/audit_reference_alignment.py --split pilot_cleaned` |
| **Run** | 1-Article Smoke Test | `python scripts/run_experiment.py --split pilot_cleaned --limit 1` |
| **Run** | Full Pilot Experiment | `python scripts/run_experiment.py --split pilot_cleaned --max-rounds 3` |
| **Run** | Main Test Experiment | `python scripts/run_experiment.py --split main_cleaned --max-rounds 3` |
| **Inspect** | Check Lengths & Diagnostics | `python scripts/inspect_v51_results.py --split main_cleaned` |
| **Evaluate**| Compute ROUGE & Statistics | `python scripts/evaluate.py --split main_cleaned` |
| **Human** | Create Blinded Sheets | `python scripts/prepare_human_eval.py --split main_cleaned --n 60` |
| **Human** | Analyze Annotations | `python scripts/analyze_human_eval.py` |

---

## Key Features & Methodology

1. **Deterministic Qualifier Preservation:** A rule-based scanner detects numerical quantities with bounding qualifiers (`কমপক্ষে`, `প্রায়`, `সর্বোচ্চ`, `এর বেশি`, etc.) to prevent overconfident or hallucinated numbers during refinement.
2. **Two-Stage Evidence Verification (System C):**
   - Stage 1: Deterministic check ensuring the cited quotation and sentence index exist in the source article.
   - Stage 2: Second-pass LLM call (temperature 0.0) validating whether the source passage entails the proposed factual correction.
3. **Semantic Equivalence Guard:** Prevents trivial rewordings or circular paraphrasing from consuming revision rounds.
4. **Length Calibration:** Guided by empirical XL-Sum validation p90 (31 whitespace words) as a soft review threshold rather than arbitrary hard truncation.
5. **Transparent Cost Allocation:** Transparently separates shared initial calls from incremental System B and C calls in `results/*_cost_summary.csv`.

---

## Project Structure & Output Map

```text
├── colab/                      # Google Colab runnable notebooks
├── data/
│   ├── raw/                    # Downloaded XL-Sum compressed archive
│   └── processed/              # Sampled, cleaned, and review datasets
├── results/                    # Experiment JSONL, CSV metrics, and cost logs
├── scripts/
│   ├── prepare_data.py         # Dataset download and extraction
│   ├── clean_dataset.py        # Conservative Bengali preprocessing & review
│   ├── audit_reference_alignment.py # Reference lexical/date audit
│   ├── run_experiment.py       # Main A/B/C execution runner
│   ├── inspect_v51_results.py  # Results inspector and diagnostics
│   ├── evaluate.py             # Bengali ROUGE and BERTScore evaluator
│   ├── prepare_human_eval.py   # Blinded human evaluation generator
│   ├── analyze_human_eval.py   # Inter-annotator agreement & significance
│   └── prepare_error_audit.py  # Round-by-round manual audit sheet
├── src/bangla_refine/          # Core library (pipeline, llm, metrics, prompts)
├── tests/                      # Unit test suite
├── .env.example                # Environment variables template
└── requirements.txt            # Python dependencies
```

---

## Important Limitations

- **Model-Based Verifier:** System C's second-pass screening utilizes the configured LLM at zero temperature. It serves as a strong screening filter, not as an infallible ground truth. Human evaluation remains essential.
- **Rule-Based Qualifier Scanner:** Detects digit-based Bengali numerical expressions; numbers written out as words or complex relational expressions require ongoing human review.
- **Evaluation Metrics:** High ROUGE or BERTScore reflects lexical/semantic alignment with references, but does not guarantee factual consistency.

---

## Citation

If you use this pipeline or XL-Sum Bengali in your research, please cite the corresponding works:

```bibtex
@inproceedings{hasan-etal-2021-xl-sum,
    title = "{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for Evaluation and Fine-tuning",
    author = "Hasan, Tahmid  and Bhattacharjee, Abhik  and Islam, Md. Saiful  and Mubarak, Kazi  and Rahman, Sheng  and Joty, Shafiq",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    year = "2021",
    pages = "4432--4449"
}
```
See [CITATION.md](CITATION.md) for full attribution details.
