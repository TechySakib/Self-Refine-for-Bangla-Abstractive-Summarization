# Bangla Self-Refine Pipeline V5.1.1

A no-GPU, API-based research pipeline for Bengali abstractive summarization using XL-Sum Bengali. V5 keeps the controlled A/B/C comparison and adds explicit qualifier-preservation checks plus a second-pass model-based source-support screen for System C.

## Systems

- **A — Single-pass:** one initial summary.
- **B — Conventional Self-Refine:** shared initial summary, free-form feedback, targeted revision. A deterministic numerical-qualifier audit is supplied to B as an explicit review cue.
- **C — Evidence-linked Self-Refine:** the same initial summary, structured Bangla-aware feedback, exact quotation/sentence-ID validation, second-pass source-support screening, and targeted revision. Deterministic qualifier flags are also represented as structured candidate issues when the reviewer misses them.

All three systems start from the same initial summary for each article. Reference summaries are kept locally for evaluation and are never sent to the LLM.

## What's included from V5

1. **Qualifier preservation:** a conservative rule-based audit looks for digit-based source quantities with qualifiers such as `কমপক্ষে`, `প্রায়`, `সর্বোচ্চ`, `এর বেশি`, and their English equivalents. If the same number appears in the generated summary without the same qualifier family nearby, it creates a review flag.
2. **Shared qualifier cues:** System B receives the flags as explicit feedback context. System C receives them as structured candidate issues and must pass the evidence checks before factual correction.
3. **Two-stage evidence screening in C:** deterministic code checks that a quotation exists in the article and cited sentence span; a separate second-pass Gemini call then assesses whether the passage actually supports the proposed issue/correction.
4. **Conservative approval:** only a `supported` second-pass verdict permits a factual correction. `contradicted`, `insufficient`, `uncertain`, missing verdicts, and verifier failures do not authorize factual edits.
5. **Length review:** the validation p90 of 31 whitespace-separated words is treated as a review threshold, not a hard cap. Results record per-system length diagnostics and flag summaries above p90 for closer review.
6. **Temporal and attribution safeguards:** prompts explicitly check dates, causal/temporal relationships, negation, attribution, names, quantities, and strength of claims.
7. **Audit trail:** results include qualifier flags, semantic verifier outputs, approved issues, failed calls, token usage, and per-round length profiles.

## V5.1 targeted patch

V5.1 keeps the same experimental systems A, B, and C. It does not change their research definitions.

1. **Python-authoritative length review:** word counts are computed in Python using whitespace-separated tokens. A summary above 31 words triggers a clause-level review cue for B and C. The 31-word p90 is not a hard cap; summaries are not automatically truncated.
2. **No-op stopping:** if a refinement returns exactly the same cleaned summary, that system records `no_change_after_review` and stops rather than spending further calls repeating an unchanged revision.
3. **Semantic-equivalence safeguard:** the System C verifier now assesses the original claim, the proposed correction, and whether the two are semantically equivalent. A factual correction is approved only if the original claim is contradicted or missing, the proposed correction is supported, the correction is materially different (or an omission is not applicable), and the verifier explicitly returns `approve_correction`.
4. **Reference-alignment audit:** each result records advisory flags for reference numbers/dates absent from the source and very low lexical overlap. A separate script exports a CSV. These flags never automatically remove or rewrite an example.
5. **Transparent cost accounting:** cost outputs separate shared initial generation, B-only incremental calls, C-only incremental calls, per-system attributed totals, and actual total experiment usage. Do not add the three attributed system totals; each includes the shared initial call.
6. **Version metadata:** result rows record pipeline `5.1.0` and prompt `v5.1`.

### Upgrade from V5

Use a new folder for V5.1. Copy only the following from the V5 folder:

- `.env`
- `data/raw/`
- `data/processed/`

Do **not** copy V5 `results/` files. V5.1 writes new results with a new version tag. Keep the V5 folder and its outputs as an immutable record.

### V5.1 local tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover deterministic length-review cues, semantic-equivalence blocking, supported corrections, incomplete verifier output, reference-alignment flags, and shared/incremental cost accounting. These are local unit tests; they do not call Gemini.

### V5.1 first smoke test

Run one article first:

```powershell
python scripts/run_experiment.py --split pilot --limit 1 --max-rounds 3
```

Then inspect:

```powershell
python scripts/inspect_v51_results.py --split pilot
python scripts/audit_reference_alignment.py --split pilot
```

For a clean repeat, rename or remove only the V5.1 `results/pilot_results.jsonl` file before rerunning. The pipeline resumes by skipping article IDs already present.

Then run the two-article smoke test:

```powershell
python scripts/run_experiment.py --split pilot --limit 2 --max-rounds 3
python scripts/evaluate.py --split pilot
```

New cost outputs:

- `results/pilot_cost_by_article.csv`
- `results/pilot_cost_summary.csv`

The cost summary labels `SHARED_INITIAL`, `B_INCREMENTAL`, `C_INCREMENTAL`, `A_TOTAL_ATTRIBUTED`, `B_TOTAL_ATTRIBUTED`, `C_TOTAL_ATTRIBUTED`, and `TOTAL_EXPERIMENT`.

### Reference-alignment interpretation

The audit is a screening tool, not an automatic dataset-cleaning rule. A missing number/date in the source may indicate a truncated article, title/body mismatch, or an abstractive reference that uses information not visible in the supplied article. Inspect the original XL-Sum record and, where possible, the linked source before deciding how to handle it. Never silently rewrite a reference summary.

### Upgrade to V5.1.1 from V5.1

Extract V5.1.1 into a new folder. Copy only `.env`, `data/raw/`, and `data/processed/` from your V5.1 folder. Do not copy V5.1 `results/`; the cleaned experiment writes separate `pilot_cleaned_*` outputs. Keep the V5.1 folder and results unchanged as the baseline record.

## V5.1.1 preprocessing-only update

V5.1.1 keeps the V5.1 prompts and A/B/C systems unchanged. It adds conservative, deterministic article-body preprocessing only. It normalizes Unicode/whitespace, removes only a bare related-reading label at the very end when no text follows it, and sends ambiguous inline related-link blocks to a manual-review CSV. It never guesses an inline boundary or rewrites article content.

Read `PREPROCESSING_V5_1_1.md` for the full procedure. From the project root:

```powershell
# First pass: create a cleaned copy and a review CSV
python scripts/clean_dataset.py --split pilot

# Edit data/processed/pilot_cleaning_review.csv:
# remove_inline_block + exact resume_at; remove_trailing_block; or keep + review_note

# Apply your reviewed decisions
python scripts/clean_dataset.py --split pilot --apply-overrides

# This must report zero records still requiring manual review

# Audit and run the separate cleaned pilot (raw V5.1 results remain untouched)
python scripts/audit_reference_alignment.py --split pilot_cleaned
python scripts/run_experiment.py --split pilot_cleaned --max-rounds 3
python scripts/evaluate.py --split pilot_cleaned
python scripts/inspect_v51_results.py --split pilot_cleaned
```

The cleaned experiment reads `data/processed/pilot_cleaned.jsonl` and writes `results/pilot_cleaned_results.jsonl`. If you change preprocessing decisions and want to rerun from scratch, rename or remove only `results/pilot_cleaned_results.jsonl` first; the runner resumes by skipping IDs already present. The original body is preserved as `text_raw`; cleaned text is stored as `text`. Experiment outputs retain both `source_article_raw` and cleaned `source_article`, along with cleaning metadata and SHA-256 hashes. All three systems receive the same cleaned source. Prompt version remains `v5.1`; pipeline version is `5.1.1`.

To prepare the test split later, use the same process with `--split main`. Do not run the main experiment until all manual-review rows are resolved and the preprocessing policy is frozen.

## Important limitations

- The second-pass support reviewer uses the **same configured LLM**, in a separate call at temperature 0.0. It is a model-based screening signal, not an independent verifier, ground truth, or a formal entailment guarantee. Human evaluation remains necessary.
- The numerical-qualifier audit is a conservative heuristic for digit-based quantities. It will not detect every Bengali number expressed in words, every paraphrase, or every complex numerical relationship. It can also produce false positives; review the recorded flags.
- The 7–21 / 22–26 / 27–31 / 32–43 word bands are practical heuristics based on the overall validation reference-length distribution. They are not measured conditional bands by article complexity and are not mandatory quotas.
- ROUGE and BERTScore measure reference overlap/semantic similarity, not factuality.
- A positive result or Q1 publication is not guaranteed. The contribution must be established by literature review and experimental evidence.

## Requirements

- Python 3.10 or newer
- Internet connection
- Gemini API access through an account eligible under the provider's current age and account rules
- Ordinary laptop or Google Colab; no GPU or fine-tuning required

The default model is `gemini-3.1-flash-lite`. Model availability, quotas, and pricing can change. Verify the model and price in the provider dashboard before a large run. Follow provider age/account eligibility; do not bypass restrictions. If the account is not eligible, use an appropriately authorized institutional or guardian-supported arrangement.

## Install in Windows VS Code

Open the extracted `bangla_self_refine_pipeline` folder in VS Code, then open Terminal → New Terminal.

```powershell
python --version
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use a Command Prompt terminal:

```cmd
.venv\Scripts\activate.bat
```

Edit `.env` and add the API key privately. Do not paste the key into source code, notebooks, chat, or GitHub. The `.gitignore` excludes `.env`.

Set the actual provider pricing in `.env` if you want a meaningful estimated-cost calculation:

```env
INPUT_USD_PER_1M=YOUR_CURRENT_INPUT_PRICE
OUTPUT_USD_PER_1M=YOUR_CURRENT_OUTPUT_PRICE
```

Leaving these values at `0` makes the script's estimated cost show `$0.00`; that does **not** prove the calls were free.

## Dataset preparation

You do not need to download the dataset manually. Run:

```powershell
python scripts/prepare_data.py --pilot-size 20 --main-size 150 --seed 42
```

The script downloads the official XL-Sum Bengali archive and prepares:

- `data/processed/pilot.jsonl`: sample from the validation/development split.
- `data/processed/main.jsonl`: sample from the official test split.

The Bengali validation split contains 1,012 examples; the test split contains 1,012 examples. The training split is not needed for this API-only experiment.

If automatic download fails, download the official archive from:

https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/data/bengali_XLSum_v2.0.tar.bz2

Place it at `data/raw/bengali_XLSum_v2.0.tar.bz2`, then rerun the preparation command. If a partial/corrupt download exists, delete that incomplete archive before retrying.

XL-Sum is licensed CC BY-NC-SA 4.0. Cite the dataset paper and review the license before redistribution or commercial use. See `CITATION.md`.

## Run V5 tests before calling the API

```powershell
python -m unittest discover -s tests -v
```

All tests should pass. These are local regression tests and do not require an API key.

## First smoke test

Run one article first:

```powershell
python scripts/run_experiment.py --split pilot --limit 1 --max-rounds 3
```

Inspect `results/pilot_results.jsonl`. Check:

- A, B, and C share the same initial summary.
- `initial_qualifier_audit` and `final_qualifier_audit` are sensible.
- C's `evidence_valid` means exact quote/sentence presence only.
- C's `semantic_support_status` is separate from quote validity.
- Only issues with `semantic_support_status: supported` are eligible for factual revision.
- `summary_length_review` flags summaries above 31 words for review; this is not an automatic failure.
- `call_log` includes semantic-verifier calls when factual candidate issues exist.

Then run the two-article smoke test:

```powershell
python scripts/run_experiment.py --split pilot --limit 2 --max-rounds 3
```

If you already ran a V5 experiment and want a clean repeat, remove or rename the V5 `results/pilot_results.jsonl` first. The pipeline resumes by skipping article IDs already present in that file. Do not copy V4 result JSONL files into V5.

## Inspect results

Run:

```powershell
python scripts/inspect_v5_results.py --split pilot
```

This reports article count, B/C identity rate, summaries above p90, remaining qualifier flags, semantic-support verdict counts, API calls, and tokens.

## Full pilot and main experiment

Only after reviewing the smoke test:

```powershell
python scripts/run_experiment.py --split pilot
python scripts/evaluate.py --split pilot
```

Freeze prompts and the method after the validation pilot. Do not tune the prompts based on main test results. Then run:

```powershell
python scripts/run_experiment.py --split main
python scripts/evaluate.py --split main
```

Completed articles are skipped on restart. Results are appended article-by-article.

## Automatic evaluation

ROUGE uses a Unicode-aware whitespace tokenizer to retain Bengali characters:

```powershell
python scripts/evaluate.py --split main
```

Outputs:

- `results/main_automatic_metrics_by_article.csv`
- `results/main_automatic_metrics_summary.csv`
- `results/main_paired_tests.json`

Optional multilingual BERTScore (CPU; may take longer):

```powershell
pip install -r requirements-bertscore.txt
python scripts/evaluate.py --split main --bertscore --bertscore-model xlm-roberta-base
```

Record the checkpoint and package versions in the paper.

## Human evaluation

After the main run:

```powershell
python scripts/prepare_human_eval.py --n 60 --annotators 3 --seed 42
```

This creates three blinded annotation sheets and an admin-only key. Never share `annotations/admin_only/annotation_key.csv` with annotators.

Rate each summary from 1 (poor) to 5 (excellent):

- factual consistency
- informativeness
- Bangla fluency
- conciseness

After collecting completed sheets, put them back in `annotations/` with the same filenames and run:

```powershell
python scripts/analyze_human_eval.py
```

## Round-by-round manual error audit

After the main experiment:

```powershell
python scripts/prepare_error_audit.py --n 30 --seed 42
```

Review each transition against the source article. Record errors corrected, errors preserved, and errors introduced. Do not treat model-based support verdicts as human ground truth.

## Output map

| Path | Contents |
|---|---|
| `data/raw/` | Downloaded XL-Sum files |
| `data/processed/` | Pilot and main samples |
| `results/*_results.jsonl` | A/B/C summaries, feedback, verifier outputs, qualifier flags, logs |
| `results/*_automatic_metrics*.csv` | Automatic metric results |
| `results/*_paired_tests.json` | Paired statistical comparisons |
| `results/human_eval_*` | Human evaluation analysis |
| `results/error_audit_template.csv` | Manual refinement error audit |
| `tests/` | Local regression tests |

## Research reporting reminders

1. Report the exact model ID, date, temperature, prompt version, API tier, token usage, actual pricing, and failed calls.
2. Distinguish deterministic quotation matching from LLM-based support screening and human factuality judgments.
3. Report the limitations of the rule-based qualifier detector.
4. Do not claim that ROUGE/BERTScore establishes factual accuracy.
5. Use the validation split for development and the held-out test split for the final comparison.
6. The same initial summary, source article, model, and temperature are shared across A/B/C to support paired comparison.
