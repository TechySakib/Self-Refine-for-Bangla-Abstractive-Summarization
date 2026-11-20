# V5.1.1 — Preprocessing-only update

## Scope

V5.1.1 keeps the V5.1 research systems A, B, and C, all prompts, the Gemini model configuration, refinement logic, and evaluation definitions unchanged. The only functional update is conservative source-article preprocessing and its audit trail.

The preprocessing code does **not** use an LLM, rewrite article sentences, paraphrase text, remove article facts, alter reference summaries, or change prompts.

## What it does

1. Applies Unicode NFC normalization.
2. Removes BOM and zero-width-space artifacts, normalizes non-breaking spaces, and collapses repeated whitespace.
3. Detects common Bengali related-reading labels such as `আরও পড়ুন:`, `আরো পড়ুন:`, `আরও পড়তে পারেন:`, `আরো পড়তে পারেন:`, and `বিবিসি বাংলায় আরো পড়ুন:`.
4. Automatically removes only a bare related-reading label at the very end when no text follows it.
5. Does **not** guess where an inline related-link block ends. It preserves that text and creates a manual-review CSV.
6. After a reviewer provides an exact `resume_at` phrase, it removes only the interval from the navigation marker up to that phrase.
7. Preserves the original article text in `text_raw`, stores the cleaned text in `text`, and records SHA-256 hashes and preprocessing decisions.
8. Requires all unresolved inline markers to be reviewed before a `*_cleaned` experiment can run.

## First pass: create cleaned data and review template

Run in VS Code from the project root:

```powershell
python scripts/clean_dataset.py --split pilot
```

Outputs:

- `data/processed/pilot_cleaned.jsonl`
- `data/processed/pilot_cleaning_review.csv`
- `data/processed/pilot_cleaning_audit.csv`

The review CSV has one row per ambiguous inline marker. Open it in Excel or VS Code.

## Resolve inline markers

For each review row, choose exactly one of these decisions:

### Option A — Remove an inline related-link block

Set:

- `decision`: `remove_inline_block`
- `resume_at`: an exact, distinctive phrase copied from the first sentence of the article that resumes after the related-link headlines.

### Option B — Remove a trailing related-link block

Use this only when the marker is the final navigation marker and **all** text after it consists of related-link headlines. Set:

- `decision`: `remove_trailing_block`
- `resume_at`: leave blank.

The script removes the marker and everything after it. This is a manual decision; the script does not infer that a non-empty trailing tail is navigation.

Example:

If the article contains:

```text
আগের সংবাদ। আরও পড়ুন: সম্পর্কিত শিরোনাম এক সম্পর্কিত শিরোনাম দুই পরবর্তী মূল অনুচ্ছেদ এখানে শুরু হয়েছে।
```

Set `resume_at` to:

```text
পরবর্তী মূল অনুচ্ছেদ এখানে শুরু হয়েছে।
```

The script removes only the marker and related-link text before that phrase. It preserves the phrase and all following article content. The phrase must occur exactly once after the marker.

### Option C — Keep after manual inspection

Set:

- `decision`: `keep`
- `review_note`: a short reason why the text was retained.

Use this only if the material after the marker is actually part of the article body or the boundary cannot safely be established. The decision is recorded in the audit.

Do not guess a boundary. If uncertain, inspect the original XL-Sum record and source page before deciding.

## Apply decisions

After editing and saving `pilot_cleaning_review.csv`, run:

```powershell
python scripts/clean_dataset.py --split pilot --apply-overrides
```

Check the terminal output. The cleaned dataset must report:

```text
Records still requiring manual review: 0
```

If unresolved rows remain, edit the review CSV and rerun the command. The script refuses to start a cleaned experiment while unresolved records remain.

## Validate and run the cleaned pilot

Run the cleaned reference audit:

```powershell
python scripts/audit_reference_alignment.py --split pilot_cleaned
```

Run the experiment:

```powershell
python scripts/run_experiment.py --split pilot_cleaned --max-rounds 3
```

Evaluate:

```powershell
python scripts/evaluate.py --split pilot_cleaned
```

Inspect:

```powershell
python scripts/inspect_v51_results.py --split pilot_cleaned
```

Outputs use the `pilot_cleaned_` prefix, so raw V5.1 results are not overwritten.

## Main split

Only after the cleaned pilot has been validated and the preprocessing decisions are frozen, prepare the test split:

```powershell
python scripts/clean_dataset.py --split main
```

Review `data/processed/main_cleaning_review.csv`, enter decisions, then:

```powershell
python scripts/clean_dataset.py --split main --apply-overrides
python scripts/audit_reference_alignment.py --split main_cleaned
```

Do not start the main experiment until all review-required records have been resolved and the preprocessing policy is frozen.

## Reproducibility and interpretation

- All systems A, B, and C receive the same cleaned article text for a given article.
- `source_article` in experiment results is the cleaned source; `source_article_raw` preserves the original source.
- Each record includes original/cleaned SHA-256 hashes and the cleaning actions.
- The cleaner is intentionally conservative. Some inline navigation blocks require human boundary decisions; it does not silently truncate long text.
- Reference summaries are never changed by this preprocessing step.
- This is preprocessing, not a new summarization or refinement method. Report V5.1.1 as the preprocessing release and keep prompt version `v5.1`.
