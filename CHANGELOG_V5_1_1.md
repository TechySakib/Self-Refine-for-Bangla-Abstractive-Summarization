# Changelog — V5.1.1

## Scope: preprocessing-only patch

- Added deterministic Unicode and whitespace normalization for Bengali article bodies.
- Added detection for common Bengali related-reading/navigation markers.
- Added safe automatic removal for a bare trailing navigation label with no following text.
- Added a manual-review workflow for ambiguous inline related-link blocks; no guessed truncation.
- Added exact `resume_at` overrides and explicit `keep` decisions with review notes.
- Preserved raw source text in `text_raw` and added cleaning metadata, actions, and SHA-256 hashes.
- Added a guard preventing cleaned experiments from running while inline markers remain unresolved.
- Added `pilot_cleaned` and `main_cleaned` split support to experiment, evaluation, inspection, and alignment-audit scripts.
- Bumped pipeline version to `5.1.1`; prompt version remains `v5.1`.

## Explicitly unchanged

- Systems A, B, and C definitions.
- All prompts in `prompts.py`.
- Gemini model, temperature, API client, refinement loops, evidence checks, and stopping logic.
- Summary-length calibration and ROUGE evaluation methodology.
- Reference summaries.

## Validation

The new preprocessing unit tests cover normalization, no-op preservation, trailing cleanup, conservative inline review, manual removal, manual keep, ambiguous boundary rejection, marker variants, and hash logging. The full existing test suite should be run locally before any API experiment.
