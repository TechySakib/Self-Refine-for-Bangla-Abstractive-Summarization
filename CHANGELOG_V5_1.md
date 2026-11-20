# Changelog — V5.1

## Version 5.1.0

V5.1 is a targeted reliability and accounting patch over V5. It preserves the same A/B/C system definitions and the shared initial-summary design.

### Added
- Python-authoritative summary word counts and deterministic p90 review cues for Systems B and C.
- Deterministic length-review issue records, with explicit instructions that 31 words is not a hard cap.
- Stop condition when a refinement produces no change to the cleaned summary.
- Semantic-equivalence assessment for System C: original-claim status (including omitted/underqualified claims), correction support, semantic equivalence, and explicit correction decision.
- Advisory source/reference alignment audit based on missing numeric/date tokens and very low lexical content overlap.
- `scripts/audit_reference_alignment.py`.
- `scripts/inspect_v51_results.py`.
- Cost allocation outputs separating shared initial generation, B/C incremental calls, attributed per-system totals, and actual total experiment usage.
- Unit tests for all new safeguards.

### Changed
- System C only approves factual corrections when the original claim is contradicted or the information is missing, the proposed correction is supported, the correction is not merely an equivalent paraphrase, and the verifier explicitly approves it.
- Length diagnostics are based on Python counts, not LLM self-reported counts.
- Cost reporting no longer leaves ambiguity about shared-call double counting.

### Research limitations retained
- The second-pass verifier uses the same configured LLM and is not independent ground truth.
- Reference-alignment flags are advisory and do not automatically exclude samples.
- The p90 length threshold is a review trigger, not a hard maximum.
- Qualifier detection remains a conservative heuristic and does not cover every Bengali number/qualifier expression.
