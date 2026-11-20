# V5 changelog

## Added
- Rule-based digit/qualifier mismatch audit for Bangla and common English qualifier phrases.
- Qualifier audit context supplied to both B and C; structured qualifier issues are added to C when not already raised by the LLM.
- Separate second-pass source-support assessment for C's factual issues.
- Conservative factual-edit gate: only `supported` verifier verdicts can authorize factual corrections.
- Explicit preservation checks for numbers, approximation, bounds, dates, attribution, negation, and temporal/causal relations.
- Per-system and per-round length-review diagnostics; 31 words is a review threshold, not a hard cap.
- V5 result metadata, semantic verifier call logs, failed verifier-call logs, and qualifier audit logs.
- Regression tests for qualifier loss, exact-quote versus semantic-support separation, and approval gating.

## Methodological caveats
- The semantic-support step uses the same configured LLM in a separate call. It is a screening signal, not independent ground truth.
- The qualifier detector is heuristic and primarily catches digit-based quantities. It is not a complete Bengali numerical parser.
- V5 should be smoke-tested and piloted before prompts are frozen for the held-out test set.
