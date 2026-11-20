# Smoke-test fixes and V5 safeguards

## Carried forward from V3/V4
1. Multi-sentence evidence quotations are checked against the ordered joined span of cited sentence IDs.
2. System C does not pass unverified factual corrections to refinement.
3. Malformed JSON is recorded as a parse failure rather than silently treated as a clean review.
4. Common generated summary labels are removed before evaluation.
5. Validation-set summary-length statistics are recorded; 31 words is a review threshold, not a hard cap.

## Added in V5
1. A rule-based numerical-qualifier audit checks digit-based quantities for possible loss of qualifiers such as `কমপক্ষে`, `প্রায়`, and `সর্বোচ্চ`.
2. The same qualifier flags are exposed to both B and C; C also records them as structured candidate issues if the LLM reviewer misses them.
3. System C performs a second-pass source-support assessment after exact quotation validation.
4. Only factual issues with a `supported` second-pass verdict can authorize factual edits.
5. Prompts explicitly check temporal/causal relationships, attribution, negation, numerical precision, and claim strength.
6. Result files preserve verifier verdicts, qualifier flags, length diagnostics, and call failures.

## Limitations
The second-pass verifier uses the same configured LLM in a separate call; it is not independent ground truth. The qualifier detector is a heuristic and does not parse every Bengali number written in words. Human review remains necessary.
