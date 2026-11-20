# V4: Adaptive summary-length calibration

This release incorporates the observed XL-Sum Bengali validation statistics (1,012 references):

- Whitespace-separated words: min 7, median 21, p75 26, p90 31, max 43.
- Sentences: median 1, p75 2, p90 2, max 4.

The prompts use flexible article-complexity bands rather than a fixed word count. These bands are practical heuristics based on the overall validation distribution; the supplied statistics do not measure a conditional relationship between article complexity and reference length. Initial generation, conventional feedback/refinement, and evidence-linked feedback/refinement all receive the same guidance. The guidance explicitly prohibits padding and dropping essential facts to meet a band.

The JSONL results now include per-summary length profiles for the reference, A, B, and C, and per-round revised-summary length profiles.

Run regression tests with:

```powershell
python -m unittest discover -s tests -v
```

Existing V3 result files do not gain these fields automatically. Start a clean V4 results file or preserve V3 results separately.
