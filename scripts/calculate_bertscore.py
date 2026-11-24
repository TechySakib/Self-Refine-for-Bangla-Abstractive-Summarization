#!/usr/bin/env python3
"""
Calculate BERTScore for Bangla Abstractive Summarization experiments.

Supports interim (e.g. 83 articles) and final (150 articles) evaluation runs,
dynamically identifying JSONL fields and writing both per-article and summary CSVs.
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "main_cleaned_results.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results"

# Candidate field names for flexible schema matching
REFERENCE_KEYS = [
    "reference_summary",
    "reference",
    "summary_reference",
    "target_summary",
    "gold_summary",
    "reference_text",
    "summary",
]

SYSTEM_KEYS = {
    "A": ["system_a_initial", "system_a_final", "system_a"],
    "B": ["system_b_final", "system_b"],
    "C": ["system_c_final", "system_c"],
}


def find_key(record, candidates, label):
    for key in candidates:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return key
    raise KeyError(
        f"Cannot identify {label}. Available keys: {list(record.keys())}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate BERTScore on abstractive summarization results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to results JSONL file (default: results/main_cleaned_results.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save output CSVs (default: results/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xlm-roberta-base",
        help="Multilingual model identifier (default: xlm-roberta-base)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for BERTScore calculation (default: 8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure your experiment results (e.g. main_cleaned_results.jsonl) are placed in the results/ directory.")
        sys.exit(1)

    # Read records from JSONL
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue

    n_records = len(records)
    if n_records == 0:
        raise ValueError(f"No valid records found in {input_file}")

    print(f"Loaded {n_records} records from {input_file.name}")

    # Identify fields from first record
    first_record = records[0]
    reference_key = find_key(first_record, REFERENCE_KEYS, "reference summary")
    system_fields = {
        name: find_key(first_record, keys, f"System {name}")
        for name, keys in SYSTEM_KEYS.items()
    }

    print("\n--- Identified Field Mapping ---")
    print(f"  Reference field: '{reference_key}'")
    for name, key in system_fields.items():
        print(f"  System {name} field: '{key}'")

    sample_ref = str(first_record.get(reference_key, ""))
    print("\n--- Reference Verification Preview ---")
    print(f"  First reference ({len(sample_ref)} chars):")
    preview = sample_ref[:150] + ("..." if len(sample_ref) > 150 else "")
    print(f"  \"{preview}\"")
    print("  (Verify that this contains an XL-Sum reference summary, not the full article body)\n")

    # Import dependencies
    try:
        import torch
        from bert_score import BERTScorer
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please install BERTScore requirements:")
        print("  pip install -r requirements-bertscore.txt")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device} | Model: {args.model}")

    scorer = BERTScorer(
        model_type=args.model,
        device=device,
        batch_size=args.batch_size,
    )

    references = [str(r.get(reference_key, "") or "") for r in records]
    article_ids = [str(r.get("id", r.get("article_id", idx))) for idx, r in enumerate(records)]

    by_article_rows = []
    summary_rows = []

    stem = input_file.stem.replace("_results", "")
    by_article_file = output_dir / f"{stem}_bertscore_{n_records}_by_article.csv"
    summary_file = output_dir / f"{stem}_bertscore_{n_records}_summary.csv"

    for sys_name, sys_key in system_fields.items():
        candidates = [str(r.get(sys_key, "") or "") for r in records]
        print(f"Scoring System {sys_name} ({len(candidates)} summaries)...")
        p_tensor, r_tensor, f1_tensor = scorer.score(candidates, references)

        p_list = [float(x) for x in p_tensor]
        r_list = [float(x) for x in r_tensor]
        f1_list = [float(x) for x in f1_tensor]

        for idx in range(n_records):
            by_article_rows.append({
                "article_id": article_ids[idx],
                "system": sys_name,
                "precision": round(p_list[idx], 4),
                "recall": round(r_list[idx], 4),
                "f1": round(f1_list[idx], 4),
                "reference_summary": references[idx],
                "generated_summary": candidates[idx],
            })

        for metric_name, values in [("precision", p_list), ("recall", r_list), ("f1", f1_list)]:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            summary_rows.append({
                "system": sys_name,
                "metric": metric_name,
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "count": n_records,
            })

    # Write per-article CSV
    with open(by_article_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["article_id", "system", "precision", "recall", "f1", "reference_summary", "generated_summary"],
        )
        writer.writeheader()
        writer.writerows(by_article_rows)

    # Write summary CSV
    with open(summary_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "metric", "mean", "std", "count"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n--- BERTScore Summary Results ---")
    print(f"{'System':<8} {'Metric':<12} {'Mean':<10} {'Std':<10} {'Count':<8}")
    print("-" * 48)
    for row in summary_rows:
        if row["metric"] == "f1":
            print(f"{row['system']:<8} {row['metric']:<12} {row['mean']:<10.4f} {row['std']:<10.4f} {row['count']:<8}")

    print(f"\nOutputs successfully saved:")
    print(f"  Per-article: {by_article_file}")
    print(f"  Summary:     {summary_file}")


if __name__ == "__main__":
    main()
