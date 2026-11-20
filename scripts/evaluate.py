import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import RESULTS_DIR, INPUT_USD_PER_1M, OUTPUT_USD_PER_1M
from src.bangla_refine.metrics import read_results, evaluate_rows, paired_rouge_tests, build_cost_allocation


def main():
    parser = argparse.ArgumentParser(description="Evaluate ROUGE and optional BERTScore.")
    parser.add_argument("--split", choices=["pilot", "main", "pilot_cleaned", "main_cleaned"], default="main")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="xlm-roberta-base")
    args = parser.parse_args()

    input_path = RESULTS_DIR / f"{args.split}_results.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist. Run the experiment first.")

    results = read_results(input_path)
    by_article, summary = evaluate_rows(
        results,
        include_bertscore=args.bertscore,
        bertscore_model=args.bertscore_model,
    )
    by_path = RESULTS_DIR / f"{args.split}_automatic_metrics_by_article.csv"
    summary_path = RESULTS_DIR / f"{args.split}_automatic_metrics_summary.csv"
    tests_path = RESULTS_DIR / f"{args.split}_paired_tests.json"
    round_summary_path = RESULTS_DIR / f"{args.split}_round_metrics_summary.csv"
    cost_summary_path = RESULTS_DIR / f"{args.split}_cost_summary.csv"

    by_article.to_csv(by_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    # Round-level quality trajectory (R0 is the shared initial summary).
    round_summary = (
        by_article.groupby(["system", "round"])[["rouge1_f1", "rouge2_f1", "rougeL_f1"]]
        .agg(["mean", "std", "count"]).round(5)
    )
    round_summary.to_csv(round_summary_path, encoding="utf-8-sig")

    # Transparent cost allocation: shared initial generation is counted once
    # in TOTAL_EXPERIMENT. Per-system attributed totals include shared cost and
    # therefore must not be added together.
    cost_by_article = build_cost_allocation(
        results,
        input_usd_per_1m=INPUT_USD_PER_1M,
        output_usd_per_1m=OUTPUT_USD_PER_1M,
    )
    cost_by_article_path = RESULTS_DIR / f"{args.split}_cost_by_article.csv"
    cost_by_article.to_csv(cost_by_article_path, index=False, encoding="utf-8-sig")
    cost_summary_path = RESULTS_DIR / f"{args.split}_cost_summary.csv"
    cost_summary = (
        cost_by_article.groupby("cost_component")[
            ["api_calls", "prompt_tokens", "output_tokens", "total_tokens", "estimated_cost_usd"]
        ].agg(["mean", "sum"]).round(6)
    )
    cost_summary.to_csv(cost_summary_path, encoding="utf-8-sig")

    tests = paired_rouge_tests(by_article)
    with open(tests_path, "w", encoding="utf-8") as f:
        json.dump(tests, f, ensure_ascii=False, indent=2)

    print(summary.to_string(index=False))
    print(f"\nSaved:\n{by_path}\n{summary_path}\n{tests_path}\n{round_summary_path}\n{cost_by_article_path}\n{cost_summary_path}")


if __name__ == "__main__":
    main()
