import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import RESULTS_DIR
from src.bangla_refine.metrics import read_results


def main():
    parser = argparse.ArgumentParser(description="Create a round-by-round error audit sheet.")
    parser.add_argument("--split", choices=["pilot", "main", "pilot_cleaned", "main_cleaned"], default="main_cleaned")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = RESULTS_DIR / f"{args.split}_results.jsonl"
    if not source.exists():
        source = RESULTS_DIR / "main_results.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found. Run the experiment first.")
    results = read_results(source)
    if args.n > len(results):
        raise ValueError(f"Requested {args.n}, but only {len(results)} results exist.")

    selected = random.Random(args.seed).sample(results, args.n)
    rows = []
    for item in selected:
        base = {
            "article_id": str(item.get("id", "")),
            "title": item.get("title", ""),
            "source_article": item.get("source_article", ""),
            "initial_summary": item.get("system_a_initial", ""),
        }
        for system, round_key, feedback_key in [
            ("B_self_refine", "system_b_rounds", "feedback"),
            ("C_evidence_linked", "system_c_rounds", "feedback"),
        ]:
            previous = item.get("system_a_initial", "")
            rounds = item.get(round_key, [])
            if not rounds:
                rows.append({
                    **base, "system": system, "round": 0,
                    "previous_summary": "", "current_summary": previous,
                    "feedback": "", "errors_corrected": "",
                    "errors_preserved": "", "errors_introduced": "", "auditor_notes": "",
                })
            for r in rounds:
                current = r.get("revised_summary", previous)
                feedback = r.get(feedback_key, "")
                if isinstance(feedback, (dict, list)):
                    feedback = json.dumps(feedback, ensure_ascii=False)
                rows.append({
                    **base,
                    "system": system,
                    "round": r.get("round", ""),
                    "previous_summary": previous,
                    "current_summary": current,
                    "feedback": feedback,
                    "errors_corrected": "",
                    "errors_preserved": "",
                    "errors_introduced": "",
                    "auditor_notes": "",
                })
                previous = current

    output = RESULTS_DIR / "error_audit_template.csv"
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Created {len(rows)} round-transition audit rows: {output}")
    print("Auditors should compare each previous/current summary against the source article.")


if __name__ == "__main__":
    main()
