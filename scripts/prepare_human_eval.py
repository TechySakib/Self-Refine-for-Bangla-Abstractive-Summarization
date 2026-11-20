import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import RESULTS_DIR, ANNOTATIONS_DIR
from src.bangla_refine.metrics import read_results


def main():
    parser = argparse.ArgumentParser(description="Create blinded human-evaluation sheets.")
    parser.add_argument("--split", choices=["pilot", "main", "pilot_cleaned", "main_cleaned"], default="main_cleaned")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--annotators", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_path = RESULTS_DIR / f"{args.split}_results.jsonl"
    if not results_path.exists():
        results_path = RESULTS_DIR / "main_results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found. Run the experiment before preparing human evaluation.")

    results = read_results(results_path)
    if args.n > len(results):
        raise ValueError(f"Requested {args.n} articles, but only {len(results)} results exist.")

    rng = random.Random(args.seed)
    selected = rng.sample(results, args.n)
    items, key_rows = [], []

    for result in selected:
        article_id = str(result.get("id", ""))
        system_outputs = {
            "A_single_pass": result.get("system_a_initial", ""),
            "B_self_refine": result.get("system_b_final", ""),
            "C_evidence_linked": result.get("system_c_final", ""),
        }
        for system, summary in system_outputs.items():
            blind_id = f"ITEM-{len(items) + 1:04d}-{rng.randrange(1000,9999)}"
            items.append({
                "blind_id": blind_id,
                "source_title": result.get("title", ""),
                "source_article": result.get("source_article", ""),
                "generated_summary": summary,
            })
            key_rows.append({
                "blind_id": blind_id,
                "article_id": article_id,
                "system": system,
            })

    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    admin_dir = ANNOTATIONS_DIR / "admin_only"
    admin_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key_rows).to_csv(
        admin_dir / "annotation_key.csv", index=False, encoding="utf-8-sig"
    )

    columns = [
        "blind_id", "source_title", "source_article", "generated_summary",
        "factual_consistency", "informativeness", "bangla_fluency",
        "conciseness", "error_categories", "comments"
    ]
    for annotator in range(1, args.annotators + 1):
        annotator_items = [dict(x) for x in items]
        rng.shuffle(annotator_items)
        for item in annotator_items:
            item.update({
                "factual_consistency": "",
                "informativeness": "",
                "bangla_fluency": "",
                "conciseness": "",
                "error_categories": "",
                "comments": "",
            })
        pd.DataFrame(annotator_items, columns=columns).to_csv(
            ANNOTATIONS_DIR / f"annotator_{annotator}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"Created {args.annotators} blinded sheets, {args.n * 3} summaries per annotator.")
    print("Do NOT share annotations/admin_only/annotation_key.csv with annotators.")


if __name__ == "__main__":
    main()
