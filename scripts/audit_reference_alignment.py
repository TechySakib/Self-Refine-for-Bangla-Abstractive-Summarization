import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import PROCESSED_DIR, RESULTS_DIR
from src.bangla_refine.dataset import read_jsonl
from src.bangla_refine.alignment import audit_reference_alignment


def main():
    parser = argparse.ArgumentParser(
        description="Audit source/reference alignment for raw or preprocessed XL-Sum splits."
    )
    parser.add_argument("--split", choices=["pilot", "main", "pilot_cleaned", "main_cleaned"], default="pilot")
    args = parser.parse_args()

    input_path = PROCESSED_DIR / f"{args.split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found. Prepare the dataset first.")

    rows = read_jsonl(input_path).to_dict(orient="records")
    output = []
    for row in rows:
        audit = audit_reference_alignment(
            f"{row.get('title', '')}\n{row.get('text', '')}",
            row.get("summary", ""),
        )
        output.append({
            "id": row.get("id", ""),
            "title": row.get("title", ""),
            "status": audit["status"],
            "review_required": audit["review_required"],
            "flags": ";".join(audit["flags"]),
            "reference_numeric_tokens_missing_from_source": ";".join(
                audit["reference_numeric_tokens_missing_from_source"]
            ),
            "lexical_content_token_coverage": audit["lexical_content_token_coverage"],
            "interpretation": audit["interpretation"],
        })

    output_path = RESULTS_DIR / f"{args.split}_reference_alignment_audit.csv"
    frame = pd.DataFrame(output)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    flagged = int(frame["review_required"].sum()) if not frame.empty else 0
    print(f"Articles audited: {len(frame)}")
    print(f"Flagged for manual review: {flagged}")
    print(f"Saved: {output_path}")
    print("Flags are advisory only; no article/reference is automatically excluded.")


if __name__ == "__main__":
    main()
