import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import PROCESSED_DIR
from src.bangla_refine.dataset import read_jsonl, write_jsonl
from src.bangla_refine.preprocessing import apply_overrides_for_row, clean_article_text

REVIEW_FIELDS = [
    "id", "title", "marker_index", "marker_text", "context_before",
    "context_after", "decision", "resume_at", "review_note",
]
AUDIT_FIELDS = [
    "id", "title", "status", "review_required", "original_characters",
    "cleaned_characters", "characters_removed_after_normalization",
    "navigation_marker_count", "actions", "original_sha256", "cleaned_sha256",
    "manual_review_reason",
]


def _read_overrides(path):
    grouped = defaultdict(list)
    if not path.exists():
        return grouped
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("id"):
                continue
            if not str(row.get("decision", "") or "").strip():
                continue
            grouped[str(row["id"])].append(row)
    return grouped


def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Conservatively clean XL-Sum Bengali article bodies without changing prompts or summaries."
    )
    parser.add_argument("--split", choices=["pilot", "main"], default="pilot")
    parser.add_argument(
        "--apply-overrides", action="store_true",
        help="Apply decisions/resume_at phrases entered in the generated cleaning review CSV.",
    )
    args = parser.parse_args()

    source_path = PROCESSED_DIR / f"{args.split}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(f"{source_path} not found. Prepare the dataset first.")

    rows = read_jsonl(source_path).to_dict(orient="records")
    review_path = PROCESSED_DIR / f"{args.split}_cleaning_review.csv"
    audit_path = PROCESSED_DIR / f"{args.split}_cleaning_audit.csv"
    output_path = PROCESSED_DIR / f"{args.split}_cleaned.jsonl"

    overrides = _read_overrides(review_path) if args.apply_overrides else defaultdict(list)
    cleaned_rows = []
    review_rows = []
    audit_rows = []
    errors = []

    for row in rows:
        article_id = str(row.get("id", ""))
        title = str(row.get("title", ""))
        raw_text = row.get("text_raw", row.get("text", ""))
        try:
            if args.apply_overrides:
                cleaned_text, metadata, pending = apply_overrides_for_row(
                    raw_text, overrides.get(article_id, [])
                )
            else:
                cleaned_text, metadata, pending = clean_article_text(raw_text)
        except Exception as exc:
            errors.append({"id": article_id, "title": title, "error": str(exc)})
            continue

        out_row = dict(row)
        out_row["text_raw"] = str(raw_text or "")
        out_row["text"] = cleaned_text
        out_row["preprocessing"] = metadata
        cleaned_rows.append(out_row)

        for item in pending:
            review_rows.append({
                "id": article_id,
                "title": title,
                **item,
            })

        audit_rows.append({
            "id": article_id,
            "title": title,
            "status": metadata["status"],
            "review_required": metadata["review_required"],
            "original_characters": metadata["original_characters"],
            "cleaned_characters": metadata["cleaned_characters"],
            "characters_removed_after_normalization": metadata["characters_removed_after_normalization"],
            "navigation_marker_count": metadata["navigation_marker_count"],
            "actions": ";".join(x["action"] for x in metadata["navigation_markers"]),
            "original_sha256": metadata["original_sha256"],
            "cleaned_sha256": metadata["cleaned_sha256"],
            "manual_review_reason": metadata["manual_review_reason"] or "",
        })

    if errors:
        error_path = PROCESSED_DIR / f"{args.split}_cleaning_errors.csv"
        _write_csv(error_path, errors, ["id", "title", "error"])
        raise RuntimeError(
            f"Cleaning stopped: {len(errors)} record(s) could not be processed. "
            f"See {error_path}. No cleaned dataset was written."
        )

    write_jsonl(cleaned_rows, output_path)
    _write_csv(audit_path, audit_rows, AUDIT_FIELDS)

    # On the first pass, create the review template. On the override pass, keep
    # the user's edited decisions intact and only report remaining unresolved rows.
    if not args.apply_overrides:
        _write_csv(review_path, review_rows, REVIEW_FIELDS)

    unresolved = sum(bool(row.get("preprocessing", {}).get("review_required")) for row in cleaned_rows)
    print(f"Source records: {len(rows)}")
    print(f"Cleaned records written: {len(cleaned_rows)}")
    print(f"Records still requiring manual review: {unresolved}")
    print(f"Cleaned dataset: {output_path}")
    print(f"Cleaning audit: {audit_path}")
    if not args.apply_overrides:
        print(f"Manual review template: {review_path}")
        if review_rows:
            print("Edit the review CSV: set decision=remove_inline_block and provide an exact resume_at phrase, or decision=keep with a review_note.")
    else:
        print("Overrides applied. If unresolved records remain, edit the review CSV and run this command again.")


if __name__ == "__main__":
    main()
