import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import RAW_DIR, PROCESSED_DIR


def read_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def summarize_lengths(values):
    values = sorted(values)
    if not values:
        return {}
    def percentile(p):
        index = round((len(values) - 1) * p)
        return values[index]
    return {
        "n": len(values),
        "min": values[0],
        "median": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": values[-1],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect XL-Sum reference-summary lengths before setting generation length."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "pilot", "main"], default="val"
    )
    args = parser.parse_args()

    if args.split in {"pilot", "main"}:
        path = PROCESSED_DIR / f"{args.split}.jsonl"
    else:
        filename = "bengali_val.jsonl" if args.split == "val" else f"bengali_{args.split}.jsonl"
        path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/prepare_data.py first.")

    rows = list(read_rows(path))
    summaries = [str(row.get("summary", "") or "").strip() for row in rows]
    char_lengths = [len(x) for x in summaries]
    whitespace_word_lengths = [len(x.split()) for x in summaries]
    sentence_lengths = [
        len([s for s in re.split(r"(?<=[।!?])\s+|(?<=[.!?])\s+", x) if s.strip()])
        for x in summaries
    ]

    print(f"Split: {args.split}")
    print(f"Rows: {len(rows)}")
    print("Reference summary characters:", summarize_lengths(char_lengths))
    print("Reference summary whitespace words:", summarize_lengths(whitespace_word_lengths))
    print("Reference summary sentences:", summarize_lengths(sentence_lengths))
    print("\nUse these observed statistics to set a realistic output-length target.")
    print("Do not assume every article should have the same number of words.")


if __name__ == "__main__":
    main()
