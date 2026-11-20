import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.pipeline import run_split


def main():
    parser = argparse.ArgumentParser(description="Run V5.1.1 A/B/C Bangla summarization experiment.")
    parser.add_argument("--split", choices=["pilot", "main", "pilot_cleaned", "main_cleaned"], default="pilot")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()
    run_split(args.split, args.limit, args.max_rounds)


if __name__ == "__main__":
    main()
