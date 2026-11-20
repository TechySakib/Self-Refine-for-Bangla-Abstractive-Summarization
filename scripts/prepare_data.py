import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.dataset import prepare_samples


def main():
    parser = argparse.ArgumentParser(description="Download and sample XL-Sum Bengali.")
    parser.add_argument("--pilot-size", type=int, default=20)
    parser.add_argument("--main-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_samples(args.pilot_size, args.main_size, args.seed, args.force)


if __name__ == "__main__":
    main()
