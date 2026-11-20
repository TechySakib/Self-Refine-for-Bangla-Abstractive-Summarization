import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bangla_refine.preprocessing import (
    normalize_article_text,
    find_navigation_markers,
    clean_article_text,
    _find_unique_after,
)

review_path = Path("data/processed/main_cleaning_review.csv")
with open(review_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} rows from {review_path}")
