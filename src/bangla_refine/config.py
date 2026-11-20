import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
ANNOTATIONS_DIR = ROOT / "annotations"

DATASET_URL = (
    "https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/"
    "data/bengali_XLSum_v2.0.tar.bz2"
)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "3"))
REQUEST_PAUSE_SECONDS = float(os.getenv("REQUEST_PAUSE_SECONDS", "1.0"))
INPUT_USD_PER_1M = float(os.getenv("INPUT_USD_PER_1M", "0") or 0)
OUTPUT_USD_PER_1M = float(os.getenv("OUTPUT_USD_PER_1M", "0") or 0)

for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, ANNOTATIONS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
