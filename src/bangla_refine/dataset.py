import json
import tarfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .config import DATASET_URL, RAW_DIR, PROCESSED_DIR


def download_and_extract():
    # The official XL-Sum repository describes the validation split as "dev".
    # Accept both dev and val naming so minor archive naming differences do not
    # break setup. Internally, normalize the split to bengali_val.jsonl.
    train_path = RAW_DIR / "bengali_train.jsonl"
    val_path = RAW_DIR / "bengali_val.jsonl"
    test_path = RAW_DIR / "bengali_test.jsonl"
    if all(p.exists() and p.stat().st_size > 0 for p in (train_path, val_path, test_path)):
        return

    archive_path = RAW_DIR / "bengali_XLSum_v2.0.tar.bz2"
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        print("Downloading official XL-Sum Bengali archive (about 13 MB)...")
        with requests.get(DATASET_URL, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with open(archive_path, "wb") as f:
                bar = tqdm(total=total, unit="B", unit_scale=True, desc="XL-Sum") if total else None
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        if bar:
                            bar.update(len(chunk))
                if bar:
                    bar.close()

    print("Extracting Bengali train, dev/validation, and test JSONL files...")
    extracted = set()
    aliases = {
        "bengali_train.jsonl": train_path,
        "bengali_dev.jsonl": val_path,
        "bengali_val.jsonl": val_path,
        "bengali_validation.jsonl": val_path,
        "bengali_test.jsonl": test_path,
    }
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive.getmembers():
            basename = Path(member.name).name
            target = aliases.get(basename)
            if target is None or not member.isfile():
                continue
            # If both dev and val aliases are present, keep the first non-empty copy.
            if target.exists() and target.stat().st_size > 0:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as out:
                out.write(source.read())
            extracted.add(target.name)

    required = (train_path, val_path, test_path)
    missing = [p.name for p in required if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(
            f"Could not find expected Bengali splits in the archive: {missing}. "
            "The official dataset may have changed its internal filenames. "
            "Inspect the archive contents and update dataset.py if necessary."
        )


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return pd.DataFrame(rows)


def stratified_length_sample(df, n, seed=42):
    if n > len(df):
        raise ValueError(f"Requested {n} rows but only {len(df)} are available.")
    if n == len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    work = df.copy()
    lengths = work["text"].fillna("").astype(str).str.len()
    q = min(4, n, max(1, lengths.nunique()))
    if q <= 1:
        return work.sample(n=n, random_state=seed).reset_index(drop=True)

    work["_length_bin"] = pd.qcut(
        lengths.rank(method="first"), q=q, labels=False, duplicates="drop"
    )
    bins = sorted(work["_length_bin"].dropna().unique())
    base, remainder = divmod(n, len(bins))
    selected = []
    selected_indices = set()

    for i, bin_id in enumerate(bins):
        group = work[work["_length_bin"] == bin_id]
        take = min(len(group), base + (1 if i < remainder else 0))
        part = group.sample(n=take, random_state=seed + i)
        selected.append(part)
        selected_indices.update(part.index.tolist())

    result = pd.concat(selected) if selected else work.iloc[0:0]
    if len(result) < n:
        remaining = work.loc[~work.index.isin(selected_indices)]
        extra = remaining.sample(n=n - len(result), random_state=seed + 999)
        result = pd.concat([result, extra])

    return (
        result.drop(columns=["_length_bin"], errors="ignore")
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_samples(pilot_size=20, main_size=150, seed=42, force=False):
    download_and_extract()

    pilot_path = PROCESSED_DIR / "pilot.jsonl"
    main_path = PROCESSED_DIR / "main.jsonl"
    if not force and pilot_path.exists() and main_path.exists():
        print("Prepared samples already exist. Use --force to resample.")
        return

    validation = read_jsonl(RAW_DIR / "bengali_val.jsonl")
    test = read_jsonl(RAW_DIR / "bengali_test.jsonl")

    for split_name, frame in [("validation", validation), ("test", test)]:
        required = {"id", "title", "text", "summary"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{split_name} split is missing required columns: {sorted(missing)}")
        frame.dropna(subset=["id", "text", "summary"], inplace=True)
        frame["id"] = frame["id"].astype(str)
        frame["text"] = frame["text"].astype(str)
        frame["summary"] = frame["summary"].astype(str)
        frame.drop_duplicates(subset=["id"], inplace=True)
        frame.reset_index(drop=True, inplace=True)

    pilot = stratified_length_sample(validation, pilot_size, seed)
    main = stratified_length_sample(test, main_size, seed + 1)

    write_jsonl(pilot.to_dict(orient="records"), pilot_path)
    write_jsonl(main.to_dict(orient="records"), main_path)

    manifest = {
        "dataset": "csebuetnlp/xlsum",
        "language": "bengali",
        "dataset_version": "2.0.0",
        "sampling_seed": seed,
        "pilot_source_split": "validation",
        "main_source_split": "test",
        "available_validation_rows_after_basic_checks": int(len(validation)),
        "available_test_rows_after_basic_checks": int(len(test)),
        "pilot_sample_size": int(len(pilot)),
        "main_sample_size": int(len(main)),
        "pilot_ids": [str(x) for x in pilot["id"].tolist()],
        "main_ids": [str(x) for x in main["id"].tolist()],
    }
    with open(PROCESSED_DIR / "sample_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Pilot: {len(pilot)} articles from validation -> {pilot_path}")
    print(f"Main:  {len(main)} articles from test       -> {main_path}")
    print("Reference summaries are retained locally for evaluation only.")
