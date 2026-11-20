import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import krippendorff

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.config import RESULTS_DIR, ANNOTATIONS_DIR


RATING_COLUMNS = [
    "factual_consistency", "informativeness", "bangla_fluency", "conciseness"
]


def main():
    key_path = ANNOTATIONS_DIR / "admin_only" / "annotation_key.csv"
    if not key_path.exists():
        raise FileNotFoundError("Annotation key missing. Run prepare_human_eval.py first.")

    key = pd.read_csv(key_path, dtype={"blind_id": str})
    frames = []
    for path in sorted(ANNOTATIONS_DIR.glob("annotator_*.csv")):
        df = pd.read_csv(path, dtype={"blind_id": str}, encoding="utf-8-sig")
        df["annotator_id"] = path.stem
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No annotator CSV files found.")

    data = pd.concat(frames, ignore_index=True)
    for col in RATING_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.merge(key, on="blind_id", how="inner", validate="many_to_one")

    if data.empty:
        raise ValueError("No annotation rows matched the admin key. Do not change blind_id values.")

    score_summary = (
        data.groupby("system")[RATING_COLUMNS]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    score_summary.to_csv(
        RESULTS_DIR / "human_eval_summary.csv", encoding="utf-8-sig"
    )
    data.to_csv(
        RESULTS_DIR / "human_eval_scores_unblinded.csv",
        index=False,
        encoding="utf-8-sig",
    )

    agreement_rows = []
    for metric in RATING_COLUMNS:
        matrix = data.pivot_table(
            index="blind_id",
            columns="annotator_id",
            values=metric,
            aggfunc="mean",
        ).T.to_numpy(dtype=float)
        try:
            alpha = float(
                krippendorff.alpha(
                    reliability_data=matrix,
                    level_of_measurement="ordinal",
                )
            )
        except Exception:
            alpha = None
        agreement_rows.append({"metric": metric, "krippendorff_alpha_ordinal": alpha})

    pd.DataFrame(agreement_rows).to_csv(
        RESULTS_DIR / "human_eval_agreement.csv",
        index=False,
        encoding="utf-8-sig",
    )

    article_system = (
        data.groupby(["article_id", "system"])[RATING_COLUMNS]
        .mean()
        .reset_index()
    )
    comparisons = [
        ("A_single_pass", "B_self_refine"),
        ("A_single_pass", "C_evidence_linked"),
        ("B_self_refine", "C_evidence_linked"),
    ]
    tests = {}
    raw_p, keys = [], []
    for metric in RATING_COLUMNS:
        pivot = article_system.pivot(index="article_id", columns="system", values=metric)
        for left, right in comparisons:
            if left not in pivot or right not in pivot:
                continue
            pair = pivot[[left, right]].dropna()
            name = f"{metric}:{left}_vs_{right}"
            if len(pair) < 2 or np.allclose(pair[left], pair[right]):
                stat, p = 0.0, 1.0
            else:
                result = wilcoxon(pair[left], pair[right], alternative="two-sided")
                stat, p = float(result.statistic), float(result.pvalue)
            tests[name] = {
                "n_articles": int(len(pair)),
                "mean_left": float(pair[left].mean()) if len(pair) else None,
                "mean_right": float(pair[right].mean()) if len(pair) else None,
                "mean_difference_right_minus_left": float((pair[right] - pair[left]).mean()) if len(pair) else None,
                "wilcoxon_statistic": stat,
                "p_value": p,
            }
            keys.append(name)
            raw_p.append(p)

    if raw_p:
        adjusted = multipletests(raw_p, method="holm")[1]
        for name, value in zip(keys, adjusted):
            tests[name]["holm_adjusted_p_value"] = float(value)

    with open(RESULTS_DIR / "human_eval_pairwise_tests.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, ensure_ascii=False, indent=2)

    print(score_summary)
    print("\nAgreement:")
    print(pd.DataFrame(agreement_rows).to_string(index=False))
    print("\nSaved human evaluation analysis in results/.")


if __name__ == "__main__":
    main()
