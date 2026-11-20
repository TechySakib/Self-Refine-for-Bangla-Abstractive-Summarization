import re
import unicodedata
import numpy as np
import pandas as pd


class BanglaWhitespaceTokenizer:
    def tokenize(self, text):
        text = unicodedata.normalize("NFC", str(text or "")).lower()
        # Preserve Unicode letters/digits, including Bengali; remove punctuation.
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
        return text.split()


def read_results(path):
    import json
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_round_outputs(result, system):
    if system == "B":
        outputs = [result.get("system_a_initial", "")]
        outputs.extend(x.get("revised_summary", "") for x in result.get("system_b_rounds", []))
        return outputs
    if system == "C":
        outputs = [result.get("system_a_initial", "")]
        outputs.extend(x.get("revised_summary", "") for x in result.get("system_c_rounds", []))
        return outputs
    return [result.get("system_a_initial", "")]


def bootstrap_ci(values, n_boot=2000, seed=42):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ]
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def evaluate_rows(results, include_bertscore=False, bertscore_model="xlm-roberta-base"):
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with: pip install -r requirements.txt") from exc
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=BanglaWhitespaceTokenizer(),
    )
    metric_rows = []
    final_preds = {
        "A_single_pass": [],
        "B_self_refine": [],
        "C_evidence_linked": [],
    }
    references = []

    for result in results:
        ref = result.get("reference_summary", "") or ""
        article_id = str(result.get("id", ""))
        systems = {
            "A_single_pass": result.get("system_a_initial", ""),
            "B_self_refine": result.get("system_b_final", ""),
            "C_evidence_linked": result.get("system_c_final", ""),
        }
        for system_name, pred in systems.items():
            scores = scorer.score(ref, pred or "")
            metric_rows.append({
                "article_id": article_id,
                "system": system_name,
                "round": "final",
                "rouge1_f1": scores["rouge1"].fmeasure,
                "rouge2_f1": scores["rouge2"].fmeasure,
                "rougeL_f1": scores["rougeL"].fmeasure,
            })
            final_preds[system_name].append(pred or "")
        references.append(ref)

        for system_letter, label in [("B", "B_self_refine"), ("C", "C_evidence_linked")]:
            outputs = get_round_outputs(result, system_letter)
            for round_no, pred in enumerate(outputs):
                scores = scorer.score(ref, pred or "")
                metric_rows.append({
                    "article_id": article_id,
                    "system": label,
                    "round": f"R{round_no}",
                    "rouge1_f1": scores["rouge1"].fmeasure,
                    "rouge2_f1": scores["rouge2"].fmeasure,
                    "rougeL_f1": scores["rougeL"].fmeasure,
                })

    by_article = pd.DataFrame(metric_rows)
    final = by_article[by_article["round"] == "final"].copy()
    summary_rows = []
    for system, group in final.groupby("system"):
        for metric in ["rouge1_f1", "rouge2_f1", "rougeL_f1"]:
            vals = group[metric].to_numpy()
            ci = bootstrap_ci(vals)
            summary_rows.append({
                "system": system,
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "bootstrap_95ci_low": ci[0],
                "bootstrap_95ci_high": ci[1],
                "n": int(len(vals)),
            })
    summary = pd.DataFrame(summary_rows)

    if include_bertscore:
        try:
            from bert_score import score as bert_score
        except ImportError as exc:
            raise RuntimeError(
                "BERTScore is optional. Install it with: "
                "pip install -r requirements-bertscore.txt"
            ) from exc

        for system_name, preds in final_preds.items():
            _, _, f1 = bert_score(
                preds,
                references,
                model_type=bertscore_model,
                device="cpu",
                batch_size=4,
                verbose=True,
                rescale_with_baseline=False,
            )
            score_map = {
                str(results[i].get("id", "")): float(f1[i])
                for i in range(len(results))
            }
            mask = (by_article["system"] == system_name) & (by_article["round"] == "final")
            by_article.loc[mask, "bertscore_f1"] = by_article.loc[mask, "article_id"].map(score_map)

    return by_article, summary



def build_cost_allocation(results, input_usd_per_1m=0.0, output_usd_per_1m=0.0):
    """Build transparent shared/incremental/attributed/total API cost rows.

    Shared A/B/C initial generation is counted once in TOTAL_EXPERIMENT.
    Per-system attributed totals intentionally include the shared initial call,
    so they must not be summed across systems.
    """
    rows = []

    def summarize_calls(article_id, label, calls):
        prompt_tokens = sum(int(call.get("prompt_tokens", 0) or 0) for call in calls)
        output_tokens = sum(int(call.get("output_tokens", 0) or 0) for call in calls)
        total_tokens = sum(
            int(call.get("total_tokens", 0) or
                (int(call.get("prompt_tokens", 0) or 0) + int(call.get("output_tokens", 0) or 0)))
            for call in calls
        )
        cost = (
            prompt_tokens * float(input_usd_per_1m) / 1_000_000
            + output_tokens * float(output_usd_per_1m) / 1_000_000
        )
        return {
            "article_id": str(article_id),
            "cost_component": label,
            "api_calls": len(calls),
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "price_configured": bool(input_usd_per_1m or output_usd_per_1m),
        }

    for result in results:
        article_id = result.get("id", "")
        calls = result.get("call_log", []) or []
        shared = [call for call in calls if call.get("system") == "A_B_C"]
        b_only = [call for call in calls if call.get("system") == "B"]
        c_only = [call for call in calls if call.get("system") == "C"]

        shared_row = summarize_calls(article_id, "SHARED_INITIAL", shared)
        b_row = summarize_calls(article_id, "B_INCREMENTAL", b_only)
        c_row = summarize_calls(article_id, "C_INCREMENTAL", c_only)
        total_row = summarize_calls(article_id, "TOTAL_EXPERIMENT", shared + b_only + c_only)

        rows.extend([
            shared_row,
            b_row,
            c_row,
            summarize_calls(article_id, "A_TOTAL_ATTRIBUTED", shared),
            summarize_calls(article_id, "B_TOTAL_ATTRIBUTED", shared + b_only),
            summarize_calls(article_id, "C_TOTAL_ATTRIBUTED", shared + c_only),
            total_row,
        ])

    return pd.DataFrame(rows)

def paired_bootstrap_difference_ci(left, right, n_boot=2000, seed=42):
    differences = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = [
        np.mean(rng.choice(differences, size=len(differences), replace=True))
        for _ in range(n_boot)
    ]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_rouge_tests(by_article):
    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests

    final = by_article[by_article["round"] == "final"]
    comparisons = [
        ("A_single_pass", "B_self_refine"),
        ("A_single_pass", "C_evidence_linked"),
        ("B_self_refine", "C_evidence_linked"),
    ]
    output = {}
    raw_p = []
    keys = []
    for metric in ["rouge1_f1", "rouge2_f1", "rougeL_f1"]:
        pivot = final.pivot(index="article_id", columns="system", values=metric)
        for left, right in comparisons:
            if left not in pivot or right not in pivot:
                continue
            pair = pivot[[left, right]].dropna()
            key = f"{metric}:{left}_vs_{right}"
            keys.append(key)
            if len(pair) < 2 or np.allclose(pair[left], pair[right]):
                p = 1.0
                statistic = 0.0
            else:
                test = wilcoxon(pair[left], pair[right], alternative="two-sided")
                p = float(test.pvalue)
                statistic = float(test.statistic)
            raw_p.append(p)
            output[key] = {
                "n": int(len(pair)),
                "mean_left": float(pair[left].mean()) if len(pair) else None,
                "mean_right": float(pair[right].mean()) if len(pair) else None,
                "mean_difference_right_minus_left": float((pair[right] - pair[left]).mean()) if len(pair) else None,
                "paired_bootstrap_95ci_difference": paired_bootstrap_difference_ci(pair[left], pair[right]),
                "wilcoxon_statistic": statistic,
                "p_value": p,
            }

    if raw_p:
        adjusted = multipletests(raw_p, method="holm")[1]
        for key, adj in zip(keys, adjusted):
            output[key]["holm_adjusted_p_value"] = float(adj)
    return output
