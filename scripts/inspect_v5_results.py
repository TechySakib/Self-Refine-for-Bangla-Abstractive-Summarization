import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def main():
    parser = argparse.ArgumentParser(description="Inspect V5 A/B/C smoke-test results.")
    parser.add_argument("--split", choices=["pilot", "main"], default="pilot")
    args = parser.parse_args()

    path = ROOT / "results" / f"{args.split}_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the V5 experiment first.")

    rows = load_jsonl(path)
    if not rows:
        print("No result rows found.")
        return

    equal_bc = 0
    over_p90 = Counter()
    qualifier_remaining = Counter()
    support_status = Counter()
    total_calls = total_tokens = prompt_tokens = output_tokens = 0
    for row in rows:
        equal_bc += row.get("system_b_final", "") == row.get("system_c_final", "")
        for system, diagnostic in row.get("summary_length_review", {}).items():
            if diagnostic.get("p90_review_required"):
                over_p90[system] += 1
        for system, flags in row.get("final_qualifier_audit", {}).items():
            qualifier_remaining[system] += len(flags or [])
        for round_data in row.get("system_c_rounds", []):
            feedback = round_data.get("feedback", {})
            for issue in feedback.get("actionable_issues", []):
                if issue.get("requires_evidence"):
                    support_status[issue.get("semantic_support_status", "missing")] += 1
        total_calls += int(row.get("api_calls", 0) or 0)
        total_tokens += int(row.get("total_tokens", 0) or 0)
        prompt_tokens += int(row.get("prompt_tokens", 0) or 0)
        output_tokens += int(row.get("output_tokens", 0) or 0)

    print(f"Split: {args.split}")
    print(f"Articles: {len(rows)}")
    print(f"B and C identical: {equal_bc}/{len(rows)} ({equal_bc / len(rows):.1%})")
    print("Summaries above validation p90 review threshold:", dict(over_p90))
    print("Remaining qualifier flags:", dict(qualifier_remaining))
    print("System C factual-issue support verdicts:", dict(support_status))
    print(f"API calls: {total_calls}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens: {total_tokens}")
    print("\nReminder: p90 is a review threshold, not a hard cap. Model support verdicts are not human ground truth.")


if __name__ == "__main__":
    main()
