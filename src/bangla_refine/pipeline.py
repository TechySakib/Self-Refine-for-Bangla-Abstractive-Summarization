import json
from tqdm import tqdm

from .config import (
    PROCESSED_DIR, RESULTS_DIR, MODEL_NAME, MAX_ROUNDS,
    INPUT_USD_PER_1M, OUTPUT_USD_PER_1M, TEMPERATURE,
)
from .dataset import read_jsonl
from .llm import GeminiTextClient
from .prompts import (
    article_prompt_text,
    initial_summary_prompt,
    conventional_feedback_prompt,
    conventional_refine_prompt,
    evidence_feedback_prompt,
    evidence_refine_prompt,
    semantic_support_prompt,
    parse_json_response,
    split_source_sentences,
    validate_evidence,
    apply_semantic_verdicts,
    clean_summary,
    summary_length_profile,
    length_review_diagnostics,
    find_qualifier_preservation_flags,
    make_qualifier_issues,
    make_length_review_issue,
)
from .alignment import audit_reference_alignment

PIPELINE_VERSION = "5.1.1"
PROMPT_VERSION = "v5.1"


def read_completed_ids(path):
    completed = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    completed.add(str(json.loads(line).get("id")))
                except Exception:
                    continue
    return completed


def _merge_rule_based_qualifier_issues(parsed_feedback, flags, sentence_map):
    """Add deterministic qualifier-loss flags not already represented by the LLM."""
    parsed = dict(parsed_feedback)
    issues = [dict(x) for x in parsed.get("actionable_issues", []) if isinstance(x, dict)]
    for candidate in make_qualifier_issues(flags, sentence_map):
        number = str(candidate.get("problem", ""))
        category = str(candidate.get("issue_category", "")).lower()
        candidate_number = None
        for flag in flags:
            if flag.get("source_quote") == candidate.get("source_quote"):
                candidate_number = str(flag.get("number", ""))
                break
        already_present = any(
            "qualifier" in str(issue.get("issue_category", "")).lower()
            and candidate_number
            and candidate_number in str(issue.get("summary_span", ""))
            for issue in issues
        )
        if not already_present:
            issues.append(candidate)
    parsed["actionable_issues"] = issues
    parsed["rule_based_qualifier_issue_count"] = sum(
        1 for issue in issues if issue.get("origin") == "deterministic_qualifier_audit"
    )
    return parsed


def _eligible_issues(checked_feedback):
    eligible = []
    for issue in checked_feedback.get("actionable_issues", []):
        if not issue.get("evidence_valid", False):
            continue
        if not issue.get("requires_evidence", False):
            eligible.append(issue)
        elif issue.get("semantic_support_approved", False):
            eligible.append(issue)
    return eligible


def run_split(split="pilot", limit=None, max_rounds=None):
    input_path = PROCESSED_DIR / f"{split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run scripts/prepare_data.py first."
        )

    rounds = MAX_ROUNDS if max_rounds is None else max_rounds
    rows = read_jsonl(input_path).to_dict(orient="records")

    if split.endswith("_cleaned"):
        missing_metadata = [
            str(row.get("id", ""))
            for row in rows
            if not isinstance(row.get("preprocessing"), dict)
        ]
        unresolved = [
            str(row.get("id", ""))
            for row in rows
            if isinstance(row.get("preprocessing"), dict)
            and row["preprocessing"].get("review_required")
        ]
        if missing_metadata:
            raise ValueError(
                f"{len(missing_metadata)} cleaned-split record(s) lack preprocessing metadata. "
                "Run scripts/clean_dataset.py first."
            )
        if unresolved:
            preview = ", ".join(unresolved[:8])
            raise ValueError(
                f"{len(unresolved)} record(s) still require manual article-body review: {preview}. "
                "Edit the cleaning review CSV and rerun scripts/clean_dataset.py --apply-overrides."
            )

    if limit is not None:
        rows = rows[:limit]

    output_path = RESULTS_DIR / f"{split}_results.jsonl"
    completed = read_completed_ids(output_path)
    client = GeminiTextClient()

    with open(output_path, "a", encoding="utf-8") as out:
        for row in tqdm(rows, desc=f"Running {split} V5.1.1"):
            article_id = str(row.get("id", ""))
            if article_id in completed:
                continue

            article = article_prompt_text(row)
            article_text = str(row.get("text", "") or "")
            article_text_raw = str(row.get("text_raw", article_text) or "")
            preprocessing_metadata = row.get("preprocessing", {
                "preprocessing_version": None,
                "status": "not_preprocessed",
                "review_required": False,
            })
            reference_summary = str(row.get("summary", "") or "")
            reference_alignment_audit = audit_reference_alignment(
                f"{row.get('title', '')}\n{article_text}",
                reference_summary,
            )
            sentence_map = split_source_sentences(article)
            call_log = []
            call_errors = []

            def call(stage, prompt, json_mode=False, system=None, round_no=None, temperature=None):
                text, meta = client.generate(
                    prompt,
                    json_mode=json_mode,
                    temperature=temperature,
                )
                meta.update({
                    "stage": stage,
                    "system": system,
                    "round": round_no,
                })
                call_log.append(meta)
                return text

            # A/B/C share exactly the same initial summary for paired comparison.
            initial_raw = call(
                "initial_generation",
                initial_summary_prompt(article),
                system="A_B_C",
                round_no=0,
            )
            initial = clean_summary(initial_raw)
            system_a = initial
            initial_qualifier_flags = find_qualifier_preservation_flags(article, initial)

            # System B: conventional unstructured Self-Refine.
            b_current = initial
            b_rounds = []
            for round_no in range(1, rounds + 1):
                b_flags = find_qualifier_preservation_flags(article, b_current)
                b_length = length_review_diagnostics(b_current)
                feedback = call(
                    "feedback",
                    conventional_feedback_prompt(article, b_current, b_flags),
                    system="B",
                    round_no=round_no,
                )
                has_model_issue = "NO_ACTIONABLE_ISSUES" not in feedback
                # Deterministic qualifier and length-review cues cannot be
                # overridden by the LLM's self-reported word count or early stop.
                if b_flags or b_length["p90_review_required"]:
                    if not has_model_issue:
                        feedback = feedback.replace("NO_ACTIONABLE_ISSUES", "").strip()
                    if b_flags:
                        feedback += (
                            "\n\n[MANDATORY RULE-BASED QUALIFIER AUDIT]\n"
                            "নিচের source-summary qualifier mismatch-গুলো যাচাই করে সঠিক হলে সংশোধন করুন:\n"
                            + json.dumps(b_flags, ensure_ascii=False, indent=2)
                        )
                    if b_length["p90_review_required"]:
                        feedback += (
                            "\n\n[MANDATORY PYTHON-MEASURED LENGTH REVIEW]\n"
                            f"Python গণনা অনুযায়ী summary-তে {b_length['whitespace_words']}টি "
                            "whitespace-separated word আছে; validation p90 review threshold 31। "
                            "এটি hard cap নয়। প্রতিটি clause অপরিহার্য কি না যাচাই করুন; "
                            "শুধু redundant/secondary অংশ থাকলে বাদ দিন। সব clause অপরিহার্য "
                            "হলে অপরিবর্তিত রাখুন।"
                        )
                    has_model_issue = True

                if not has_model_issue:
                    b_rounds.append({
                        "round": round_no,
                        "feedback": feedback,
                        "revised_summary": b_current,
                        "stopped_early": True,
                        "stop_reason": "no_actionable_issues",
                        "qualifier_flags_before": b_flags,
                        "length_review_before": b_length,
                        "revised_summary_length": summary_length_profile(b_current),
                    })
                    break

                revised = call(
                    "refinement",
                    conventional_refine_prompt(article, b_current, feedback, b_flags),
                    system="B",
                    round_no=round_no,
                )
                cleaned_revised = clean_summary(revised)
                previous_summary = b_current
                if cleaned_revised:
                    b_current = cleaned_revised
                no_change = b_current == previous_summary
                b_rounds.append({
                    "round": round_no,
                    "feedback": feedback,
                    "revised_summary": b_current,
                    "qualifier_flags_before": b_flags,
                    "qualifier_flags_after": find_qualifier_preservation_flags(article, b_current),
                    "length_review_before": b_length,
                    "length_review_after": length_review_diagnostics(b_current),
                    "length_review_triggered": bool(b_length["p90_review_required"]),
                    "revised_summary_length": summary_length_profile(b_current),
                    "stopped_early": no_change,
                    "stop_reason": "no_change_after_review" if no_change else None,
                })
                if no_change:
                    break

            # System C: structured Bangla-aware feedback + exact quote check +
            # second-pass model-based source-support screening.
            c_current = initial
            c_rounds = []
            for round_no in range(1, rounds + 1):
                c_flags = find_qualifier_preservation_flags(article, c_current)
                c_length = length_review_diagnostics(c_current)
                raw_feedback = call(
                    "evidence_feedback",
                    evidence_feedback_prompt(article, c_current, sentence_map, c_flags),
                    json_mode=True,
                    system="C",
                    round_no=round_no,
                )
                parsed = parse_json_response(raw_feedback)
                parsed = _merge_rule_based_qualifier_issues(parsed, c_flags, sentence_map)
                length_issue = make_length_review_issue(c_current)
                if length_issue is not None:
                    parsed = dict(parsed)
                    parsed_issues = [
                        dict(issue) for issue in parsed.get("actionable_issues", [])
                        if isinstance(issue, dict)
                    ]
                    if not any(
                        issue.get("origin") == "deterministic_length_review"
                        for issue in parsed_issues
                    ):
                        parsed_issues.append(length_issue)
                    parsed["actionable_issues"] = parsed_issues
                checked = validate_evidence(parsed, sentence_map, article)

                if parsed.get("parse_error") and not checked.get("actionable_issues"):
                    c_rounds.append({
                        "round": round_no,
                        "feedback_raw": raw_feedback,
                        "feedback": checked,
                        "revised_summary": c_current,
                        "stopped_early": True,
                        "stop_reason": "feedback_json_parse_error",
                        "qualifier_flags_before": c_flags,
                        "length_review_before": c_length,
                        "revised_summary_length": summary_length_profile(c_current),
                    })
                    break

                factual_valid = [
                    (index, issue)
                    for index, issue in enumerate(checked.get("actionable_issues", []))
                    if issue.get("requires_evidence") and issue.get("evidence_valid")
                ]
                verifier_raw = ""
                verifier_parsed = {"verdicts": []}
                verifier_call_failed = False

                if factual_valid:
                    try:
                        verifier_raw = call(
                            "semantic_support_verification",
                            semantic_support_prompt(article, c_current, sentence_map, factual_valid),
                            json_mode=True,
                            system="C",
                            round_no=round_no,
                            temperature=0.0,
                        )
                        verifier_parsed = parse_json_response(verifier_raw)
                    except Exception as exc:
                        verifier_call_failed = True
                        call_errors.append({
                            "stage": "semantic_support_verification",
                            "system": "C",
                            "round": round_no,
                            "error": str(exc),
                        })
                        verifier_parsed = {
                            "parse_error": True,
                            "verdicts": [],
                            "error": str(exc),
                        }
                    checked = apply_semantic_verdicts(checked, verifier_parsed)
                else:
                    # Style-only issues can be handled without source entailment review.
                    checked = apply_semantic_verdicts(checked, {"verdicts": []})

                eligible = _eligible_issues(checked)
                if not eligible:
                    issue_count = len(checked.get("actionable_issues", []))
                    if issue_count == 0:
                        stop_reason = "no_actionable_issues"
                    elif verifier_call_failed:
                        stop_reason = "semantic_verifier_call_failed"
                    elif factual_valid:
                        stop_reason = "no_semantically_approved_factual_issues"
                    else:
                        stop_reason = "no_validated_actionable_issues"
                    c_rounds.append({
                        "round": round_no,
                        "feedback_raw": raw_feedback,
                        "feedback": checked,
                        "semantic_verifier_raw": verifier_raw,
                        "semantic_verifier_result": verifier_parsed,
                        "revised_summary": c_current,
                        "eligible_issues": [],
                        "stopped_early": True,
                        "stop_reason": stop_reason,
                        "qualifier_flags_before": c_flags,
                        "length_review_before": c_length,
                        "revised_summary_length": summary_length_profile(c_current),
                    })
                    break

                refinement_feedback = dict(checked)
                refinement_feedback["actionable_issues"] = eligible
                revised = call(
                    "evidence_refinement",
                    evidence_refine_prompt(article, c_current, refinement_feedback),
                    system="C",
                    round_no=round_no,
                )
                cleaned_revised = clean_summary(revised)
                previous_summary = c_current
                if cleaned_revised:
                    c_current = cleaned_revised
                no_change = c_current == previous_summary
                c_rounds.append({
                    "round": round_no,
                    "feedback_raw": raw_feedback,
                    "feedback": checked,
                    "semantic_verifier_raw": verifier_raw,
                    "semantic_verifier_result": verifier_parsed,
                    "revised_summary": c_current,
                    "eligible_issues": eligible,
                    "qualifier_flags_before": c_flags,
                    "qualifier_flags_after": find_qualifier_preservation_flags(article, c_current),
                    "length_review_before": c_length,
                    "length_review_after": length_review_diagnostics(c_current),
                    "length_review_triggered": bool(c_length["p90_review_required"]),
                    "revised_summary_length": summary_length_profile(c_current),
                    "stopped_early": no_change,
                    "stop_reason": "no_change_after_review" if no_change else None,
                })
                if no_change:
                    break

            prompt_tokens = sum(x.get("prompt_tokens", 0) for x in call_log)
            output_tokens = sum(x.get("output_tokens", 0) for x in call_log)
            total_tokens = sum(x.get("total_tokens", 0) for x in call_log)
            estimated_cost = (
                prompt_tokens * INPUT_USD_PER_1M / 1_000_000
                + output_tokens * OUTPUT_USD_PER_1M / 1_000_000
            )

            result = {
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "id": article_id,
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "source_article": article_text,
                "source_article_raw": article_text_raw,
                "preprocessing": preprocessing_metadata,
                "reference_summary": reference_summary,
                "reference_alignment_audit": reference_alignment_audit,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "max_rounds": rounds,
                "system_a_initial": system_a,
                "system_b_final": b_current,
                "system_c_final": c_current,
                "system_b_rounds": b_rounds,
                "system_c_rounds": c_rounds,
                "initial_qualifier_audit": initial_qualifier_flags,
                "final_qualifier_audit": {
                    "system_a": find_qualifier_preservation_flags(article, system_a),
                    "system_b": find_qualifier_preservation_flags(article, b_current),
                    "system_c": find_qualifier_preservation_flags(article, c_current),
                },
                "call_log": call_log,
                "call_errors": call_errors,
                "api_calls": len(call_log),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": round(estimated_cost, 8),
                "semantic_support_policy": {
                    "method": "second-pass assessment by the same configured LLM",
                    "temperature": 0.0,
                    "approval_rule": "only support_status=supported permits factual correction",
                    "limitation": "model-based screening signal, not independent ground truth or a formal entailment guarantee",
                },
                "length_calibration": {
                    "source": "XL-Sum Bengali validation split",
                    "n": 1012,
                    "reference_whitespace_words": {
                        "min": 7, "median": 21, "p75": 26, "p90": 31, "max": 43
                    },
                    "reference_sentences": {
                        "median": 1, "p75": 2, "p90": 2, "max": 4
                    },
                    "policy": "adaptive_by_article_complexity_not_fixed_word_count",
                    "p90_is_review_threshold_not_hard_cap": True,
                },
                "summary_length_profile": {
                    "reference": summary_length_profile(row.get("summary", "")),
                    "system_a_initial": summary_length_profile(system_a),
                    "system_b_final": summary_length_profile(b_current),
                    "system_c_final": summary_length_profile(c_current),
                },
                "summary_length_review": {
                    "system_a": length_review_diagnostics(system_a),
                    "system_b": length_review_diagnostics(b_current),
                    "system_c": length_review_diagnostics(c_current),
                },
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            completed.add(article_id)

    print(f"Saved/resumed V5.1 results: {output_path}")
