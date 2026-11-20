import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.prompts import (
    make_length_review_issue,
    summary_length_profile,
    validate_evidence,
    apply_semantic_verdicts,
    semantic_support_prompt,
    issue_requires_evidence,
)
from src.bangla_refine.alignment import audit_reference_alignment
from src.bangla_refine.metrics import build_cost_allocation


class V51PatchTests(unittest.TestCase):
    def test_python_word_count_triggers_review_above_p90(self):
        text = " ".join(["বাংলা"] * 36)
        issue = make_length_review_issue(text)
        self.assertIsNotNone(issue)
        self.assertEqual(issue["measured_word_count"], 36)
        self.assertEqual(issue["review_threshold_words"], 31)
        self.assertFalse(issue_requires_evidence(issue))

    def test_length_review_is_not_a_hard_cap(self):
        text = " ".join(["বাংলা"] * 31)
        self.assertIsNone(make_length_review_issue(text))
        text = " ".join(["বাংলা"] * 32)
        issue = make_length_review_issue(text)
        self.assertIn("not an automatic error or hard cap", issue["problem"])

    def test_python_length_count_is_whitespace_based(self):
        text = "প্রথম বাক্য। দ্বিতীয় বাক্য!"
        self.assertEqual(summary_length_profile(text)["whitespace_words"], 4)

    def test_equivalent_paraphrase_is_not_approved(self):
        article = "দেশে প্রায় ৫০০টি ই-কমার্স ওয়েবসাইট রয়েছে।"
        sentence_map = {"S001": article}
        feedback = {"actionable_issues": [{
            "issue_category": "factual inconsistency",
            "summary_span": "প্রায় ৫০০টি",
            "problem": "The wording may need adjustment.",
            "proposed_correction": "পাঁচশটির মতো",
            "source_sentence_ids": ["S001"],
            "source_quote": "প্রায় ৫০০টি ই-কমার্স ওয়েবসাইট",
        }]}
        checked = validate_evidence(feedback, sentence_map, article)
        result = apply_semantic_verdicts(checked, {"verdicts": [{
            "issue_index": 0,
            "original_claim_status": "supported",
            "correction_status": "supported",
            "semantic_equivalence": "equivalent",
            "decision": "preserve_original",
            "reason": "দুটি expression একই আনুমানিক অর্থ প্রকাশ করে।",
        }]})
        issue = result["actionable_issues"][0]
        self.assertFalse(issue["semantic_support_approved"])
        self.assertEqual(issue["semantic_support_status"], "blocked_equivalent_or_original_supported")

    def test_genuine_supported_correction_can_be_approved(self):
        article = "বরিস ২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন।"
        sentence_map = {"S001": article}
        feedback = {"actionable_issues": [{
            "issue_category": "factual inconsistency",
            "summary_span": "২০১৬ সাল পর্যন্ত মেয়র ছিলেন",
            "problem": "The end date is not supported.",
            "proposed_correction": "২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন",
            "source_sentence_ids": ["S001"],
            "source_quote": "২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন",
        }]}
        checked = validate_evidence(feedback, sentence_map, article)
        result = apply_semantic_verdicts(checked, {"verdicts": [{
            "issue_index": 0,
            "original_claim_status": "contradicted",
            "correction_status": "supported",
            "semantic_equivalence": "materially_different",
            "decision": "approve_correction",
            "reason": "Source supports election in 2008 but not the 2016 endpoint.",
        }]})
        self.assertTrue(result["actionable_issues"][0]["semantic_support_approved"])

    def test_qualifier_loss_can_be_approved_as_not_fully_supported(self):
        article = "ফেসবুকে কমপক্ষে ৩০০০ পেজ রয়েছে।"
        sentence_map = {"S001": article}
        feedback = {"actionable_issues": [{
            "issue_category": "numerical qualifier omission",
            "summary_span": "৩০০০ পেজ",
            "problem": "The lower-bound qualifier was omitted.",
            "proposed_correction": "কমপক্ষে ৩০০০ পেজ",
            "source_sentence_ids": ["S001"],
            "source_quote": "কমপক্ষে ৩০০০ পেজ",
        }]}
        checked = validate_evidence(feedback, sentence_map, article)
        result = apply_semantic_verdicts(checked, {"verdicts": [{
            "issue_index": 0,
            "original_claim_status": "not_fully_supported",
            "correction_status": "supported",
            "semantic_equivalence": "materially_different",
            "decision": "approve_correction",
            "reason": "The source explicitly states a lower bound.",
        }]})
        self.assertTrue(result["actionable_issues"][0]["semantic_support_approved"])

    def test_incomplete_verifier_schema_blocks_approval(self):
        article = "বরিস ২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন।"
        sentence_map = {"S001": article}
        feedback = {"actionable_issues": [{
            "issue_category": "factual inconsistency",
            "summary_span": "২০১৬",
            "problem": "Unsupported date.",
            "proposed_correction": "২০০৮",
            "source_sentence_ids": ["S001"],
            "source_quote": "২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন",
        }]}
        checked = validate_evidence(feedback, sentence_map, article)
        result = apply_semantic_verdicts(checked, {"verdicts": [{
            "issue_index": 0,
            "support_status": "supported",
            "reason": "Old schema only; missing equivalence fields.",
        }]})
        self.assertFalse(result["actionable_issues"][0]["semantic_support_approved"])

    def test_alignment_audit_flags_reference_year_absent_from_article(self):
        result = audit_reference_alignment(
            "বাংলাদেশে অনলাইন বাণিজ্যের প্রসার নিয়ে প্রতিবেদন প্রকাশিত হয়েছে।",
            "২০১৫ সালকে বাংলাদেশে ই-কমার্স বর্ষ ঘোষণা করা হয়েছে।",
        )
        self.assertTrue(result["review_required"])
        self.assertIn("2015", result["reference_numeric_tokens_missing_from_source"])
        self.assertIn("reference_contains_numeric_or_date_token_absent_from_source", result["flags"])

    def test_alignment_audit_does_not_auto_exclude_or_rewrite(self):
        result = audit_reference_alignment(
            "২০১৫ সালে বাংলাদেশে ই-কমার্স নিয়ে একটি উদ্যোগ ঘোষণা করা হয়।",
            "২০১৫ সালে বাংলাদেশে ই-কমার্স উদ্যোগ ঘোষণা করা হয়।",
        )
        self.assertIn(result["status"], {"review_required", "no_obvious_mismatch_detected"})
        self.assertIn("Do not automatically remove or rewrite", result["interpretation"])

    def test_cost_allocation_counts_shared_call_once_in_total(self):
        results = [{
            "id": "x1",
            "call_log": [
                {"system": "A_B_C", "prompt_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                {"system": "B", "prompt_tokens": 200, "output_tokens": 20, "total_tokens": 220},
                {"system": "C", "prompt_tokens": 300, "output_tokens": 30, "total_tokens": 330},
            ],
        }]
        frame = build_cost_allocation(results, 1.0, 2.0)
        total = frame[frame["cost_component"] == "TOTAL_EXPERIMENT"].iloc[0]
        self.assertEqual(total["api_calls"], 3)
        self.assertEqual(total["prompt_tokens"], 600)
        self.assertEqual(total["output_tokens"], 60)
        self.assertEqual(total["total_tokens"], 660)
        b_attr = frame[frame["cost_component"] == "B_TOTAL_ATTRIBUTED"].iloc[0]
        self.assertEqual(b_attr["total_tokens"], 330)
        c_attr = frame[frame["cost_component"] == "C_TOTAL_ATTRIBUTED"].iloc[0]
        self.assertEqual(c_attr["total_tokens"], 440)

    def test_semantic_prompt_requires_equivalence_assessment(self):
        prompt = semantic_support_prompt(
            "মূল সংবাদ।", "বর্তমান সারাংশ।", {"S001": "মূল সংবাদ।"},
            [(0, {"issue_category": "factual inconsistency", "source_quote": "মূল সংবাদ।"})],
        )
        self.assertIn("semantic_equivalence", prompt)
        self.assertIn("original_claim_status", prompt)
        self.assertIn("preserve_original", prompt)


if __name__ == "__main__":
    unittest.main()
