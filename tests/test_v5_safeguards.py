import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.prompts import (
    SUMMARY_LENGTH_GUIDANCE,
    QUALIFIER_PRESERVATION_GUIDANCE,
    summary_length_profile,
    length_review_diagnostics,
    find_qualifier_preservation_flags,
    split_source_sentences,
    make_qualifier_issues,
    validate_evidence,
    apply_semantic_verdicts,
    initial_summary_prompt,
    conventional_feedback_prompt,
    evidence_feedback_prompt,
    semantic_support_prompt,
)


class V5SafeguardTests(unittest.TestCase):
    def test_length_policy_uses_validation_distribution_and_no_hard_quota(self):
        for value in ("১,০১২", "২১", "২৬", "৩১", "৭–৪৩"):
            self.assertIn(value, SUMMARY_LENGTH_GUIDANCE)
        self.assertIn("hard cap নয়", SUMMARY_LENGTH_GUIDANCE)

    def test_p90_is_review_threshold_not_automatic_failure(self):
        diagnostics = length_review_diagnostics(" ".join(["বাংলা"] * 35))
        self.assertTrue(diagnostics["p90_review_required"])
        self.assertFalse(diagnostics["above_observed_validation_max"])
        self.assertIn("not an automatic error", diagnostics["interpretation"])

    def test_bengali_qualifier_loss_is_flagged(self):
        source = "সংগঠনটি বলছে, ফেসবুকে কমপক্ষে ৩০০০ পেজ রয়েছে।"
        summary = "ফেসবুকে ৩০০০ পেজ রয়েছে।"
        flags = find_qualifier_preservation_flags(source, summary)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]["number"], "3000")
        self.assertEqual(flags[0]["qualifier_family"], "lower_bound")

    def test_preserved_qualifier_is_not_flagged(self):
        source = "দেশে প্রায় ৫০০টি ওয়েবসাইট রয়েছে।"
        summary = "দেশে প্রায় ৫০০টি ওয়েবসাইট রয়েছে।"
        self.assertEqual(find_qualifier_preservation_flags(source, summary), [])

    def test_rule_based_issue_has_exact_source_sentence_id(self):
        article = "ফেসবুকে কমপক্ষে ৩০০০ পেজ রয়েছে। অন্য তথ্য এখানে।"
        sentence_map = split_source_sentences(article)
        flags = find_qualifier_preservation_flags(article, "ফেসবুকে ৩০০০ পেজ রয়েছে।")
        issues = make_qualifier_issues(flags, sentence_map)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["source_sentence_ids"], ["S001"])
        checked = validate_evidence({"actionable_issues": issues}, sentence_map, article)
        self.assertTrue(checked["actionable_issues"][0]["evidence_valid"])
        self.assertEqual(checked["actionable_issues"][0]["semantic_support_status"], "pending")

    def test_exact_quote_does_not_automatically_approve_semantic_support(self):
        article = "বরিস ২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন।"
        sentence_map = split_source_sentences(article)
        feedback = {"actionable_issues": [{
            "issue_category": "factual inconsistency",
            "summary_span": "২০১৬ সাল পর্যন্ত মেয়র ছিলেন",
            "problem": "The source quote only confirms the 2008 election date.",
            "proposed_correction": "The source does not establish the 2016 endpoint.",
            "source_sentence_ids": ["S001"],
            "source_quote": "২০০৮ সালে লন্ডনের মেয়র নির্বাচিত হন",
            "suggested_action": "correct",
        }]}
        checked = validate_evidence(feedback, sentence_map, article)
        issue = checked["actionable_issues"][0]
        self.assertTrue(issue["evidence_valid"])
        self.assertFalse(issue["semantic_support_approved"])
        self.assertEqual(issue["semantic_support_status"], "pending")

    def test_only_supported_verdict_is_approved(self):
        article = "সংগঠনটি বলছে, কমপক্ষে ৩০০০ পেজ রয়েছে।"
        sentence_map = split_source_sentences(article)
        feedback = {"actionable_issues": [{
            "issue_category": "numerical qualifier omission",
            "summary_span": "৩০০০ পেজ",
            "problem": "Qualifier missing.",
            "proposed_correction": "Preserve কমপক্ষে.",
            "source_sentence_ids": ["S001"],
            "source_quote": "কমপক্ষে ৩০০০ পেজ",
            "suggested_action": "correct",
        }]}
        checked = validate_evidence(feedback, sentence_map, article)
        verified = apply_semantic_verdicts(checked, {"verdicts": [{
            "issue_index": 0,
            "original_claim_status": "not_stated",
            "correction_status": "supported",
            "semantic_equivalence": "materially_different",
            "decision": "approve_correction",
            "reason": "The source explicitly states কমপক্ষে ৩০০০.",
        }]})
        self.assertTrue(verified["actionable_issues"][0]["semantic_support_approved"])

        blocked = apply_semantic_verdicts(checked, {"verdicts": [{
            "issue_index": 0,
            "original_claim_status": "contradicted",
            "correction_status": "insufficient",
            "semantic_equivalence": "materially_different",
            "decision": "uncertain",
            "reason": "The quote does not establish the proposed correction.",
        }]})
        self.assertFalse(blocked["actionable_issues"][0]["semantic_support_approved"])

    def test_prompts_include_qualifier_and_temporal_checks(self):
        self.assertIn("কমপক্ষে ৩০০০", QUALIFIER_PRESERVATION_GUIDANCE)
        self.assertIn("temporal", initial_summary_prompt("সংবাদ"))
        self.assertIn("qualifier", conventional_feedback_prompt("সংবাদ", "সারাংশ"))
        self.assertIn("Rule-based qualifier audit flags", evidence_feedback_prompt("সংবাদ", "সারাংশ", {}))

    def test_semantic_verifier_is_explicit_about_quote_vs_support(self):
        prompt = semantic_support_prompt(
            "মূল সংবাদ।", "সারাংশ।", {"S001": "মূল সংবাদ।"},
            [(0, {"issue_category": "factual inconsistency", "source_quote": "মূল সংবাদ।"})],
        )
        self.assertIn("quote আছে কি না দেখা নয়", prompt)
        self.assertIn("insufficient", prompt)


if __name__ == "__main__":
    unittest.main()
