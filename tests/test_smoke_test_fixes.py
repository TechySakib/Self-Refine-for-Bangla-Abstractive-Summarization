import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.prompts import clean_summary, validate_evidence


class SmokeTestFixes(unittest.TestCase):
    def test_quote_spanning_two_cited_sentences_is_valid(self):
        article = "প্রথম বাক্যটি এখানে। দ্বিতীয় বাক্যটি এখানে।"
        sentence_map = {
            "S001": "প্রথম বাক্যটি এখানে।",
            "S002": "দ্বিতীয় বাক্যটি এখানে।",
        }
        feedback = {
            "actionable_issues": [{
                "issue_category": "factual inconsistency",
                "source_sentence_ids": ["S001", "S002"],
                "source_quote": article,
            }]
        }
        checked = validate_evidence(feedback, sentence_map, article)
        self.assertTrue(checked["actionable_issues"][0]["evidence_valid"])

    def test_quote_in_wrong_cited_sentence_is_rejected(self):
        article = "প্রথম বাক্যটি এখানে। দ্বিতীয় বাক্যটি এখানে।"
        sentence_map = {
            "S001": "প্রথম বাক্যটি এখানে।",
            "S002": "দ্বিতীয় বাক্যটি এখানে।",
        }
        feedback = {
            "actionable_issues": [{
                "issue_category": "factual inconsistency",
                "source_sentence_ids": ["S001"],
                "source_quote": "দ্বিতীয় বাক্যটি এখানে।",
            }]
        }
        checked = validate_evidence(feedback, sentence_map, article)
        self.assertFalse(checked["actionable_issues"][0]["evidence_valid"])

    def test_summary_label_is_removed(self):
        self.assertEqual(
            clean_summary("Revised Bangla summary:\n\nএটি সারাংশ।"),
            "এটি সারাংশ।",
        )
        self.assertEqual(
            clean_summary("বাংলা সারসংক্ষেপ:\n\nএটি সারাংশ।"),
            "এটি সারাংশ।",
        )


if __name__ == "__main__":
    unittest.main()
