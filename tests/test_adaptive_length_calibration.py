import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.prompts import (
    SUMMARY_LENGTH_GUIDANCE,
    summary_length_profile,
    initial_summary_prompt,
    conventional_feedback_prompt,
    evidence_feedback_prompt,
)


class AdaptiveLengthCalibrationTests(unittest.TestCase):
    def test_guidance_uses_observed_validation_statistics(self):
        for value in ("১,০১২", "২১", "২৬", "৩১", "৭–৪৩"):
            self.assertIn(value, SUMMARY_LENGTH_GUIDANCE)
        self.assertIn("hard cap নয়", SUMMARY_LENGTH_GUIDANCE)
        self.assertIn("৩১ শব্দের বেশি হলে", SUMMARY_LENGTH_GUIDANCE)

    def test_length_profile_counts_whitespace_words_and_sentences(self):
        profile = summary_length_profile("প্রথম বাক্য। দ্বিতীয় বাক্য!")
        self.assertEqual(profile["whitespace_words"], 4)
        self.assertEqual(profile["sentences"], 2)

    def test_generation_and_feedback_prompts_include_adaptive_guidance(self):
        self.assertIn("XL-Sum Bengali validation split", initial_summary_prompt("সংবাদ"))
        self.assertIn("qualifier", conventional_feedback_prompt("সংবাদ", "সারাংশ"))
        self.assertIn("Rule-based qualifier audit flags", evidence_feedback_prompt("সংবাদ", "সারাংশ", {}))


if __name__ == "__main__":
    unittest.main()
