import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine.preprocessing import clean_article_text, normalize_article_text


class PreprocessingV511Tests(unittest.TestCase):
    def test_unicode_and_whitespace_normalization(self):
        text = "  প্রথম\n\n   সংবাদ\u00a0এখানে।  "
        cleaned, meta, review = clean_article_text(text)
        self.assertEqual(cleaned, "প্রথম সংবাদ এখানে।")
        self.assertEqual(meta["status"], "normalized_only")
        self.assertFalse(meta["review_required"])
        self.assertEqual(review, [])

    def test_no_marker_text_is_not_paraphrased_or_rewritten(self):
        text = "বাংলাদেশে আজ নতুন সিদ্ধান্ত ঘোষণা করা হয়েছে।"
        cleaned, meta, _ = clean_article_text(text)
        self.assertEqual(cleaned, text)
        self.assertEqual(meta["navigation_marker_count"], 0)

    def test_bare_trailing_navigation_label_is_removed(self):
        text = "মূল সংবাদ এখানে শেষ হয়েছে। আরও পড়ুন:"
        cleaned, meta, review = clean_article_text(text)
        self.assertEqual(cleaned, "মূল সংবাদ এখানে শেষ হয়েছে।")
        self.assertEqual(meta["status"], "auto_bare_trailing_label_cleaned")
        self.assertFalse(meta["review_required"])
        self.assertEqual(review, [])

    def test_trailing_related_link_block_requires_explicit_review(self):
        text = "মূল সংবাদ এখানে শেষ হয়েছে। আরও পড়ুন: সম্পর্কিত খবর এক সম্পর্কিত খবর দুই"
        cleaned, meta, review = clean_article_text(text)
        self.assertEqual(cleaned, normalize_article_text(text))
        self.assertTrue(meta["review_required"])
        self.assertEqual(len(review), 1)

    def test_inline_navigation_is_not_guessed_or_silently_removed(self):
        text = (
            "আগের সংবাদ বাক্য। আরও পড়ুন: সম্পর্কিত শিরোনাম এক সম্পর্কিত শিরোনাম দুই "
            "পরবর্তী মূল অনুচ্ছেদ এখানে শুরু হয়েছে।"
        )
        cleaned, meta, review = clean_article_text(text)
        self.assertEqual(cleaned, normalize_article_text(text))
        self.assertTrue(meta["review_required"])
        self.assertEqual(meta["status"], "manual_review_required")
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["marker_index"], 0)

    def test_manual_resume_phrase_removes_only_inline_navigation_block(self):
        text = (
            "আগের সংবাদ বাক্য। আরও পড়ুন: সম্পর্কিত শিরোনাম এক সম্পর্কিত শিরোনাম দুই "
            "পরবর্তী মূল অনুচ্ছেদ এখানে শুরু হয়েছে।"
        )
        resume = "পরবর্তী মূল অনুচ্ছেদ এখানে শুরু হয়েছে।"
        cleaned, meta, review = clean_article_text(text, {
            0: {"decision": "remove_inline_block", "resume_at": resume}
        })
        self.assertEqual(cleaned, "আগের সংবাদ বাক্য। " + resume)
        self.assertEqual(meta["status"], "manual_navigation_cleaned")
        self.assertFalse(meta["review_required"])
        self.assertEqual(review, [])

    def test_manual_keep_requires_a_reason(self):
        text = "আগের বাক্য। আরও পড়ুন: সম্ভাব্য লিংক পরবর্তী মূল বাক্য।"
        with self.assertRaisesRegex(ValueError, "review_note"):
            clean_article_text(text, {0: {"decision": "keep"}})

    def test_manual_keep_is_logged(self):
        text = "আগের বাক্য। আরও পড়ুন: সম্ভাব্য লিংক পরবর্তী মূল বাক্য।"
        cleaned, meta, review = clean_article_text(text, {
            0: {"decision": "keep", "review_note": "পর্যালোচনা করে article content হিসেবে রাখা হয়েছে"}
        })
        self.assertIn("আরও পড়ুন:", cleaned)
        self.assertEqual(meta["status"], "manually_reviewed_kept")
        self.assertFalse(meta["review_required"])
        self.assertEqual(review, [])

    def test_ambiguous_resume_phrase_is_rejected(self):
        text = "শুরু। আরও পড়ুন: লিংক শিরোনাম। একই বাক্য। একই বাক্য।"
        with self.assertRaisesRegex(ValueError, "exactly once"):
            clean_article_text(text, {
                0: {"decision": "remove_inline_block", "resume_at": "একই বাক্য।"}
            })

    def test_bbc_bangla_related_read_marker_is_detected(self):
        text = "মূল সংবাদ। বিবিসি বাংলায় আরো পড়ুন: চিকিৎসকেরা বলছেন, নতুন তথ্য।"
        cleaned, meta, review = clean_article_text(text)
        self.assertTrue(meta["review_required"])
        self.assertEqual(meta["navigation_marker_count"], 1)
        self.assertEqual(review[0]["marker_text"], "বিবিসি বাংলায় আরো পড়ুন:")

    def test_manual_trailing_block_removal(self):
        text = "মূল সংবাদ। আরও পড়ুন: সম্পর্কিত খবর এক সম্পর্কিত খবর দুই"
        cleaned, meta, review = clean_article_text(text, {
            0: {"decision": "remove_trailing_block"}
        })
        self.assertEqual(cleaned, "মূল সংবাদ।")
        self.assertEqual(meta["status"], "manual_navigation_cleaned")
        self.assertFalse(meta["review_required"])
        self.assertEqual(review, [])

    def test_original_and_cleaned_hashes_are_recorded(self):
        text = "মূল সংবাদ। আরও পড়ুন: সম্পর্কিত লিংক"
        cleaned, meta, _ = clean_article_text(text, {
            0: {"decision": "remove_trailing_block"}
        })
        self.assertIn("original_sha256", meta)
        self.assertIn("cleaned_sha256", meta)
        self.assertNotEqual(meta["original_sha256"], meta["cleaned_sha256"])
        self.assertEqual(meta["cleaned_characters"], len(cleaned))


if __name__ == "__main__":
    unittest.main()
