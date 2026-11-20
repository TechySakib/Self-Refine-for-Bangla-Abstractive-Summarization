import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bangla_refine import pipeline


class FakeClient:
    def __init__(self, summary):
        self.summary = summary
        self.calls = 0

    def generate(self, prompt, json_mode=False, temperature=None):
        self.calls += 1
        if "বাংলা সারসংক্ষেপ:" in prompt:
            output = self.summary
        elif "Revised Bangla summary:" in prompt:
            output = self.summary
        elif "বর্তমান summary:" in prompt and "Feedback:" in prompt:
            output = "NO_ACTIONABLE_ISSUES"
        elif "JSON feedback:" in prompt:
            output = json.dumps({"overall_assessment": "পর্যালোচনা করা হয়েছে", "actionable_issues": []}, ensure_ascii=False)
        elif "অনুমোদিত feedback:" in prompt:
            output = self.summary
        else:
            output = self.summary
        return output, {
            "model": "fake-model",
            "temperature": 0.0 if temperature == 0 else 0.2,
            "latency_seconds": 0.01,
            "prompt_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
        }


class PipelineV51IntegrationTests(unittest.TestCase):
    def test_python_length_review_is_injected_into_b_and_c(self):
        long_summary = " ".join(["বাংলা"] * 36)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            processed = tmp_path / "processed"
            results = tmp_path / "results"
            processed.mkdir()
            results.mkdir()
            input_path = processed / "pilot.jsonl"
            row = {
                "id": "mock-1",
                "title": "পরীক্ষামূলক শিরোনাম",
                "text": " ".join(["মূলসংবাদ"] * 40) + "।",
                "summary": "সংক্ষিপ্ত রেফারেন্স।",
                "url": "",
            }
            input_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            fake_client = FakeClient(long_summary)

            with patch.object(pipeline, "PROCESSED_DIR", processed), \
                 patch.object(pipeline, "RESULTS_DIR", results), \
                 patch.object(pipeline, "GeminiTextClient", return_value=fake_client):
                pipeline.run_split("pilot", limit=1, max_rounds=2)

            output = json.loads((results / "pilot_results.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(output["pipeline_version"], "5.1.1")
            self.assertTrue(output["system_b_rounds"][0]["length_review_triggered"])
            self.assertTrue(output["system_c_rounds"][0]["length_review_triggered"])
            self.assertTrue(any(
                issue.get("origin") == "deterministic_length_review"
                for issue in output["system_c_rounds"][0]["eligible_issues"]
            ))
            self.assertEqual(output["summary_length_profile"]["system_c_final"]["whitespace_words"], 36)
            self.assertIn("reference_alignment_audit", output)
            self.assertEqual(output["api_calls"], 5)


if __name__ == "__main__":
    unittest.main()
