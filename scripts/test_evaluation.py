import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).with_name("summarize-evaluation.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_known_binary_errors_flips_and_threshold_crossings(self):
        suite = {"version": 1, "cases": [{"id": "case", "expected": False}]}

        def report(mode, positive):
            return {"precision": mode, "suite_version": 1, "cases": [{"id": "case", "bindings": {}, "response": {"answers": {"q": {"type": "noul", "noul": positive}}}}]}

        result = evaluation.summarize(suite, [report("checkpoint", 0.1), report("int8", 0.6)])["modes"]
        self.assertEqual(result["checkpoint"]["semantic_failures"], [])
        self.assertAlmostEqual(result["checkpoint"]["labeled_brier_sum_over_classes"], 0.02)
        self.assertAlmostEqual(result["checkpoint"]["labeled_nll"], -math.log(0.9))
        self.assertAlmostEqual(result["checkpoint"]["labeled_ece_10_bins"], 0.1)
        self.assertEqual(result["int8"]["answer_flips_vs_checkpoint"], ["case/q"])
        self.assertEqual(result["int8"]["semantic_failures"], ["case"])
        self.assertAlmostEqual(result["int8"]["max_absolute_probability_drift"], 0.5)
        self.assertEqual(result["int8"]["confidence_threshold_crossings_vs_checkpoint"], {"0.8": 1, "0.9": 1, "0.95": 0})

    def test_mismatched_evaluation_corpus_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cases/version"):
            evaluation.summarize({"version": 2, "cases": []}, [{"precision": "checkpoint", "suite_version": 1, "cases": []}])


if __name__ == "__main__":
    unittest.main()
