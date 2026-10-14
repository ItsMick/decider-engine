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


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("gates", Path(__file__).with_name("check-quality-gates.py"))
        self.gates = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gates)
        self.suite = {"version": 1, "cases": [{"id": "x", "expected": True}]}
        self.reports = [{"precision": mode, "suite_version": 1, "suite_sha256": "suite", "model_revisions": {"base": "pinned", "decider": "pinned"}, "config_sha256": "config", "cases": [{"id": "x", "bindings": {}, "response": {"answers": {"q": {"type": "noul", "noul": p}}}}]} for mode, p in [("checkpoint", 0.9), ("int8", 0.4)]]
        self.policy = {"version": 1, "status": "frozen", "id": "test-only", "suite_sha256": "suite", "model_revisions": {"base": "pinned", "decider": "pinned"}, "config_sha256_by_mode": {"checkpoint": "config", "int8": "config"}, "minimum_labeled_decisions": 1, "modes": {mode: {"answer_flips_vs_checkpoint": 0} for mode in ["checkpoint", "int8"]}}

    def test_gate_detects_mode_specific_flip(self):
        result = self.gates.check(self.suite, self.reports, self.policy, "suite")
        self.assertEqual(result["gate_status"], "FAIL")
        self.assertIn("int8/answer_flips", result["failures"][0])
        self.assertIn("UNVALIDATED", result["validation_status"])

    def test_cannot_compare_different_calibration_or_missing_answers(self):
        self.reports[1]["config_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            self.gates.check(self.suite, self.reports, self.policy, "suite")
        self.reports[1]["config_sha256"] = "config"
        self.reports[1]["cases"][0]["response"]["answers"] = {}
        with self.assertRaisesRegex(ValueError, "missing or extra answers"):
            self.gates.check(self.suite, self.reports, self.policy, "suite")

    def test_duplicate_cases_and_nonfinite_gates_fail_closed(self):
        self.policy["modes"]["int8"]["answer_flips_vs_checkpoint"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.gates.check(self.suite, self.reports, self.policy, "suite")
        self.reports[0]["cases"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluation.summarize(self.suite, self.reports)

    def test_template_or_unlabeled_data_cannot_pass(self):
        self.policy["status"] = "template"
        with self.assertRaisesRegex(ValueError, "frozen"):
            self.gates.check(self.suite, self.reports, self.policy, "suite")
        self.policy["status"] = "frozen"
        self.suite["cases"][0].pop("expected")
        self.assertIn("too few labeled", self.gates.check(self.suite, self.reports, self.policy, "suite")["failures"][0])


if __name__ == "__main__":
    unittest.main()
