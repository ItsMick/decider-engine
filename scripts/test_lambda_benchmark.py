import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).resolve().parents[1] / "lambda/benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_report_matches_request_and_billed_cost_does_not_double_count_init(self):
        log = "REPORT RequestId: id\tDuration: 1000.00 ms\tBilled Duration: 1500 ms\tMemory Size: 8192 MB\tMax Memory Used: 3456 MB\tInit Duration: 500.00 ms\n"
        report = benchmark.parse_report(log, "id")
        with self.assertRaisesRegex(ValueError, "matching"):
            benchmark.parse_report(log, "other")
        samples = [{"payload": {"metrics": {"cold": cold}}, "report": report} for cold in [True, False]]
        summary = benchmark.summarize(samples, 0.01, 0.001)
        self.assertAlmostEqual(summary["warm"]["mean_billed_cost"], .121)
        self.assertEqual(summary["cold_init_ms"], [500.0])
        self.assertIsNone(summary["warm"]["duration_ms_p99"])

    def test_tail_quantiles_require_enough_samples_and_use_nearest_rank(self):
        samples = [{"payload": {"metrics": {"cold": False}}, "report": {"Duration": i, "Billed Duration": i, "Memory Size": 10240, "Max Memory Used": 4000}} for i in range(1, 101)]
        warm = benchmark.summarize(samples, .01, 0)["warm"]
        self.assertEqual([warm["duration_ms_p50"], warm["duration_ms_p95"], warm["duration_ms_p99"]], [50, 95, 99])
        self.assertIsNone(benchmark.summarize(samples[:19], .01, 0)["warm"]["duration_ms_p95"])

    def test_output_drift_detects_boolean_flip(self):
        a = {"answers": {"q": {"type": "noul", "noul": .9}}, "usage": {"input_tokens": 80}}
        b = {"answers": {"q": {"type": "noul", "noul": .4}}, "usage": {"input_tokens": 80}}
        result = benchmark.output_drift(a, b)
        self.assertEqual(result["answer_flips"], ["q"])
        self.assertAlmostEqual(result["max_absolute_probability_drift"], .5)
