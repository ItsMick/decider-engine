"""A timing gain must not hide changed logits or a reference failure."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class BlockingEvidence(unittest.TestCase):
    def run_check(self, before, after, reference=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"id": "probe", "encoding": {"input_ids": [1, 2, 3]},
                    "logits": reference if reference is not None else [1.0, 0.0]}
            (root / "cases.json").write_text(json.dumps([case]))
            for name, logits, elapsed in [("before", before, 20), ("after", after, 10)]:
                records = [{"threads": 4}]
                records += [{"id": "probe", "sample": i, "warmup": i == 0,
                             "tokens": 3, "elapsed_ms": elapsed, "logits": logits}
                            for i in range(2)]
                (root / name).write_text("\n".join(json.dumps(row) for row in records))
            return subprocess.run(
                [sys.executable, str(Path(__file__).with_name("check-blocking-results.py")),
                 str(root / "cases.json"), str(root / "before"), str(root / "after")],
                capture_output=True, text=True)

    def test_matching_logits_report_speedup(self):
        result = self.run_check([1.0, 0.0], [1.0, 0.0])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["cases"][0]["speedup"], 2.0)

    def test_faster_changed_logits_fail(self):
        self.assertNotEqual(self.run_check([1.0, 0.0], [1.01, 0.0]).returncode, 0)

    def test_equal_but_wrong_logits_fail(self):
        self.assertNotEqual(self.run_check([1.1, 0.0], [1.1, 0.0]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
