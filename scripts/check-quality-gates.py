#!/usr/bin/env python3
"""Fail-closed, versioned regression gates. Passing is not deployment approval.

Freeze policy and held-out data before collecting candidate results. The policy
names each precision separately; calibration must be fitted on a separate split.
Use real deployment labels to assess quality. Public smoke data cannot do that.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("evaluation", Path(__file__).with_name("summarize-evaluation.py"))
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

METRICS = {
    "answer_flips_vs_checkpoint": len,
    "semantic_failures": len,
    "confidence_threshold_crossings_vs_checkpoint": lambda values: sum(values.values()),
    "max_absolute_probability_drift": float,
    "max_expected_score_drift": float,
    "labeled_brier_sum_over_classes": float,
    "labeled_nll": float,
    "labeled_ece_10_bins": float,
}


def check(suite, reports, policy, suite_sha256):
    if policy.get("version") != 1 or policy.get("status") != "frozen":
        raise ValueError("a version-1 frozen policy is required; templates cannot pass")
    if not policy.get("id") or policy.get("suite_sha256") != suite_sha256:
        raise ValueError("policy id and exact suite SHA256 are required")
    if not policy.get("model_revisions") or not policy.get("config_sha256_by_mode"):
        raise ValueError("policy must bind model revisions and checkpoint calibration/config SHA256")
    if not isinstance(policy.get("minimum_labeled_decisions"), int) or policy["minimum_labeled_decisions"] < 1:
        raise ValueError("positive minimum_labeled_decisions required")
    if not reports or set(policy.get("modes", {})) != {r["precision"] for r in reports}:
        raise ValueError("policy must cover exactly the evaluated modes, including checkpoint")
    if set(policy["config_sha256_by_mode"]) != set(policy["modes"]):
        raise ValueError("bind a calibration/config SHA256 separately for every precision")
    for report in reports:
        if not report.get("config_sha256") or report["config_sha256"] != policy["config_sha256_by_mode"][report["precision"]]:
            raise ValueError(f"{report['precision']}: config_sha256 mismatch or missing")
        for key in ("suite_sha256", "model_revisions"):
            if report.get(key) != policy.get(key):
                raise ValueError(f"{report['precision']}: {key} mismatch or missing")
    summary = evaluation.summarize(suite, reports)
    failures = []
    for mode, metric in summary["modes"].items():
        limits = policy["modes"][mode]
        if not limits or not set(limits).issubset(METRICS):
            raise ValueError(f"{mode}: empty or unknown gate metrics")
        if metric["labeled_decisions"] < policy["minimum_labeled_decisions"]:
            failures.append(f"{mode}: too few labeled decisions")
        for key, maximum in limits.items():
            if isinstance(maximum, bool) or not isinstance(maximum, (float, int)) or not math.isfinite(maximum) or maximum < 0:
                raise ValueError(f"{mode}/{key}: maximum must be finite and nonnegative")
            value = metric[key]
            if value is None or METRICS[key](value) > maximum:
                failures.append(f"{mode}/{key}: {value!r} exceeds {maximum}")
    return {"policy_id": policy["id"], "gate_status": "FAIL" if failures else "PASS", "failures": failures, "validation_status": summary["validation_status"], "modes": summary["modes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        result = check(json.loads(args.suite.read_text()), [json.loads(p.read_text()) for p in args.reports], json.loads(args.policy.read_text()), hashlib.sha256(args.suite.read_bytes()).hexdigest())
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"invalid quality evaluation: {error}\n")
    print(json.dumps(result, indent=2))
    return 0 if result["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
