#!/usr/bin/env python3
"""Compare evaluate-example reports to checkpoint precision; no model execution.

Thresholds describe probability crossings only, never a deployment approval.
This smoke corpus is too small and targeted for calibration or quality claims.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def distribution(answer):
    if answer["type"] == "noul":
        return {"false": 1 - answer["noul"], "true": answer["noul"]}
    return answer["probabilities"]


def summarize(suite, reports):
    cases = {case["id"]: case for case in suite["cases"]}
    modes = {report["precision"]: {c["id"]: c for c in report["cases"]} for report in reports}
    if len(modes) != len(reports) or "checkpoint" not in modes:
        raise ValueError("provide one checkpoint report and at most one report per other mode")
    for report in reports:
        if report["suite_version"] != suite["version"] or set(modes[report["precision"]]) != set(cases):
            raise ValueError("report cases/version do not match suite")
    reference = modes["checkpoint"]
    result = {}
    for mode, rows in modes.items():
        drift, relative_drift, flips, labeled, failures, scores = [], [], [], [], [], []
        crossings = {str(t): 0 for t in [0.8, 0.9, 0.95]}
        for id, row in rows.items():
            if row["bindings"] != reference[id]["bindings"]:
                raise ValueError(f"{mode}/{id}: token binding changed")
            for name, answer in row["response"]["answers"].items():
                baseline = reference[id]["response"]["answers"][name]
                p, b = distribution(answer), distribution(baseline)
                if p.keys() != b.keys() or answer["type"] != baseline["type"]:
                    raise ValueError(f"{mode}/{id}/{name}: answer domain changed")
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()):
                    raise ValueError("invalid probabilities")
                if abs(sum(p.values()) - 1) > 1e-5:
                    raise ValueError("probabilities do not sum to one")
                for key in p:
                    delta = abs(p[key] - b[key])
                    drift.append(delta)
                    relative_drift.append(delta / max(abs(b[key]), 1e-6))
                # Use the model's first-option tie behavior for choice. Score
                # comparisons use expected-value drift, not rounded categories.
                if answer["type"] == "choice":
                    winner, base_winner = answer["choice"], baseline["choice"]
                elif answer["type"] == "noul":
                    winner = "true" if answer["noul"] > 0.5 else "false"
                    base_winner = "true" if baseline["noul"] > 0.5 else "false"
                else:
                    scores.append(abs(answer["score"] - baseline["score"]))
                    continue
                if winner != base_winner:
                    flips.append(id + "/" + name)
                for threshold in crossings:
                    if (max(p.values()) >= float(threshold)) != (max(b.values()) >= float(threshold)):
                        crossings[threshold] += 1
                expected = cases[id].get("expected")
                if expected is not None:
                    expected = str(expected).lower() if isinstance(expected, bool) else expected
                    correct = winner == expected
                    confidence = p[winner]
                    labeled.append((confidence, correct, sum((v - (k == expected)) ** 2 for k, v in p.items()), -math.log(max(p[expected], 1e-12))))
                    if not correct:
                        failures.append(id)
        count = len(labeled)
        ece = 0.0
        for i in range(10):
            bucket = [r for r in labeled if min(int(r[0] * 10), 9) == i]
            if bucket:
                ece += abs(sum(r[0] - r[1] for r in bucket)) / count
        result[mode] = {
            "cases": len(rows), "labeled_decisions": count, "semantic_failures": failures,
            "answer_flips_vs_checkpoint": flips,
            "mean_absolute_probability_drift": sum(drift) / len(drift),
            "max_absolute_probability_drift": max(drift),
            "max_relative_probability_drift_floor_1e_6": max(relative_drift),
            "max_expected_score_drift": max(scores, default=0),
            "confidence_threshold_crossings_vs_checkpoint": crossings,
            "labeled_brier_sum_over_classes": sum(r[2] for r in labeled) / count if count else None,
            "labeled_nll": sum(r[3] for r in labeled) / count if count else None,
            "labeled_ece_10_bins": ece if count else None,
            "confident_wrong_decisions": {str(t): sum(c >= t and not ok for c, ok, _, _ in labeled) for t in [0.8, 0.9, 0.95]},
            "correct_but_below_threshold": {str(t): sum(c < t and ok for c, ok, _, _ in labeled) for t in [0.8, 0.9, 0.95]},
        }
    return {"validation_status": "UNVALIDATED: smoke data cannot establish deployment quality or calibration", "modes": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.suite.read_text()), [json.loads(p.read_text()) for p in args.reports])
    result["suite_sha256"] = hashlib.sha256(args.suite.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
