#!/usr/bin/env python3
"""Validate native numerical gates and report matched before/after timings."""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--allow-rounding-differences", action="store_true",
                        help="require the frozen reference tolerances instead of identical logits")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "summary", Path(__file__).with_name("summarize-llama-comparison.py"))
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    cases = {case["id"]: case for case in json.loads(args.cases.read_text())}
    bm, baseline = summary.read_runs(args.baseline, cases)
    cm, candidate = summary.read_runs(args.candidate, cases)
    if bm["threads"] != cm["threads"]:
        raise ValueError("worker count mismatch")
    output = []
    identical = True
    for key, case in cases.items():
        b, c = baseline[key], candidate[key]
        if len(b) != len(c):
            raise ValueError(f"{key}: sample count mismatch")
        for before, after in zip(b, c):
            if [float(x).hex() for x in before["logits"]] != [float(x).hex() for x in after["logits"]]:
                identical = False
                if not args.allow_rounding_differences:
                    raise ValueError(f"{key}: candidate changed logits")
            if max(abs(a-r) for a, r in zip(after["logits"], case["logits"])) >= 1e-4:
                raise ValueError(f"{key}: reference logit tolerance exceeded")
            if args.allow_rounding_differences:
                actual = summary.probabilities(after["logits"], case["temperature"])
                expected = summary.probabilities(case["logits"], case["temperature"])
                if max(abs(a-r) for a,r in zip(actual,expected)) >= 1e-5:
                    raise ValueError(f"{key}: reference probability tolerance exceeded")
                if case["kind"] == "score" and abs(sum(i*(a-r) for i,(a,r) in enumerate(zip(actual,expected)))) >= 1e-5:
                    raise ValueError(f"{key}: reference score tolerance exceeded")
        bt = statistics.median(row["elapsed_ms"] for row in b if not row["warmup"])
        ct = statistics.median(row["elapsed_ms"] for row in c if not row["warmup"])
        output.append({"id": key, "tokens": len(case["encoding"]["input_ids"]),
                       "baseline_ms": bt, "candidate_ms": ct, "speedup": bt/ct,
                       "max_reference_logit_error": max(abs(a-r) for row in c for a,r in zip(row["logits"],case["logits"]))})
    print(json.dumps({"identical_logits": identical, "rounding_differences_allowed": args.allow_rounding_differences, "cases": output}, indent=2))


if __name__ == "__main__":
    main()
