#!/usr/bin/env python3
"""Summarize matched single-question warm runs; no tail-latency claims."""
import argparse
import json
import math
import statistics
from pathlib import Path


def read_runs(path, cases):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    rows = records[1:]
    assert set(row["id"] for row in rows) == set(cases), "case mismatch"
    grouped = {}
    for key, case in cases.items():
        samples = [row for row in rows if row["id"] == key]
        assert len(samples) >= 2
        assert sorted(row["sample"] for row in samples) == list(range(len(samples)))
        for row in samples:
            assert row["warmup"] == (row["sample"] == 0)
            assert row["tokens"] == len(case["encoding"]["input_ids"])
            assert len(row["logits"]) == len(case["logits"])
            assert all(math.isfinite(v) for v in row["logits"])
            assert math.isfinite(row["elapsed_ms"]) and row["elapsed_ms"] > 0
        grouped[key] = samples
    return records[0], grouped


def probabilities(logits, temperature):
    values = [math.exp((v - max(logits))/temperature) for v in logits]
    return [v/sum(values) for v in values]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("llama", type=Path)
    args = parser.parse_args()
    cases = {c["id"]: c for c in json.loads(args.cases.read_text())}
    native_meta, native = read_runs(args.native, cases)
    llama_meta, llama = read_runs(args.llama, cases)
    assert native_meta["threads"] == llama_meta["threads"]
    summary = []
    for key, case in cases.items():
        n, l = native[key], llama[key]
        assert len(n) == len(l), "sample count mismatch"
        nt = [row["elapsed_ms"] for row in n if not row["warmup"]]
        lt = [row["elapsed_ms"] for row in l if not row["warmup"]]
        drift = max(abs(a-b) for nr, lr in zip(n,l) for a,b in zip(nr["logits"],lr["logits"]))
        pd = max(abs(a-b) for nr,lr in zip(n,l) for a,b in zip(probabilities(nr["logits"],case["temperature"]),probabilities(lr["logits"],case["temperature"])))
        summary.append({"id":key,"tokens":len(case["encoding"]["input_ids"]),
                        "warm_samples":len(nt),"native_median_ms":statistics.median(nt),
                        "llama_median_ms":statistics.median(lt),
                        "native_over_llama_ratio":statistics.median(nt)/statistics.median(lt),
                        "max_abs_logit_difference":drift,
                        "max_abs_calibrated_probability_difference":pd,
                        "argmax_matches":all(max(range(len(nr["logits"])),key=nr["logits"].__getitem__)==max(range(len(lr["logits"])),key=lr["logits"].__getitem__) for nr,lr in zip(n,l)),
                        "native_max_reference_error":max(abs(a-b) for row in n for a,b in zip(row["logits"],case["logits"])),
                        "llama_max_reference_error":max(abs(a-b) for row in l for a,b in zip(row["logits"],case["logits"]))})
    print(json.dumps({"threads":native_meta["threads"],"cases":summary,
                      "note":"Per-question raw-logit execution including pointer head; no shared-prefix cache, tokenizer, HTTP, score confidence or response serialization. Different arithmetic: strict reference equivalence is not assumed."},indent=2))


if __name__ == "__main__":
    main()
