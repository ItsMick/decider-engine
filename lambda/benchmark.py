#!/usr/bin/env python3
"""Plan or execute a private, disposable CloudFormation Lambda benchmark.

No resources are created by this script. Deploy template.yaml deliberately first.
The default is read-only planning; --execute updates only the tagged benchmark
stack and restores its parameters afterward. Each result is checkpointed to disk.
Pricing is supplied explicitly; estimates exclude storage, logs and data transfer.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

MODES = ("checkpoint", "int8", "int8-compute")
MEMORIES = (6144, 8192, 10240)


def aws(*args):
    result = subprocess.run(["aws", "--no-cli-pager", "--output", "json", "--cli-read-timeout", "960", *map(str, args)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def quantile(values, p):
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]  # nearest rank


def parse_report(log, request_id):
    lines = [line for line in log.splitlines() if line.startswith("REPORT RequestId: " + request_id + "\t")]
    if len(lines) != 1:
        raise ValueError("missing or ambiguous matching Lambda REPORT; keep raw evidence")
    fields = dict(re.findall(r"(?:^|\t)([A-Za-z ]+): ([0-9.]+) (?:ms|MB)(?=\t|$)", lines[0]))
    required = ("Duration", "Billed Duration", "Memory Size", "Max Memory Used")
    if any(key not in fields for key in required):
        raise ValueError("incomplete REPORT")
    return {key: float(value) for key, value in fields.items()}


def summarize(samples, gb_second_price, request_price):
    if not samples:
        raise ValueError("no samples")
    for price in (gb_second_price, request_price):
        if not math.isfinite(price) or price < 0:
            raise ValueError("pricing must be finite and nonnegative")
    warm = [s for s in samples if not s["payload"]["metrics"]["cold"]]
    cold = [s for s in samples if s["payload"]["metrics"]["cold"]]
    def stats(rows):
        if not rows:
            return None
        durations = [s["report"]["Duration"] for s in rows]
        return {"n": len(rows), "duration_ms_p50": quantile(durations, .5), "duration_ms_p95": quantile(durations, .95) if len(rows) >= 20 else None, "duration_ms_p99": quantile(durations, .99) if len(rows) >= 100 else None, "max_memory_used_mb": max(s["report"]["Max Memory Used"] for s in rows), "mean_billed_cost": sum(s["report"]["Billed Duration"] / 1000 * s["report"]["Memory Size"] / 1024 * gb_second_price + request_price for s in rows) / len(rows)}
    return {"warm": stats(warm), "cold": stats(cold), "cold_init_ms": [s["report"].get("Init Duration") for s in cold], "cost_scope": "Lambda billed duration and requests only, using caller-supplied prices; no free tier or discounts"}


def update(stack, parameters):
    # Preserve the reviewed template, role, tags and image. A nonce forces a fresh
    # environment; metrics verify that the resulting invocation is actually cold.
    aws("cloudformation", "update-stack", "--stack-name", stack, "--use-previous-template", "--capabilities", "CAPABILITY_IAM", "--parameters", json.dumps(parameters))
    aws("cloudformation", "wait", "stack-update-complete", "--stack-name", stack)


def invoke(function, request):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        (path / "request.json").write_text(json.dumps(request, ensure_ascii=False))
        start = time.monotonic()
        result = aws("lambda", "invoke", "--function-name", function, "--invocation-type", "RequestResponse", "--log-type", "Tail", "--payload", "fileb://" + str(path / "request.json"), path / "response.json")
        elapsed = (time.monotonic() - start) * 1000
        payload = json.loads((path / "response.json").read_text())
    log = base64.b64decode(result.get("LogResult", "")).decode()
    if result.get("FunctionError") or result.get("StatusCode") != 200:
        raise RuntimeError(f"Lambda invocation failed: {result.get('FunctionError')}; {payload}")
    metrics = payload["metrics"]
    if metrics["architecture"] != "aarch64" or metrics["integer_kernel"] != "aarch64-sdot":
        raise ValueError("host did not verify ARM64/SDOT dispatch")
    report = parse_report(log, metrics["request_id"])
    return {"client_wall_ms": elapsed, "payload": payload, "report": report, "lambda_log": log}


def output_drift(baseline, candidate):
    base_answers, answers = baseline["answers"], candidate["answers"]
    if base_answers.keys() != answers.keys() or baseline["usage"] != candidate["usage"]:
        raise ValueError("output domain or token usage changed")
    drift, score_drift, flips = [], [], []
    for name, answer in answers.items():
        base = base_answers[name]
        if answer["type"] != base["type"]:
            raise ValueError("answer type changed")
        if answer["type"] == "noul":
            drift.append(abs(answer["noul"] - base["noul"]))
            if (answer["noul"] > .5) != (base["noul"] > .5):
                flips.append(name)
        else:
            if answer["probabilities"].keys() != base["probabilities"].keys():
                raise ValueError("answer labels changed")
            drift.extend(abs(p - base["probabilities"][k]) for k, p in answer["probabilities"].items())
            if answer["type"] == "choice" and answer["choice"] != base["choice"]:
                flips.append(name)
            if answer["type"] == "score":
                score_drift.append(abs(answer["score"] - base["score"]))
    return {"max_absolute_probability_drift": max(drift), "max_expected_score_drift": max(score_drift, default=0), "answer_flips": flips}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stack")
    parser.add_argument("request", type=Path, help="a three-question request; also measures its first question alone")
    parser.add_argument("--output", type=Path, default=Path("lambda/results/benchmark.json"))
    parser.add_argument("--samples", type=int, default=100, help="warm invocations per workload/configuration")
    parser.add_argument("--cold-samples", type=int, default=5, help="fresh environments per workload/configuration")
    parser.add_argument("--workers", type=int, nargs="+", default=[4])
    parser.add_argument("--gb-second-price", type=float, required=True)
    parser.add_argument("--request-price", type=float, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.samples < 100 or args.cold_samples < 1 or not args.workers or any(w not in range(1, 7) for w in args.workers) or len(set(args.workers)) != len(args.workers):
        parser.error("need >=100 warm samples, >=1 cold sample, and unique worker counts in 1..6")
    if any(not math.isfinite(p) or p < 0 for p in (args.gb_second_price, args.request_price)):
        parser.error("prices must be finite and nonnegative")
    request = json.loads(args.request.read_text())
    if len(request["questions"]) != 3:
        parser.error("supply exactly three questions to compare one and three forwards")
    workloads = {"three": request, "one": {**request, "questions": dict(list(request["questions"].items())[:1])}}
    plan = [{"memory_mb": memory, "workers": workers, "precision": mode, "workload": workload} for memory in MEMORIES for workers in args.workers for mode in MODES for workload in workloads]
    if not args.execute:
        print(json.dumps({"execute": False, "configurations": plan, "invocations": len(plan) * (args.samples + args.cold_samples), "note": "--execute updates the tagged disposable stack and incurs AWS charges; delete the stack afterward"}, indent=2))
        return
    if args.output.exists():
        parser.error("output already exists; use a new filename to preserve evidence")
    stack = aws("cloudformation", "describe-stacks", "--stack-name", args.stack)["Stacks"][0]
    if not any(t["Key"] == "decider-purpose" and t["Value"] == "disposable-benchmark" for t in stack.get("Tags", [])):
        raise ValueError("refusing to modify an untagged/non-benchmark stack")
    function = next(o["OutputValue"] for o in stack["Outputs"] if o["OutputKey"] == "FunctionName")
    host = aws("lambda", "get-function", "--function-name", function)
    if host["Configuration"]["Architectures"] != ["arm64"] or "@sha256:" not in host["Code"]["ResolvedImageUri"]:
        raise ValueError("requires ARM64 container pinned by digest")
    tags = aws("lambda", "list-tags", "--resource", host["Configuration"]["FunctionArn"])["Tags"]
    if tags.get("decider-purpose") != "disposable-benchmark":
        raise ValueError("function must be owned by the disposable benchmark stack")
    parameters = [{"ParameterKey": p["ParameterKey"], "ParameterValue": p["ParameterValue"]} for p in stack["Parameters"]]
    if set(p["ParameterKey"] for p in parameters) != {"ImageUri", "Memory", "Workers", "Precision", "RunNonce"}:
        raise ValueError("unexpected benchmark template parameters")
    result = {"version": 1, "started_utc": datetime.now(timezone.utc).isoformat(), "image_uri": host["Code"]["ResolvedImageUri"], "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(), "pricing": {"gb_second": args.gb_second_price, "request": args.request_price}, "runs": [], "restored": False, "validation_status": "PERFORMANCE EXPERIMENT: not deployment quality/calibration validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temp = args.output.with_suffix(".json.part")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.output)
    baselines = {}
    save()
    try:
        for config in plan:
            run = {**config, "samples": []}
            result["runs"].append(run)
            first = None
            for cold_index in range(args.cold_samples):
                changes = {"Memory": str(config["memory_mb"]), "Workers": str(config["workers"]), "Precision": config["precision"], "RunNonce": uuid.uuid4().hex}
                updated = [{"ParameterKey": p["ParameterKey"], "ParameterValue": changes.get(p["ParameterKey"], p["ParameterValue"])} for p in parameters]
                update(args.stack, updated)
                # First environment contributes warm samples; all others cold only.
                warm_count = 0
                for _ in range(args.samples + 10 if cold_index == 0 else 1):
                    sample = invoke(function, workloads[config["workload"]])
                    metrics = sample["payload"]["metrics"]
                    if metrics["precision"] != config["precision"] or metrics["rayon_workers"] != config["workers"] or sample["report"]["Memory Size"] != config["memory_mb"]:
                        raise ValueError("invocation configuration differs from the experiment")
                    if cold_index > 0 and not metrics["cold"]:
                        raise ValueError("requested cold reset reused a warm environment")
                    response = sample["payload"]["response"]
                    first = first or response
                    if response != first:
                        raise ValueError("resident host returned nondeterministic responses")
                    run["samples"].append(sample)
                    save()
                    warm_count += not metrics["cold"]
                    if cold_index != 0 or warm_count == args.samples:
                        break
                if cold_index == 0 and (warm_count != args.samples or not run["samples"][0]["payload"]["metrics"]["cold"]):
                    raise ValueError("insufficient warm samples or initial invocation was not cold")
            key = (config["memory_mb"], config["workers"], config["workload"])
            if config["precision"] == "checkpoint":
                baselines[key] = first
            run["drift_vs_checkpoint"] = output_drift(baselines[key], first)
            run["summary"] = summarize(run["samples"], args.gb_second_price, args.request_price)
            save()
    finally:
        try:
            update(args.stack, parameters)
            result["restored"] = True
        finally:
            save()
    print(args.output)


if __name__ == "__main__":
    main()
