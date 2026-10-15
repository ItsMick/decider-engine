#!/usr/bin/env python3
"""Run both clients sequentially, recording raw samples and Linux process RSS."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def measure(command, prefix):
    start = time.monotonic()
    with prefix.with_suffix(".jsonl").open("w") as out, prefix.with_suffix(".log").open("w") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err)
        _, status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(status)
    metadata = {"command": command, "exit_code": process.returncode,
                "wall_s": time.monotonic()-start, "peak_rss_kib": usage.ru_maxrss}
    prefix.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if process.returncode:
        raise RuntimeError(f"{command[0]} failed: see {prefix}.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", required=True)
    parser.add_argument("--llama", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("positive repeats/threads required")
    args.output.mkdir(parents=True, exist_ok=False)
    cases = str(args.prepared / "cases.json")
    common = [cases, str(args.repeats), str(args.threads)]
    measure([args.native, args.model, *common], args.output / "native")
    measure([args.llama, args.gguf, args.lora, str(args.prepared / "head.json"), *common], args.output / "llama")
    script = Path(__file__).with_name("summarize-llama-comparison.py")
    import sys
    with (args.output / "summary.json").open("w") as out:
        subprocess.run([sys.executable, str(script), cases, str(args.output / "native.jsonl"), str(args.output / "llama.jsonl")], stdout=out, check=True)


if __name__ == "__main__":
    main()
