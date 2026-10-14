#!/usr/bin/env python3
"""Export pinned reference encodings and the FP32 head for llama-compare.cpp."""
import argparse
import json
from pathlib import Path
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    head = load_file(args.model / "head.safetensors")
    data = {k: v.flatten().tolist() for k, v in head.items()}
    data.update(hidden=len(head["norm.weight"]), dim=len(head["q.bias"]))
    (args.output / "head.json").write_text(json.dumps(data))
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / "tests/fixtures/v21-expanded-reference.json").read_text())
    if json.loads((args.model / "revisions.json").read_text()) != fixture["provenance"]["revisions"]:
        raise ValueError("model revisions do not match the reference fixture")
    config = json.loads((args.model / "strands_decider_config.json").read_text())
    cases = []
    for case in fixture["cases"]:
        if case["id"] not in ("original-three", "window-1024"):
            continue
        for name, probe in case["probes"].items():
            kind = case["request"]["questions"][name]["type"]
            cases.append({"id": case["id"] + "/" + name, **probe,
                          "kind": kind, "temperature": config["temperature_by_kind"].get(kind, config["temperature"])})
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")


if __name__ == "__main__":
    main()
