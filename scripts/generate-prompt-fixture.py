#!/usr/bin/env python3
"""Regenerate tests/fixtures/prompt-parity.json from upstream Strands prompting.

Usage: generate-prompt-fixture.py UPSTREAM_SRC_DIR OUTPUT_JSON

UPSTREAM_SRC_DIR is the `src` directory of a strands-labs/strands-decider
checkout. Its prompting.py and schema.py must match the hashes pinned in
tests/fixtures/README.md. Needs pydantic; no model, tokenizer or network.
"""
import argparse
import hashlib
import json
import platform
import random
import struct
import sys
from pathlib import Path

PINNED = {
    "prompting.py": "5ac5b772795dfa5790504b4956d73abdcd09862cd0603dcda4a1df660808faec",
    "schema.py": "a4b0e9081064f3be9e15fced49328ca78c3acc9a86183b3bd77fd282684748a8",
}

EDGE_FLOATS = [
    0.0, -0.0, 0.1, -0.1, 0.5, 1.0, -1.0, 1.5, 2 / 3, 100.0, 12.5,
    1e-4, 9.999e-5, 1e-5, 0.00001234, 1.5e-7, -2.5e-9, 1e-100, 5e-324, 2.2250738585072014e-308,
    1e15, 9999999999999998.0, 1e16, 1.2345678901234567e16, 1e20, 1e22, 1e100,
    1.7976931348623157e308, 123456789.123, 3.141592653589793, 299792458.0, 0.30000000000000004,
]
INTS = [0, 1, -1, 42, 2**53 + 1, 2**63 - 1, -(2**63), 2**64 - 1]
SEPARATORS = "\x1c\x1d\x1e\x1f"


def random_floats(rng, count):
    values = []
    while len(values) < count:
        if len(values) % 2:
            value = struct.unpack("<d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        else:
            value = rng.uniform(-1, 1) * 10 ** rng.uniform(-30, 30)
        if value == value and abs(value) != float("inf"):
            values.append(value)
    return values


def content_cases(rng):
    strings = [
        "  padded\t", f"{SEPARATORS}separated{SEPARATORS} ", "", "Café 東京", "line\nbreak  ",
        "quote \" backslash \\ controls " + "".join(chr(c) for c in range(32)) + "\x7f",
        " line separators　", "\x85next line\xa0nbsp",
    ]
    structured = [
        {},
        [],
        {"nested": {"empty_list": [], "empty_object": {}, "list": [1, [2, [3, {}]]]}},
        {"flags": [True, False, None], "ints": INTS, "floats": EDGE_FLOATS},
        {"keys \"quoted\"\n": "values", "ünïcode": "東京", "controls": "".join(chr(c) for c in range(32))},
        {"z_first": 1, "a_second": 2, "m_third": 3},
        [{"amount": 0.00001, "rate": 1.5e-7, "count": 3}, {"amount": 1e20}],
    ]
    floats = random_floats(rng, 1000)
    structured.extend(floats[i:i + 100] for i in range(0, len(floats), 100))
    return strings + structured


def prompt_requests():
    return [
        {
            "state": {"ticket": {"amount": 0.00001, "attempts": 3, "error_rate": 1.5e-7}, "tags": ["payout", "retry"]},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": {"task": "Pick a team", "threshold": 1e-5},
                    "criteria": {
                        "billing": {"handles": ["payments", "payouts"], "sla_hours": 0.5},
                        "technical": f"  bugs,{SEPARATORS}outages \n API errors ",
                        "sales": None,
                    },
                },
                "urgent": {
                    "type": "noul",
                    "instructions": f"{SEPARATORS} Needs attention? \x1d",
                    "criteria": {"true": f"  holds{SEPARATORS}now ", "false": None},
                },
                "severity": {
                    "type": "score",
                    "instructions": "How severe?",
                    "criteria": [f"  none\x1c ", "minor　issue", f"{SEPARATORS}", "blocked\n\tservice"],
                },
            },
        },
        {
            "state": [1e-6, {"nested": [], "empty": {}}, "text", 2**63 - 1],
            "questions": {
                "level": {
                    "type": "score",
                    "instructions": ["step one", {"step": 2}],
                    "criteria": [f"level {i}" for i in range(10)],
                },
                "pick": {
                    "type": "choice",
                    "instructions": "Choose",
                    "criteria": {"ä": None, "東京": "", "x — y": {"weight": 1e16}},
                },
            },
        },
        {
            "state": f"{SEPARATORS}  Café 東京 failing for 0.00001s \x1f",
            "questions": {"plain": {"type": "noul", "instructions": "Is it failing?"}},
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_src", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package = args.upstream_src / "strands_decider"
    hashes = {}
    for name, digest in PINNED.items():
        hashes[name] = hashlib.sha256((package / name).read_bytes()).hexdigest()
        if hashes[name] != digest:
            raise SystemExit(f"{name} does not match the pinned upstream hash")
    sys.path.insert(0, str(args.upstream_src))
    from strands_decider import prompting, schema

    rng = random.Random(20261013)
    contents = [{"value": v, "rendered": prompting.render_content(v)} for v in content_cases(rng)]
    prompts = []
    for raw in prompt_requests():
        request = schema.SystemOneRequest.model_validate(raw)
        questions = {}
        for name, question in request.questions.items():
            rendered = prompting.render_question(question)
            questions[name] = {
                "text": rendered.text,
                "spans": [list(span) for span in rendered.option_spans],
                "labels": list(rendered.slot_labels),
                "descriptions": list(rendered.slot_descriptions),
            }
        prompts.append({"request": raw, "state": prompting.render_state(request.state), "questions": questions})
    fixture = {
        "generator": "scripts/generate-prompt-fixture.py",
        "python": platform.python_version(),
        "upstream_sha256": hashes,
        "content": contents,
        "prompts": prompts,
    }
    args.output.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
