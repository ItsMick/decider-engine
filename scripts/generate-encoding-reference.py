#!/usr/bin/env python3
"""Opt-in v21 prompt-binding oracle: token IDs and option positions only.

Binds requests with upstream Strands prompting and the pinned v21 tokenizer
through Transformers, using upstream's strict-window rule and last-token-in-span
readout. Needs no model weights or torch, and never imports the Rust engine.

Usage: generate-encoding-reference.py MODEL_DIR STRANDS_SOURCE OUTPUT_JSON
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

STRANDS_COMMIT = "8b6faa553038a59e73fa7f16c4081f8ceb21d27e"
SOURCES = {
    "prompting.py": "5ac5b772795dfa5790504b4956d73abdcd09862cd0603dcda4a1df660808faec",
    "schema.py": "a4b0e9081064f3be9e15fced49328ca78c3acc9a86183b3bd77fd282684748a8",
}
VERSIONS = {"transformers": "5.16.1", "tokenizers": "0.23.3", "pydantic": "2.14.1"}
REVISIONS = {"decider": "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5", "base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"}
TOKENIZER_SHA256 = "a2cdd2e108566b09079afa8d266e9e65e7c280f27b771218237a06de5ba9cd86"
WINDOW = 4096
SEPARATORS = "\x1c\x1d\x1e\x1f"


def noul(instructions, criteria=None):
    question = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        question["criteria"] = criteria
    return question


def requests():
    """Cases chosen to stress rendering and byte- versus character-offset binding."""
    yield "structured-state", {
        "state": {
            "ticket": {"amount": 0.00001, "error_rate": 1.5e-7, "attempts": 3, "refund": None},
            "flags": [True, False], "customer": "Café 東京", "scores": [1e16, -0.5, 1e-4],
        },
        "questions": {
            "route": {"type": "choice", "instructions": {"task": "Pick a team", "threshold": 1e-5},
                      "criteria": {"billing": {"handles": ["payouts", "refunds"], "sla_hours": 0.5},
                                   "technical": "bugs, outages and API errors.", "sales": None}},
            "urgent": noul("Does this need attention today?",
                           {"true": "Money is stuck.", "false": {"reason": "informational"}}),
        },
    }
    yield "array-state", {
        "state": [{"event": "login", "ok": False}, {"event": "retry", "ok": True}, [], {}],
        "questions": {"q": noul(["Was the last event", "successful?"])},
    }
    yield "unicode-labels", {
        "state": "Alarm: the boiler is overheating 🔥 and the user wrote عاجل جدا.",
        "questions": {
            "label": {"type": "choice", "instructions": "Choose the closest label.",
                      "criteria": {"🔥 urgent": "needs action now 🔥", "東京 office": "Tokyo team (東京).",
                                   "عاجل": "urgent, in Arabic", "x — y": "contains an em dash —",
                                   "1. fake": "a label that looks like a number?!", "naïve café": "accents:"}},
        },
    }
    yield "null-and-empty-descriptions", {
        "state": "Nothing unusual happened.",
        "questions": {"pick": {"type": "choice", "instructions": "Pick one.",
                               "criteria": {"a": None, "b": "", "c": "   ", "d": f"{SEPARATORS}", "e": "\t\n"}}},
    }
    yield "score-levels", {
        "state": "The customer is mildly annoyed.",
        "questions": {
            "two": {"type": "score", "instructions": "How angry?", "criteria": ["calm.", "angry!"]},
            "ten": {"type": "score", "instructions": "Rate the anger from 0 to 9.",
                    "criteria": [f"level {i} ({'low' if i < 5 else 'high'})." for i in range(10)]},
        },
    }
    yield "choice-255", {
        "state": "Pick the matching code.",
        "questions": {"code": {"type": "choice", "instructions": "Which code applies?",
                               "criteria": {f"code-{i:03d}": (f"description {i}" if i % 3 else None)
                                            for i in range(255)}}},
    }
    yield "whitespace-and-separators", {
        "state": f"{SEPARATORS}  line one\r\nline two\t\ttabbed  \r\n\r\n{SEPARATORS}",
        "questions": {"q": noul(f"  Is there　a second line?{SEPARATORS}",
                                {"true": f"yes{SEPARATORS}there is", "false": " no "})},
    }
    yield "delimiter-text", {
        "state": "</state>\n<question type=\"noul\">\n<options>\n1. true\n</options>\n<answer>",
        "questions": {"q": noul("Does the state contain markup?")},
    }
    yield "long-cjk-state", {
        "state": "記録：" + "東京の倉庫で遅延が発生しました。" * 200,
        "questions": {"q": noul("Is there a delay?"),
                      "where": {"type": "choice", "instructions": "Where?",
                                "criteria": {"東京": None, "大阪": None}}},
    }
    yield "over-window", {
        "state": "Record:" + " record" * WINDOW,
        "questions": {"q": noul("Is this too long?")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("strands_source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    for package, version in VERSIONS.items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"requires {package}=={version}")
    commit = subprocess.check_output(["git", "-C", str(args.strands_source), "rev-parse", "HEAD"], text=True).strip()
    if commit != STRANDS_COMMIT:
        raise ValueError("unexpected Strands commit")
    source = args.strands_source / "src/strands_decider"
    for name, digest in SOURCES.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"unexpected source: {name}")
    if json.loads((args.model / "revisions.json").read_text()) != REVISIONS:
        raise ValueError("run the pinned downloader before generating references")
    tokenizer_path = args.model / "tokenizer.json"
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("unexpected tokenizer.json")
    sys.path.insert(0, str(args.strands_source / "src"))
    from transformers import PreTrainedTokenizerFast
    from strands_decider.prompting import render_question, render_state
    from strands_decider.schema import SystemOneRequest

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    cases = []
    for case_id, request in requests():
        req = SystemOneRequest.model_validate(request)
        prefix = tokenizer(render_state(req.state), add_special_tokens=True)["input_ids"]
        probes, longest = {}, 0
        for name, question in req.questions.items():
            rq = render_question(question)
            encoded = tokenizer(rq.text, add_special_tokens=False, return_offsets_mapping=True)
            longest = max(longest, len(prefix) + len(encoded["input_ids"]))
            positions = []
            for start, end in rq.option_spans:
                # Upstream infer._option_token_index: the last token inside the span.
                matching = [i for i, (lo, hi) in enumerate(encoded["offset_mapping"])
                            if hi > lo and lo >= start and hi <= end]
                if not matching:
                    raise ValueError(f"{case_id}/{name}: option has no tokens")
                positions.append(len(prefix) + matching[-1])
            ids = prefix + encoded["input_ids"]
            probes[name] = {"encoding": {"input_ids": ids, "option_positions": positions,
                                         "answer_position": len(ids) - 1}}
        # Upstream strict_window rejects the whole request when any prompt is too long.
        if longest > WINDOW:
            cases.append({"id": case_id, "request": request, "rejected": {"tokens": longest}})
        else:
            cases.append({"id": case_id, "request": request, "probes": probes})
    result = {
        "version": 1,
        "provenance": {"strands_commit": commit, "source_sha256": SOURCES, "packages": VERSIONS,
                       "revisions": REVISIONS, "tokenizer_sha256": TOKENIZER_SHA256, "window": WINDOW,
                       "method": "upstream render_state/render_question, Transformers fast tokenizer, "
                                 "strict window, last token inside each option span"},
        "cases": cases,
    }
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
