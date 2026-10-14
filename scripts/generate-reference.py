#!/usr/bin/env python3
"""Opt-in independent FP32 v21 oracle; requires a pinned Strands source checkout.

Uses the unmodified Transformers decoder layers and Strands pointer/readout.
Streams one FP32 layer at a time to avoid keeping a 2B FP32 model resident.
Never imports the Rust implementation or downloads a model. Output is atomic.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

STRANDS_COMMIT = "8b6faa553038a59e73fa7f16c4081f8ceb21d27e"
SOURCES = {
    "modeling.py": "f3a62d408d9e08f7d8e8b42b657bd427d91983ab50ae0e72a7263b453d58e153",
    "prompting.py": "5ac5b772795dfa5790504b4956d73abdcd09862cd0603dcda4a1df660808faec",
    "schema.py": "a4b0e9081064f3be9e15fced49328ca78c3acc9a86183b3bd77fd282684748a8",
}
VERSIONS = {"torch": "2.14.0+cpu", "transformers": "5.16.1", "peft": "0.21.0", "tokenizers": "0.23.3", "safetensors": "0.8.0"}


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
    revisions = json.loads((args.model / "revisions.json").read_text())
    if revisions != {"decider": "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5", "base": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"}:
        raise ValueError("run the pinned downloader before generating references")
    sys.path.insert(0, str(args.strands_source / "src"))
    import torch
    from torch import nn
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, Qwen3_5RMSNorm
    from transformers.masking_utils import create_causal_mask
    from strands_decider.modeling import PointerHead
    from strands_decider.prompting import render_state, render_question, read_noul, read_choice, read_score, score_legend
    from strands_decider.schema import SystemOneRequest, derive_confidence, derive_score_confidence

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(4242)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.model / "tokenizer.json"))
    config = Qwen3_5TextConfig(**json.loads((args.model / "base/config.json").read_text())["text_config"])
    config._attn_implementation = "eager"
    dc = json.loads((args.model / "strands_decider_config.json").read_text())
    ac = json.loads((args.model / "lora/adapter_config.json").read_text())
    adapter = load_file(args.model / "lora/adapter_model.safetensors")

    original = json.loads((Path(__file__).resolve().parents[1] / "tests/request.json").read_text())
    reversed_request = copy.deepcopy(original)
    reversed_request["questions"] = {"route": reversed_request["questions"]["route"]}
    reversed_request["questions"]["route"]["criteria"] = dict(reversed(list(original["questions"]["route"]["criteria"].items())))
    short = {"state": "", "questions": {"q": {"type": "noul", "instructions": "Two plus two equals four."}}}
    cases = [{"id": "original-three", "request": original}, {"id": "reversed-choice", "request": reversed_request}, {"id": "short-noul", "request": short}]

    def bind(request):
        req = SystemOneRequest.model_validate(request)
        prefix = tokenizer(render_state(req.state), add_special_tokens=True)["input_ids"]
        out = []
        for name, question in req.questions.items():
            rq = render_question(question)
            encoded = tokenizer(rq.text, add_special_tokens=False, return_offsets_mapping=True)
            positions = []
            for start, end in rq.option_spans:
                matching = [i for i, (lo, hi) in enumerate(encoded["offset_mapping"]) if hi > lo and lo >= start and hi <= end]
                if not matching:
                    raise ValueError("option has no tokens")
                positions.append(len(prefix) + matching[-1])
            ids = prefix + encoded["input_ids"]
            out.append((name, rq, {"input_ids": ids, "option_positions": positions, "answer_position": len(ids) - 1}))
        return out

    for target in (1024, 4096):
        request = {"state": "", "questions": {"q": {"type": "noul", "instructions": "The final status is approved.", "criteria": {"true": "approved", "false": "rejected"}}}}
        # Each added ' record' is one token here; assert the resulting BPE length.
        request["state"] = "Record:" + " record" * target + ". Final status: approved."
        excess = len(bind(request)[0][2]["input_ids"]) - target
        request["state"] = "Record:" + " record" * (target - excess) + ". Final status: approved."
        assert len(bind(request)[0][2]["input_ids"]) == target
        cases.append({"id": f"window-{target}", "request": request})

    class SeparateLora(nn.Module):
        def __init__(self, linear, a, b):
            super().__init__()
            self.linear = linear
            self.register_buffer("a", a.float())
            self.register_buffer("b", b.float())
        def forward(self, x):
            return self.linear(x) + F.linear(F.linear(x, self.a), self.b) * (ac["lora_alpha"] / ac["r"])

    with torch.inference_mode():
        rows = []
        for case in cases:
            case["probes"] = {}
            case["response"] = {"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
            for name, rq, encoding in bind(case["request"]):
                assert len(encoding["input_ids"]) <= 4096
                rows.append({"case": case, "name": name, "rq": rq, "encoding": encoding})
        index = json.loads((args.model / "base/model.safetensors.index.json").read_text())["weight_map"]
        handles = {name: safe_open(args.model / "base" / name, framework="pt") for name in set(index.values())}
        prefix = "model.language_model."
        def weight(name):
            key = prefix + name
            return handles[index[key]].get_tensor(key)
        embeddings = weight("embed_tokens.weight")
        rotary = Qwen3_5TextRotaryEmbedding(config)
        for row in rows:
            ids = torch.tensor([row["encoding"]["input_ids"]])
            row["hidden"] = F.embedding(ids, embeddings).float()
            row["positions"] = torch.arange(ids.shape[1]).unsqueeze(0)
            row["rotary"] = rotary(row["hidden"], row["positions"].unsqueeze(0).expand(3, -1, -1))
            row["mask"] = create_causal_mask(config=config, inputs_embeds=row["hidden"], attention_mask=torch.ones_like(ids), past_key_values=None, position_ids=row["positions"])
        del embeddings
        for i in range(config.num_hidden_layers):
            with torch.device("meta"):
                layer = Qwen3_5DecoderLayer(config, i)
            lp = f"layers.{i}."
            state = {k: weight(lp + k).float() for k in layer.state_dict()}
            layer.load_state_dict(state, assign=True)
            del state
            for name, module in list(layer.named_modules()):
                key = f"base_model.model.{lp}{name}"
                if key + ".lora_A.weight" in adapter:
                    parent, _, attr = name.rpartition(".")
                    setattr(layer.get_submodule(parent), attr, SeparateLora(module, adapter[key + ".lora_A.weight"], adapter[key + ".lora_B.weight"]))
            layer.eval()
            for row in rows:
                row["hidden"] = layer(row["hidden"], position_embeddings=row["rotary"], attention_mask=row["mask"] if config.layer_types[i] == "full_attention" else None, position_ids=row["positions"], past_key_values=None, use_cache=False)
            del layer
            print(f"completed layer {i + 1}/{config.num_hidden_layers}", file=sys.stderr, flush=True)
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": weight("norm.weight").float()})
        head = PointerHead(config.hidden_size, dc["pointer_dim"], 0.0)
        head.load_state_dict(load_file(args.model / "head.safetensors"))
        head.eval()
        for row in rows:
            hidden = norm(row["hidden"])
            logits = head(hidden[:, -1], hidden[:, row["encoding"]["option_positions"]])[0]
            rq = row["rq"]
            temperature = dc["temperature_by_kind"].get(rq.kind, dc["temperature"])
            probs = (logits / temperature).log_softmax(dim=-1).exp().tolist()
            answer = {"type": rq.kind}
            if rq.kind == "noul":
                answer["noul"] = read_noul(probs, rq)
            elif rq.kind == "choice":
                distribution = read_choice(probs, rq)
                answer.update(choice=max(distribution, key=distribution.get), probabilities=distribution, confidence=derive_confidence(list(distribution.values())))
            else:
                score, distribution = read_score(probs, rq)
                answer.update(score=score, probabilities=distribution, legend=score_legend(rq), confidence=derive_score_confidence(list(distribution.values()), ordinal_smoothing=dc["ordinal_smoothing"]))
            case = row["case"]
            case["probes"][row["name"]] = {"encoding": row["encoding"], "logits": logits.tolist()}
            case["response"]["answers"][row["name"]] = answer
            case["response"]["usage"]["input_tokens"] += len(row["encoding"]["input_ids"])
    result = {"version": 2, "provenance": {"strands_commit": commit, "source_sha256": SOURCES, "packages": VERSIONS, "revisions": revisions, "method": "CPU FP32, eager attention, stock Transformers layers streamed, separate LoRA, no prefix cache; unrounded Strands readout"}, "cases": cases}
    tmp = args.output.with_suffix(".json.part")
    tmp.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
