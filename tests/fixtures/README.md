# Reference fixture provenance

Generated 2026-10-12 using Transformers 5.16.1
and PyTorch 2.14.0 on CPU FP32, eager full attention, and the upstream Strands
`PointerHead`, prompting and confidence formulas. Tiny checkpoint seed: 4242.
The tiny model has one Gated DeltaNet and one full attention layer, nontrivial
normalization weights, and LoRA on all torso projections. Its weights are
synthetic; no trained model weights are bundled in the repository.

`tiny/reference.json` covers the public request in `tests/request.json`.
`tiny/raw-reference.json` covers independent token sequences, including two
sequences with matching prefixes and different future tokens. Cargo tests compare
hidden states, logits, token positions, probabilities, and derived answers.

`v21-reference.json` was produced with:

- Strands export `StrandsAgents/strands-decider-2B-hobson-v21`, revision
  `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`.
- Base `Qwen/Qwen3.5-2B-Base`, revision
  `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
- Separate FP32 LoRA projections, scale `alpha/r`, with no merge or quantization.
- Strands' 4096-token strict window; no state/question truncation or prefix cache.

The full-weight test is ignored unless explicitly requested; it never downloads
anything. See the project README for the invocation.

SHA256 of the upstream Strands source files used by the reference generator:

```text
944394834207844dc264b2546d80d8cd2c8f86d70a78b09c4fadda87a8b9cb21  modeling.py
5ac5b772795dfa5790504b4956d73abdcd09862cd0603dcda4a1df660808faec  prompting.py
a4b0e9081064f3be9e15fced49328ca78c3acc9a86183b3bd77fd282684748a8  schema.py
```

## Expanded v21 oracle (version 2)

`v21-expanded-reference.json` adds reversed choice options, an empty-state short
noul question, a 1,024-token context, and the exact 4,096-token boundary. The
original three-question request covers all kinds and shared-state question
ordering. The Rust release test also rejects the same boundary request with one
extra token, and retains the original reference checks. This is a deterministic
numerical conformance corpus, not held-out decision-quality data.

The generator is now checked in. It uses unmodified Transformers decoder layers,
FP32 arithmetic, eager causal attention, and separate LoRA projections. It loads
one layer at a time to keep memory bounded and uses the upstream Strands pointer
head and unrounded readout formulas. It does not call the Rust engine. Upstream
source hashes, package versions, model revisions and the exact prompts are in
the fixture. The original fixture's logits reproduced within `6e-7` locally.

```sh
python3 -m venv reference-venv
reference-venv/bin/pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
reference-venv/bin/pip install -r scripts/reference-requirements.txt
git clone https://github.com/strands-labs/strands-decider.git strands-reference
git -C strands-reference checkout 8b6faa553038a59e73fa7f16c4081f8ceb21d27e
python3 scripts/download-model.py models/v21
reference-venv/bin/python scripts/generate-reference.py models/v21 strands-reference \
  tests/fixtures/v21-expanded-reference.json
```

Fixture regeneration is deliberate and reviewed, never part of the Rust parity
job. The generator rejects changed source hashes, revisions or core package
versions. Run on CPU with about 8 GiB available memory; generation and the native
4,096-token forward can take minutes. The independent Python dependencies do not
enter the engine, its release package, or normal CI.
