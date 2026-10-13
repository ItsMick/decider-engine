# decider-engine

Minimal native Rust CPU runtime for [Strands Decider 2B v21](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21).
Use it as a library or a small JSON/JSONL command. Supports boolean (`noul`),
categorical (`choice`), and ordinal (`score`) decisions with probabilities and confidence.

Implements the Qwen3.5 text decoder, Gated DeltaNet, causal attention, LoRA,
pointer head, prompt binding, and calibration. No Python, PyTorch, ONNX,
HTTP server, or model downloads during inference. Public APIs use checked Rust
slices. Unsafe code is denied except in a small ARM integer kernel with checked
bounds and CPU feature detection; dependencies have their own implementations
and licenses.

## Run

Requires Rust 1.88+ and a C compiler for the tokenizer dependency.
The optional download helper needs Python 3.11+ and `curl`.

```sh
git clone https://github.com/ItsMick/decider-engine.git
cd decider-engine
python3 scripts/download-model.py models/v21
cargo build --release --locked
target/release/decider-engine models/v21 tests/request.json
```

The downloader pins the adapter and base revisions and checks the published
SHA256 hashes of weight files. It also downloads their license files.
It downloads approximately 4.5 GB; allow 8 GB available RAM for initial loading
and inference. Checkpoint weights retain their original BF16/F16/F32 storage;
checkpoint arithmetic, adapters, normalization, recurrent state, and pointer
readout use FP32.
The v21 model occupies roughly 4 GB resident memory. Loading streams tensors
through a bounded buffer instead of holding a second copy of the checkpoint.

With no request file, stdin accepts one JSON request per line and stdout returns
one response per line. The model stays loaded until EOF. Blank lines are skipped;
invalid requests stop the process with an error on stderr and a nonzero exit status.
The CLI limits each request to 2 MiB.

```json
{"state":"A payment failed three times.","questions":{"urgent":{"type":"noul","instructions":"Does this need attention?"}}}
```

`noul` returns the probability that the statement is true. For `choice`, supply
`criteria` as an ordered JSON object mapping option labels to descriptions or
`null`. For `score`, supply an array of 2–10 descriptions from lowest to highest;
the answer is an expected zero-based level. See [tests/request.json](tests/request.json).
State and instructions accept strings, objects, or arrays. Option order is preserved.

## Library

```rust,no_run
use decider_engine::{DecisionEngine, NativeEngine, Request};
use std::path::Path;

# fn main() -> Result<(), String> {
let engine = NativeEngine::load(Path::new("models/v21"))?;
let request: Request = serde_json::from_str(r#"{
    "state": "A payment failed three times.",
    "questions": {"urgent": {"type": "noul", "instructions": "Does this need attention?"}}
}"#).map_err(|e| e.to_string())?;
let response = engine.decide(&request)?;
println!("{}", serde_json::to_string(&response).map_err(|e| e.to_string())?);
# Ok(())
# }
```

For a Git dependency:

```toml
decider-engine = { git = "https://github.com/ItsMick/decider-engine", tag = "v0.1.2" }
```

## Scope and checks

Text only, CPU only, no generation or prefix/KV cache. Each
question gets a separate forward pass. Prompts exceeding the checkpoint's
4096-token window are rejected without truncation. Use trusted checkpoints;
the loader validates tensor shapes and uses safetensors, but does not sandbox
malicious files or enforce a process memory/time budget. Library callers should
bound request sizes and question counts. Decisions and confidence can be wrong.

```sh
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
DECIDER_TEST_MODEL_DIR=models/v21 cargo test --release --locked --test model -- --ignored
```

Default tests use tiny synthetic weights and never download a model. They
compare hidden states, token positions, logits, and typed answers against a
PyTorch reference, and check causal isolation and invalid inputs. The optional
full-weight test compares three questions against the pinned v21 reference;
this is a parity check, not a benchmark of decision quality.
See [fixture provenance](tests/fixtures/README.md).
Rayon controls parallelism through its standard `RAYON_NUM_THREADS` setting.

## Lower precision and ARM64 Lambda

```sh
target/release/decider-engine --precision int8 models/v21 tests/request.json
target/release/decider-engine --precision int8-compute models/v21 tests/request.json
target/release/decider-engine --precision int4 models/v21 tests/request.json
```

The library equivalents are `NativeEngine::load_with_precision(path, Precision::Int8)`
and `Precision::Int4`; `Precision::Int8Compute` enables integer computation.
Default `load` preserves checkpoint precision.
INT8 and INT4 quantize only the base linear projections at load, using symmetric
groups of 64 weights within each output row, FP32 scales, and signed ranges
[-127, 127] or [-7, 7]. INT4 packs two values per byte. Embeddings retain checkpoint
precision; adapters, norms, and the pointer head receive no added quantization.
In weight-only `int8` and `int4`, rows expand to FP32 in reusable worker buffers
before multiplication. GGUF, AWQ, and GPTQ files are not supported.

Opt-in `int8-compute` uses the same INT8 weights and additionally quantizes base
projection inputs dynamically, with one symmetric FP32 scale per token and
64-element group. Input quantization is reused across output rows. Integer dot
products accumulate in INT32; group results are scaled and summed in FP32.
On little-endian ARM64, runtime feature detection selects NEON `SDOT`; other
CPUs use a portable integer fallback. Non-finite projection inputs fall back to
the weight-only FP32 path. LoRA, attention, recurrent state and the pointer head
are not integer-quantized. This is an experimental compute mode, not calibrated
W8A8 across the whole model or SmoothQuant.

Responses identify the mode in `model`, with an `-int8`, `-int8-compute` or `-int4`
suffix. All are experimental: quantization changes probabilities and can invalidate the
checkpoint's calibration. Original temperatures are retained; validate accuracy
and refit calibration for your workload before relying on confidence thresholds.
Weight compression primarily saves memory. Integer computation requires
benchmarking on the deployment CPU; no Lambda speedup is established.

The checked ARM kernel in `src/int8.rs` uses bounded 16-byte loads and a
register-only `SDOT` instruction. Inline assembly preserves Rust 1.88 support;
the compiler's SDOT intrinsic was not stable at that version. Each dot product
is limited to 64 bytes, preventing INT32 overflow even for -128 inputs. ARM CI
requires dot-product support and compares the accelerated and scalar results.

See [ARM64 Lambda notes](docs/lambda.md) for build targets, sizing, and limitations.

## License

The original Rust implementation and synthetic fixtures are [MIT](LICENSE).
The separately downloaded [Strands checkpoint](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)
and [Qwen base](https://huggingface.co/Qwen/Qwen3.5-2B-Base) are Apache-2.0;
their licenses remain applicable. Dependencies retain their own licenses.
No trained weights or upstream Python source are included. This is an independent
implementation compatible with Strands Decider, with no upstream affiliation.
