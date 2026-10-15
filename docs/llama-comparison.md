# Comparing with llama.cpp

This is a CPU, single-question, backbone-plus-pointer-head comparison. It is not a token-generation benchmark. Both clients consume the exact same pinned v21 token IDs and option positions from the independent reference fixtures. Neither client tokenizes or serves HTTP inside the timed region.

The first measured results are in the [2026-10-14 x86-64 report](reports/llama-comparison-2026-10-14.md).

The native client uses checkpoint precision and `raw_logits`. The llama client uses the pinned Qwen base converted to GGUF, the same separately applied FP32 LoRA adapter at scale 1 (the GGUF adapter carries alpha/r), and the same FP32 pointer-head weights. It implements the head's LayerNorm, Q/K projections and scaled dot product. It uses F32 attention caches. The default llama GGML CPU kernels retain their own arithmetic; identical storage does not imply identical logits.

The llama client sets LAST embedding pooling to suppress the unrelated vocabulary head. A public evaluation callback reads the selected `result_norm` rows before pooling, then computes the pointer head. It does not modify llama.cpp. The callback and final pooling overhead are included. Each question starts with cleared recurrent and attention state; neither side gets cross-question or cross-repetition prefix caching. Model loading is outside warm timings. One warmup is excluded per case; the remaining samples are reported as medians, not latency quantiles.

## Reproduce on Linux

Use a machine with enough memory for the chosen storage format. BF16 GGUF needs about 4 GiB resident weights; F32 needs about twice that. Do not run the two clients simultaneously. Avoid other compute jobs during timing.

From the repository root, with Python, a C++ compiler, CMake and Rust 1.98 installed:

```sh
python3 scripts/download-model.py models/v21
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
git -C ../llama.cpp checkout 6868a52cb337450658006ad2f9281c2eb63f1666
python3 -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install transformers==5.19.0 numpy safetensors sentencepiece
cp models/v21/tokenizer*.json models/v21/base/
python3 ../llama.cpp/convert_hf_to_gguf.py models/v21/base --outtype bf16 --outfile models/base-bf16.gguf
python3 ../llama.cpp/convert_lora_to_gguf.py --base models/v21/base --outtype f32 --outfile models/lora-f32.gguf models/v21/lora
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build --target llama -j 4
g++ -O3 -std=c++17 -I ../llama.cpp/include -I ../llama.cpp/ggml/include -I ../llama.cpp/vendor scripts/llama-compare.cpp -L ../llama.cpp/build/bin -Wl,-rpath,"$(realpath ../llama.cpp/build/bin)" -lllama -lggml -lggml-base -o /tmp/llama-compare
RUSTFLAGS='-C target-cpu=native' cargo build --release --locked --example compare_logits
python3 scripts/prepare-llama-comparison.py models/v21 /tmp/decider-comparison
python3 scripts/run-llama-comparison.py --native target/release/examples/compare_logits --llama /tmp/llama-compare --model models/v21 --gguf models/base-bf16.gguf --lora models/lora-f32.gguf --prepared /tmp/decider-comparison --output /tmp/decider-results --threads 4 --repeats 3
```

For an F32 comparison, change the base converter's `--outtype` to `f32` and use a different GGUF output filename. F32 conversion expands the original BF16 values exactly. LoRA stays separate in both cases. The prepared files contain weights and should stay outside source control.

On N1-compatible ARM builds, use F16 and F32 controls. At the pinned llama revision, `ggml_vec_dot_bf16` has no ARM SIMD branch and the BF16 `llamafile_sgemm` case has no ARM kernel, so its fallback is an unrepresentative primary speed target. F16 is a practical low-memory control, but conversion and activation arithmetic can introduce numerical differences; F32 expands the original values exactly. Record both output differences and performance for each format.

The runner writes complete JSONL samples, process metadata and a summary. It refuses to overwrite an existing output directory. Compare logits with the independent reference as well as between engines; a faster run that misses the numerical gate is not strict correctness parity. The calibrated probability differences are computed with the pinned per-kind temperatures. Argmax equality alone is not calibration validation.

## Cache-blocking experiment on ARM64

The `cache-blocking` feature is opt-in. For projections with at least 64 rows, 256 input columns and 24 output columns, it computes a 24-row activation block against 24 output columns before advancing to the next row block. It keeps each original dot product's arithmetic order. The INT8-compute kernel, small projections and default build keep their existing selection.

The converted weight block needs at most `24 * input_width * 4` bytes per active worker, with reusable worker scratch. This increases temporary conversion storage compared with the original one/three-column tasks, while avoiding a full-model FP32 weight expansion. The sweep compares several row/column block sizes to expose shape-dependent regressions.

The first ARM experiment compares this feature with a pinned main source revision and pinned llama.cpp. The later GEMM experiment below uses the same probes and baseline.

The cache-blocked candidate must keep native logits identical and within the independent `1e-4` logit tolerance. The feature should remain opt-in unless full-model measurements justify a default change.

## Optional Rust GEMM backend

The `fast-gemm` feature uses `ndarray`'s safe matrix views and Rust matrix-multiplication backend for projections with at least 64 token rows, 1,024 input columns and 128 output columns. Input/output views borrow existing buffers. Each Rayon task converts at most 128 adjacent weight rows into reusable scratch; model weights stay in their original storage. Bias, LoRA and epilogues retain their existing formulas. The INT8-compute kernel and smaller projections retain their current paths. No new unsafe code is introduced in this crate.

This path changes the base projection's FP32 reduction order. It must pass the original full-checkpoint test, including 4,096 tokens and the unchanged `1e-4` logit / `1e-5` probability, confidence and score tolerances. Per-projection tests check packed formats, tails, bias/LoRA and one/four workers against the original kernel with an explicit rounding bound. No bit-identity claim is made for GEMM.

The **Matched CPU performance** workflow compares pinned main, this feature and llama.cpp F16/F32 serially on `ubuntu-24.04-arm`. Both native builds target Neoverse N1; llama uses ARMv8.2+FP16+dotprod tuned for N1, without BF16/I8MM/SVE instructions. The instruction-set restriction does not emulate Graviton2's cache, clock or memory system.

The workflow retains raw JSONL samples, RSS, source/model hashes, the full-checkpoint numerical test log, summaries and an instrumented real-model profile as a 30-day artifact. Baseline/candidate rounding differences are reported, while reference tolerances remain fixed. The optional backend must remain experimental until its correctness and full-model performance have both been measured. These experiments do not deploy AWS resources.

## Output layout and real-model profiling

Large projection outputs are transposed and combined in 16-by-16 tiles, with disjoint row blocks written in parallel. A fixed 1 KiB scratch tile reads complete source cache lines before applying the original bias/LoRA and write/add/SwiGLU arithmetic to contiguous destination spans. Small outputs retain the original loop. Boundary, signed-zero, adapter and epilogue tests compare every output bit on one and four workers.

The optional `profile` feature emits `DECIDER_PROFILE` JSON records to stderr. It times projections, GEMM, output conversion, convolution, recurrence, attention and the model body on actual requests. Projection GEMM/write records are children of the projection span; do not add parent and child durations together. The normal benchmark binaries exclude this feature. A separate instrumented run follows the ARM latency measurements, because profiling can perturb timings. No allocation-count or hardware-counter claim is made.

For a local operator trace:

```sh
RUSTFLAGS='-C target-cpu=native' cargo build --release --locked --features fast-gemm,profile --example compare_logits
target/release/examples/compare_logits models/v21 /tmp/decider-comparison/cases.json 1 4 > /tmp/profile-results.jsonl 2> /tmp/profile.log
```

## Limits

- Four cases: three short prompts (choice, noul, score) and one 1,024-token prompt. This is not representative deployment traffic.
- The three short prompts are timed individually, not as a shared-state three-question request. This does not measure either engine's multi-question caching advantage.
- No INT8/INT4 quantization comparison, concurrency, 4,096-token timing, Graviton or Lambda measurement.
- Load timings are diagnostic: llama uses memory mapping, while the Rust loader reads eagerly. Warm filesystem caches and deferred faults prevent interpreting these values as comparable cold starts.
- The harness depends on the pinned llama public API and `result_norm` callback name. It fails if the expected hidden rows are unavailable; update and revalidate it when changing llama revisions.
