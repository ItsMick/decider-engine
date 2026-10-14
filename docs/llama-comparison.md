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

The runner writes complete JSONL samples, process metadata and a summary. It refuses to overwrite an existing output directory. Compare logits with the independent reference as well as between engines; a faster run that misses the numerical gate is not strict correctness parity. The calibrated probability differences are computed with the pinned per-kind temperatures. Argmax equality alone is not calibration validation.

## Limits

- Four cases: three short prompts (choice, noul, score) and one 1,024-token prompt. This is not representative deployment traffic.
- The three short prompts are timed individually, not as a shared-state three-question request. This does not measure either engine's multi-question caching advantage.
- No INT8/INT4 quantization comparison, concurrency, 4,096-token timing, Graviton or Lambda measurement.
- Load timings are diagnostic: llama uses memory mapping, while the Rust loader reads eagerly. Warm filesystem caches and deferred faults prevent interpreting these values as comparable cold starts.
- The harness depends on the pinned llama public API and `result_norm` callback name. It fails if the expected hidden rows are unavailable; update and revalidate it when changing llama revisions.
