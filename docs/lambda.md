# ARM64 Lambda

This library can be embedded in a Rust Lambda handler. The bundled CLI uses
stdin/stdout; it is not a Lambda Runtime API client. Use the official
[`lambda_runtime`](https://github.com/aws/aws-lambda-rust-runtime) crate for the
host and create one `NativeEngine` before its event loop. Keep that engine alive
for warm invocations. Run synchronous inference using `spawn_blocking` if the
host uses Tokio; do not block its async worker with a forward pass.

## Build and package

AWS documents ARM64 Lambda as Graviton2 / Neoverse N1. Use Rust 1.98 or newer,
which provides the stable `vdotq_s32` intrinsic. For a Linux ARM64 build,
use `aarch64-unknown-linux-gnu` in an Amazon Linux 2023 build environment, or
`aarch64-unknown-linux-musl` for a static binary. The tokenizer needs a C compiler
for that same target. Host glibc builds from a newer distribution may be
incompatible with Amazon Linux; build for the deployment environment.

```sh
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS='-C target-cpu=neoverse-n1' \
  cargo build --release --locked --target aarch64-unknown-linux-gnu
```

This command builds the library and CLI when the target toolchain is installed;
your Lambda host also needs to be compiled for ARM64 and packaged as `bootstrap`.
The target CPU flag is appropriate for Lambda's documented processor. Use a
generic CPU target for binaries that must run on other ARM64 processors. Rust
FP32 `mul_add` enables native fused multiply-add on AArch64; numerical results can
differ in their last digits. CI exercises the Neoverse N1 target on Linux ARM64
and the portable target on Apple ARM64, with synthetic parity fixtures.

The FP32 dot loop already compiles to NEON fused multiply-add instructions with
this target. Additional forward-pass optimizations reuse position-dependent
RoPE sine/cosine values across heads and layers, cache constant decay rates at
load, and reuse normalization and attention-score buffers. Causal attention
parallelizes independent token rows for at least 32 tokens and a query width of
at least 128, when the Rayon pool has multiple workers. The order of sums within
each token is preserved. Scratch memory belongs to the forward pass or Rayon
job; no request state or answer is cached on the model.

`--precision int8-compute` / `Precision::Int8Compute` enables dynamic INT8
projection inputs and INT32 dot products for base projections. Graviton2 supports
NEON `SDOT` and the engine detects it at runtime. The existing `int8` mode is
weight-only and still computes in FP32. INT4 requires unpacking; there is no
native INT4 dot-product path here. Graviton2 also lacks native BF16 arithmetic;
do not assume Graviton3/4 BF16 or SVE optimizations apply to Lambda.
See the [AWS SIMD guide](https://aws.github.io/graviton/SIMD_and_vectorization.html)
and [processor comparison](https://aws.github.io/graviton/).

Integer inputs use symmetric groups of 64, quantized once per projection and
reused across output rows. Group accumulators are scaled back to FP32 before
summing. LoRA, attention, recurrent state and the pointer head stay FP32.
This adds input-rounding error beyond weight-only INT8 and requires separate
decision-quality and calibration validation. It is not a full-model W8A8 or
SmoothQuant implementation. Unsupported CPUs use a scalar integer fallback;
non-finite inputs use the weight-only FP32 path.

To compare synthetic 2048-by-2048 projections at 1, 32 and 128 input tokens:

```sh
RAYON_NUM_THREADS=4 cargo test --release --locked --lib benchmark_int8_projections -- --ignored --nocapture
```

ARM CI runs this comparison, including input quantization and weight-only
dequantization in the timed operation. It is not a full-model or Lambda benchmark.

The full checkpoint download is approximately 4.5 GB and will not fit Lambda's
250 MB unzipped ZIP/layer package limit. A container image can hold the model and
its licenses within the 10 GB uncompressed image limit. Alternatively provision
model files separately; the engine does not download them. Avoid downloading or
quantizing a new model for every request. The native precision choice is applied
once at engine load; prequantized export files are not currently supported.

## Size and validate

Start with the 10,240 MB memory setting when profiling. Lambda allocates CPU
proportionally to memory, so a smaller memory setting can increase latency even
when the compressed weights fit. Compare 6,144, 8,192, and 10,240 MB with actual
requests, recording cold initialization, warm latency, maximum memory, and cost
per invocation. Set Rayon worker count to the CPU allocation; avoid extra
parallel requests inside one execution environment until measured.

Bound request bytes, question count, and total tokens in the host. A useful
initial experiment is 1–4 questions with at most 512 tokens per bound prompt;
use `engine.encode` to check the token count before forwarding. This is a
starting point, not the model's context limit. Increase it after measuring.
Activation memory and duration grow with the prompt length. Questions currently
repeat the state forward; no shared-prefix cache is implemented.

Checkpoint mode preserves the published weights and provides the parity
baseline. INT8 is a conservative first compression experiment; INT4 saves more
memory but has more rounding error. These choices are this implementation's
recommendations. Strands documents FP32 CPU inference and an FP32 pointer head;
its serving guide does not specify an INT4/INT8 preset for Lambda. Keep the head
and adapters unquantized. Evaluate decision accuracy and calibration on a
representative held-out workload before enabling either quantized mode.

AWS's broader [quantization guide](https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/)
describes weight-only INT8 as a practical starting point, and discusses AWQ/GPTQ
and asymmetric 4-bit mapping for more aggressive compression. That article
targets general LLM inference on SageMaker, rather than this checkpoint on
Lambda. The native INT4 mode here uses simple symmetric rounding, without
activation-aware calibration or GPTQ error compensation. It should not be
treated as equivalent to those methods or as an AWS-endorsed preset.

## ARM kernel check

[CI run 38390317785](https://github.com/ItsMick/decider-engine/actions/runs/38390317785)
executed the accelerated kernel on a Linux ARM64 runner with `SDOT` detected,
four Rayon workers, Rust 1.98 and `-C target-cpu=neoverse-n1`. The v0.2.0
kernel uses the stable `vdotq_s32` intrinsic. A warmed synthetic 2048-by-2048
projection averaged three timed runs per mode:

| Input tokens | INT8 weights, FP32 compute (ms) | INT8 compute (ms) |
| ---: | ---: | ---: |
| 1 | 1.219 | 0.097 |
| 32 | 4.432 | 1.866 |
| 128 | 21.828 | 7.220 |

Input quantization and weight dequantization are included. The 32/128-token
projections were about 2.4/3.0 times faster with integer computation. This is
one synthetic CI measurement on an ARM runner, not measured on Graviton2.
It excludes model loading, other operators, real checkpoint weights and Lambda
overhead. It does not establish an end-to-end speedup or cost improvement.

## Forward-pass checks

[CI run 38392563538](https://github.com/ItsMick/decider-engine/actions/runs/38392563538)
compared the same causal attention implementation with one and four Rayon
workers on Linux ARM64, Rust 1.98, Neoverse N1 compiler flags. Synthetic inputs
used eight query heads, two key/value heads and head dimension 256. Each result
is the mean of three warmed operations, including output and score allocation:

| Tokens | One worker (ms) | Four workers (ms) |
| ---: | ---: | ---: |
| 32 | 0.352 | 0.181 |
| 128 | 5.496 | 1.577 |
| 512 | 103.314 | 44.315 |

The 128-token attention operator was about 3.5 times faster with four workers.
This measures additional CPU parallelism within attention, not the complete
forward pass. It is not a Lambda measurement or a one-core speedup.

A separate local full-model comparison of v0.2.0 and v0.2.1 used Rust 1.98,
Linux x86_64, six Rayon workers and the pinned v21 checkpoint. Each binary/mode
ran in a separate resident process: initialization and the first request were
excluded, followed by three repeated requests using the three questions in
`tests/request.json` (261 bound tokens in total). Warm medians were:

| Mode | v0.2.0 (s) | v0.2.1 (s) |
| --- | ---: | ---: |
| Checkpoint | 7.517 | 7.378 |
| INT8 weights | 7.988 | 7.931 |
| INT8 compute, portable integer fallback | 15.081 | 14.840 |
| INT4 weights | 8.189 | 8.075 |

Every response, including probabilities and confidence, was exactly unchanged
within each precision mode. Improvements on this short sample were only about
0.7–1.9%; three repeated requests on one workload do not establish a general
speedup. New buffers are scoped to a forward pass or Rayon job. The published
model's six full-attention layers and ten query/key heads share each rotary
sine/cosine pair, reducing those calculations by a factor of 60. Quantization,
temperatures and the order of attention reductions are unchanged.

## Local memory check

One run per mode, Linux x86_64, six Rayon workers, the pinned v21 checkpoint,
and the three typed questions in `tests/request.json`. Peak process RSS includes
loading and inference. Elapsed time includes model load and all three answers.

| Mode | Peak RSS (GiB) | Total seconds | Largest probability shift |
| --- | ---: | ---: | ---: |
| v0.1.0, expanded FP32 storage | 11.39 | 24.38 | baseline |
| Checkpoint storage, FP32 computation | 3.66 | 28.84 | 0 |
| INT8 base projections | 2.49 | 19.51 | 0.00496 |
| INT4 base projections | 1.85 | 21.34 | 0.04932 |

Checkpoint mode returned identical answers and probabilities to the previous
runtime on this sample. The categorical choice remained the same in both
quantized modes, while ordinal scores and probabilities changed. Three questions
are a smoke check, not a quality benchmark or calibration validation. These
numbers are not ARM64 or Lambda measurements, and one run is insufficient for
a reliable latency comparison. Lambda memory accounting can differ from RSS.

A separate v0.1.2 full-model `int8-compute` smoke check on Linux x86_64 used
the portable integer fallback, six workers and the same three questions.
Peak RSS was 2.49 GiB; the largest probability shift from checkpoint precision
was 0.01273 (about 1.3 percentage points), versus 0.00496 for weight-only INT8.
The categorical choice remained the same. This small sample does not validate
decision quality or calibration; the original checkpoint temperatures remain
unchanged.

Lambda deployment and real-weight performance on Graviton have not been tested
by this project. Native ARM64 CI verifies compilation and synthetic numerical
tests; it does not establish Lambda latency or a production memory guarantee.

Sources: [Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html),
[Lambda ARM architecture](https://docs.aws.amazon.com/lambda/latest/dg/foundation-arch.html),
[Rust Lambda handlers](https://docs.aws.amazon.com/lambda/latest/dg/rust-handler.html),
[Strands inference](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md),
[Strands architecture](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md).
