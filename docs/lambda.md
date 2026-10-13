# ARM64 Lambda

This library can be embedded in a Rust Lambda handler. The bundled CLI uses
stdin/stdout; it is not a Lambda Runtime API client. Use the official
[`lambda_runtime`](https://github.com/aws/aws-lambda-rust-runtime) crate for the
host and create one `NativeEngine` before its event loop. Keep that engine alive
for warm invocations. Run synchronous inference using `spawn_blocking` if the
host uses Tokio; do not block its async worker with a forward pass.

## Build and package

AWS documents ARM64 Lambda as Graviton2 / Neoverse N1. For a Linux ARM64 build,
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

Lambda deployment and real-weight performance on Graviton have not been tested
by this project. Native ARM64 CI verifies compilation and synthetic numerical
tests; it does not establish Lambda latency or a production memory guarantee.

Sources: [Lambda quotas](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html),
[Lambda ARM architecture](https://docs.aws.amazon.com/lambda/latest/dg/foundation-arch.html),
[Rust Lambda handlers](https://docs.aws.amazon.com/lambda/latest/dg/rust-handler.html),
[Strands inference](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md),
[Strands architecture](https://github.com/strands-labs/strands-decider/blob/main/docs/architecture.md).
