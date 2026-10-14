# Opt-in Graviton Lambda experiment

This is a private benchmark host, not a deployed service. It uses the official
`lambda_runtime` client, loads one `NativeEngine` before the event loop, and keeps
it for warm invocations. A semaphore follows the blocking inference task even if
its async caller is cancelled. There is no public function URL, model download,
request cache, or per-request quantization. The separate crate/lockfile keeps
Lambda/Tokio dependencies out of the engine library.

The transport middleware rejects events over 256 KiB before JSON deserialization,
including outer whitespace. The SDK has already buffered the provider's event;
this is not a streaming network-body limit. Every request then uses the library's
Lambda limits (3 questions, 32 options, 64 KiB canonical state, 2,048 tokens per
question and 4,096 total). The complete request is preflighted before inference.

Responses contain the typed engine response plus measurements, request ID,
environment ID, invocation sequence, one-time model load duration, precision,
worker count, model revisions and process peak RSS. `integer_kernel()` reports
the runtime dispatcher also used by integer projections. `DECIDER_REQUIRE_SDOT=1`
refuses a host without ARM SDOT. This verifies feature selection, not a measured
speedup; non-finite projection inputs can still take the documented FP32 fallback.

## Local verification

```sh
cargo test --locked --manifest-path lambda/Cargo.toml
cargo build --locked --manifest-path lambda/Cargo.toml
DECIDER_LAMBDA_BIN=lambda/target/debug/decider-lambda \
  python3 -m unittest discover -s scripts -p 'test_lambda_runtime.py'
```

The integration test runs the actual Runtime API client against a local test
server with the synthetic checkpoint. It proves cold/warm engine reuse and
rejects an oversized whitespace-padded body without poisoning the next request.
CI runs this on Linux x86_64 and ARM64. It does not emulate AWS billing, memory
allocation, network cold starts, permissions, or the Graviton2 microarchitecture.

## Build and deliberately deploy

Needs Docker/buildx on ARM64 (or an explicitly configured ARM64 builder), AWS CLI
v2, and an existing private ECR repository. Commands below create billable AWS
resources only when you run them. Use a disposable account/stack appropriate to
your experiment. ECR image storage outlives the CloudFormation stack.

First download and verify the exact pinned checkpoint and its licenses. This is
about 4.5 GB; keep sufficient room for the build context and image. The image
includes the model, licenses, manifest, provenance and revisions. Quantization
occurs once during initialization. Do not put credentials or private data in the
model directory or image.

```sh
python3 scripts/download-model.py models/v21
# Set ECR_REPOSITORY to your repository URI and AWS_REGION to its region.
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${ECR_REPOSITORY%%/*}"
docker buildx build --platform linux/arm64 --provenance=false \
  -f lambda/Dockerfile -t "$ECR_REPOSITORY:decider-v21" --push .
# Resolve the resulting image digest in ECR; pass URI@sha256:..., never a mutable tag.
aws cloudformation deploy --stack-name decider-benchmark \
  --template-file lambda/template.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides ImageUri="$IMAGE_DIGEST_URI" \
  --tags decider-purpose=disposable-benchmark
```

The build uses Rust 1.98 and `-C target-cpu=neoverse-n1` in Amazon Linux 2023;
the runtime uses the matching Lambda AL2023 base. The final image digest binds
the binary and model even though base-image tags can change between builds.
Record build logs, source commit, Docker version, base-image digests and Rust
version with your experiment. Build native ARM images: running an emulated x86
model is not a valid Graviton benchmark. This container recipe has not yet been
built or deployed by this project.

The stack grants only basic Lambda logging to the execution role, reserves
concurrency at one, keeps logs for one day, and starts at 10,240 MB. It does not
grant the function permission to download model files or contact another AWS
service. Your deployment identity separately needs ECR, CloudFormation, Lambda,
IAM and CloudWatch Logs permissions. New accounts may need a concurrency quota
increase before a reserved-concurrency function can be created.

## Measure

The runner requires a three-question request and also measures its first question
alone. It compares checkpoint, INT8 weights, and INT8 compute at 6,144, 8,192 and
10,240 MB. Every worker count supplied is measured explicitly. Use workload data
you are authorized to send to the selected AWS account. Results contain answers
and probabilities, but the input text is represented only by SHA256.

Supply current regional ARM GB-second and per-request prices yourself; they are
not hardcoded. The following shows argument structure, not actual prices:

```sh
python3 lambda/benchmark.py decider-benchmark tests/request.json \
  --workers 4 --samples 100 --cold-samples 5 \
  --gb-second-price "$ARM_GB_SECOND_PRICE" --request-price "$REQUEST_PRICE"
# Review the plan, then repeat with --execute --output lambda/results/run-001.json.
```

The default plan has 18 workload/configuration combinations and 1,890 invocations;
one resident environment per combination supplies 100 warm samples, and five
fresh environments supply cold samples. Additional worker counts multiply the
experiment. Initialization is expensive: consider its total cost before running.
The runner never deploys or deletes resources. It refuses stacks/functions without
the disposable benchmark tag, requires a digest-pinned ARM64 image, updates only
the stack's memory/worker/precision/nonce parameters, then restores the original
parameters in `finally`. Evidence is saved atomically after every invocation. If
restoration fails, the report says `restored: false`; inspect and remove the stack.

Cold classification uses the resident host sequence, not the assumption that the
first client request was cold. Every reset must produce a cold environment. Warm
samples exclude cold invocations, including unexpected environment replacement.
Report latency is Lambda's `Duration`; client wall time and host inference time
are also retained separately. Cold `Init Duration`, one-time model load, maximum
Lambda memory and process peak RSS are distinct quantities. Cost uses **Billed
Duration**, so it does not add initialization a second time. Storage, logs, data
transfer, free-tier credits and negotiated discounts are excluded.

Quantiles use nearest rank. p95 requires at least 20 warm samples; p99 at least
100, and the runner insists on 100. At that count p99 is a rough tail estimate;
use more samples and repeated experiments for a consequential comparison. Cold
init durations are retained individually rather than advertising a five-sample
p99. Responses must remain exactly deterministic within a configuration; the
summary also measures answer flips, probability drift and expected-score drift
against checkpoint at the same memory/worker/workload setting. This request is
only a performance/parity probe, not a quality or calibration corpus.

Preserve raw JSON, its SHA256, source/image/model revisions, region, runtime,
compiler, worker count, memory and pricing date in a versioned report. Include
all failures and restored state. Do not substitute synthetic CI timings for AWS
measurements. #7 remains open until this experiment produces real evidence.

## Teardown

```sh
aws cloudformation delete-stack --stack-name decider-benchmark
aws cloudformation wait stack-delete-complete --stack-name decider-benchmark
# Remove only your experiment image from ECR separately, if no longer needed.
```

The stack deletes its function, IAM role and log group. Verify teardown and keep
the local report. ECR repository/image creation and deletion are deliberately
outside the runner so it cannot remove an unrelated repository.

Sources: [Rust handlers](https://docs.aws.amazon.com/lambda/latest/dg/rust-handler.html),
[Runtime API](https://docs.aws.amazon.com/lambda/latest/dg/runtimes-api.html),
[ARM architecture](https://docs.aws.amazon.com/lambda/latest/dg/foundation-arch.html).
