# Validation and remaining evidence

## Checkpoint reference parity

The `Pinned checkpoint parity` workflow runs explicitly through `workflow_dispatch`
or on `v*` tags. It downloads the pinned v21 checkpoint with the existing checksum
verifier, then runs checkpoint-precision reference parity on x86_64 and ARM64.
Normal PRs continue to use the small synthetic fixtures. Logs and machine,
toolchain, source, model revisions and fixture hashes are retained as artifacts;
model weights are never uploaded.

Reproduce locally:

```sh
python3 scripts/download-model.py models/v21
RAYON_NUM_THREADS=4 DECIDER_TEST_MODEL_DIR=models/v21 \
  cargo test --release --locked --test model -- --ignored --nocapture
```

The existing reference includes three questions, one per kind. Exact token IDs,
option/answer positions, answer domains, token usage and choice are checked;
absolute logit tolerance is `1e-4`, probability/confidence/score tolerance `1e-5`.
Quantized modes do not participate in this reference parity claim.

Release procedure: require both architecture jobs to succeed for the exact tag
commit before publishing or describing that release as validated. The workflow
does **not** publish releases or prevent someone with repository permission from
manually publishing. Automated release publication depending on these jobs, and
an independently generated expanded reference matrix (long/near-4096 contexts,
option permutations and question counts), are still outstanding under #2. A
green three-question job does not satisfy those missing acceptance criteria.

## Adversarial and precision smoke evaluation

`tests/fixtures/adversarial.json` is a versioned, public, hand-written 12-case
suite. It covers injected state/question/answer delimiters, fake options,
Unicode confusables, long separators, conflicting instructions, option reversal,
and hostile instructions, descriptions and labels. It is **not** a representative
held-out deployment corpus. Nine cases carry deliberately simple semantic labels;
the other three exercise structural handling without claiming an intended answer.

See the [2026-10-14 v21 report](reports/adversarial-v1.md) for measured outputs,
probability drift and the remaining acceptance criteria for each issue.

Default tests exercise all four precision modes with synthetic weights. They
check deterministic binding, byte-span/option ordering, legal answer domains,
finite probabilities and token-window admission. These are software properties,
not proof that the trained model follows the intended instructions.

Run the real model separately, one precision at a time to avoid concurrent model
loads (approximately 4.5 GB download; packed memory depends on precision):

```sh
cargo build --release --locked --example evaluate
mkdir -p evaluation
for mode in checkpoint int8 int8-compute int4; do
  RAYON_NUM_THREADS=4 target/release/examples/evaluate \
    models/v21 tests/fixtures/adversarial.json "$mode" > "evaluation/$mode.json"
done
python3 scripts/summarize-evaluation.py tests/fixtures/adversarial.json \
  evaluation/checkpoint.json evaluation/int8.json \
  evaluation/int8-compute.json evaluation/int4.json > evaluation/summary.json
```

The summary separates semantic mismatches from quantization-induced flips and
reports absolute/relative probability drift, expected-score drift, Brier score,
NLL, ECE and illustrative probability-threshold crossings at 0.8/0.9/0.95. Relative
drift uses a `1e-6` denominator floor. Brier sums over classes. ECE uses ten equal
width bins. The threshold statistics use maximum category probability, not the
upstream `confidence` field. `confident_wrong_decisions` and
`correct_but_below_threshold` are toy abstention-policy counts; they are not
deployment false-accept/false-reject rates. Scores have no semantic labels here.
Single elapsed-time samples are included for diagnostics, not performance claims.

All modes remain **unvalidated for consequential automated decisions**. No
temperatures, acceptance thresholds, or prompt templates are changed. Before
using them for acceptance, define a representative held-out workload, labeled
outcomes, separate calibration/validation splits, and an explicit cost of false
acceptance/rejection. Choose gates before running the evaluation. Version any
calibration with the model revision and precision. Marginal or out-of-distribution
cases must be eligible for escalation rather than forced acceptance. This smoke
evaluation is groundwork for #3, not completion of its quality/calibration gate.

## Semantic trust limits

Prompt tags are ordinary model input. A state can imitate instructions, tags or
options and influence the model even when the runtime preserves correct option
binding. Typed answers constrain the output domain; they do not establish that
an answer is true or safe. A high probability can accompany a wrong answer.

The renderer deliberately keeps the upstream-compatible template. Escaping tags
would change tokenization and model behavior, and would not remove conflicting
natural-language instructions. There is no measured evidence here that escaping
improves robustness. Any future encoding/template experiment needs its own
reference and semantic evaluation before adoption. Downstream systems must retain
their own authorization rules and workload-specific escalation policy.

## Hardware work still pending

Issues #6 and #7 require full-model measurements on the intended ARM/Lambda host.
This change does not introduce an unmeasured GEMM rewrite, deploy AWS resources,
or claim a Graviton speedup. The evaluation executable reuses one engine across
requests and records load time separately, so it can assist later experiments.
It does not replace a Lambda host, cold/warm sampling harness, operator/allocation
profiling, cost measurement, or real SDOT dispatch verification on deployment.
