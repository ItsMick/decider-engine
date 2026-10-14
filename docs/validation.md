# Validation and remaining evidence

## Checkpoint reference parity

The `Pinned checkpoint parity` workflow runs explicitly through `workflow_dispatch`
or on `v*` tags. It downloads the pinned v21 checkpoint with the existing checksum
verifier, then runs checkpoint-precision reference parity on x86_64 and ARM64.
Normal PRs continue to use the small synthetic fixtures; changes to the oracle
or this workflow also trigger the full matrix. Logs and machine,
toolchain, source, model revisions and fixture hashes are retained as artifacts;
model weights are never uploaded.

Reproduce locally:

```sh
python3 scripts/download-model.py models/v21
RAYON_NUM_THREADS=4 DECIDER_TEST_MODEL_DIR=models/v21 \
  cargo test --release --locked --test model -- --ignored --nocapture
```

The original and independently generated expanded references cover all three
question kinds, one and three questions, reversed options, an empty state,
Unicode, and exact 1,024/4,096-token contexts. A 4,097-token variant is rejected.
Exact token IDs, option/answer positions, answer domains, token usage and choice
are checked; absolute logit tolerance is `1e-4`, probability/confidence/score
tolerance `1e-5`. These tolerances are unchanged from the original reference.
Quantized modes do not participate in checkpoint parity.

The workflow's `publish` job depends on the complete x86_64/ARM64 matrix. On a
version tag it checks that Cargo's version matches, verifies the source package,
and creates a GitHub release with parity evidence and checksums. A failure on
either architecture prevents that automated release. Manual runs validate without
publishing. No crates.io publishing is configured. Repository administrators can
still publish outside this workflow; access policy is outside this change.

Cadence: normal PRs run synthetic tests and the Lambda Runtime API integration.
Run the full matrix manually for changes to kernels, rendering, binding, loading,
or readout; every release tag runs it obligatorily before automated publication.
See [fixture provenance and regeneration](../tests/fixtures/README.md) for the
pinned Python oracle. Expected outputs must never be regenerated from Rust.

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

## Versioned quality regression gates

New `evaluate` reports include exact suite SHA256, model revisions, and the
checkpoint configuration SHA256 (including temperatures). The summarizer rejects
duplicate cases, missing/extra answers, changed domains, invalid distributions,
and inconsistent categorical choices. `expected_answers` supports per-question
labels. Historical smoke reports remain readable, but cannot pass the new gate
without the required identity metadata.

`scripts/check-quality-gates.py POLICY SUITE REPORT...` applies an explicit frozen
policy, with separate limits for every precision including checkpoint. It exits
0 for passing limits, 1 for measured failures, and 2 for invalid/mismatched
inputs. It checks corpus identity, checkpoint/configuration identity, minimum
labeled decision count, flips, probability/score drift, Brier, NLL, ECE, semantic
failures and optional aggregate threshold crossings. The aggregate sums crossings
at 0.8, 0.9 and 0.95, so one decision can contribute more than one crossing.

Example policy structure (fill in the identity fields, choose limits from your
requirements, and freeze it **before** collecting candidate results):

```json
{
  "version": 1,
  "status": "template",
  "id": "workload-v1-checkpoint-calibration-v1",
  "suite_sha256": "REPLACE",
  "model_revisions": {"decider": "REPLACE", "base": "REPLACE"},
  "config_sha256_by_mode": {
    "checkpoint": "REPLACE", "int8": "REPLACE",
    "int8-compute": "REPLACE", "int4": "REPLACE"
  },
  "minimum_labeled_decisions": 1000,
  "modes": {
    "checkpoint": {"semantic_failures": 0},
    "int8": {"answer_flips_vs_checkpoint": 0, "max_absolute_probability_drift": 0.01},
    "int8-compute": {"answer_flips_vs_checkpoint": 0, "max_absolute_probability_drift": 0.01},
    "int4": {"answer_flips_vs_checkpoint": 0, "max_absolute_probability_drift": 0.01}
  }
}
```

Those numbers illustrate the format; they are **not** established production
acceptance limits. A template is rejected until explicitly frozen. Store the
policy with the workload and calibration split identifiers in version control.
Each precision is a distinct candidate even when it retains the same checkpoint
temperatures. Separately fitted calibration configurations require a new policy
and evaluation, not reuse of an old pass. No public smoke corpus or software gate
establishes deployment representativeness, absence of leakage, or an acceptable
false-accept/false-reject tradeoff. #3 remains open for that empirical work.

## Hardware evidence

The [Lambda host and experiment guide](../lambda/README.md) now provide a resident
Runtime API service, ARM64 container recipe, private disposable stack, and
cold/warm benchmark runner. Local tests exercise real Runtime API requests with
synthetic weights, including rejection before deserialization and engine reuse.
They do not establish container deployment success or actual Lambda performance.
#7 still requires the real AWS experiment and a committed measured report.

#6 requires an operator/allocation/cache profile on the intended ARM host before
choosing a projection-buffer rewrite. No unmeasured GEMM rewrite is included.
Capture CPU/toolchain/compiler flags, warm full-model inputs, precision and
worker count; use native sampling/allocation tooling on a matching Graviton2
host when Lambda does not expose those counters. Keep profile instrumentation
out of the production timing runs. An operator gain must survive an alternating
full-model baseline/candidate comparison at short and long prompts in every
precision, including scalar fallback, before replacing a projection path.
