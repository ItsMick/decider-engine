# v21 adversarial and precision smoke evaluation — 2026-10-14

All four modes returned structurally valid answers on the 12-case suite and
matched all nine simple semantic labels. No boolean/choice answer changed from
checkpoint precision. This limited result does not establish prompt-injection
resistance or quantized calibration. In particular, probability thresholds changed
even when answer categories did not.

| Precision | Labeled errors | Boolean/choice flips vs checkpoint | Mean absolute probability drift | Maximum absolute probability drift | Crossings at 0.90 maximum probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| Checkpoint | 0 / 9 | 0 | — | — | — |
| INT8 weights | 0 / 9 | 0 | 0.096 percentage points | 0.210 percentage points | 1 |
| INT8 compute | 0 / 9 | 0 | 0.132 percentage points | 0.594 percentage points | 0 |
| INT4 weights | 0 / 9 | 0 | 2.652 percentage points | 9.397 percentage points | 4 |

The drift statistics cover every option probability, including the unlabeled
structural cases. Threshold crossings cover boolean and choice questions, using
maximum category probability, not the API's derived `confidence` field. The
single score case's expected-value drift was 0.00124 / 0.00524 / 0.04523 ordinal
levels for INT8 / INT8-compute / INT4. No calibration was fitted and no acceptance
policy was enabled. All modes remain unvalidated for consequential deployment.

The attacks cover closing/opening prompt tags, invented options, Unicode
confusables, long separators and conflicting instructions. They did not defeat
the nine deliberately simple labeled checks in this run; no semantic failure was
observed to illustrate. That absence is not evidence that stronger attacks fail.
Three further cases exercise hostile instructions/descriptions/labels and have no
semantic ground truth. The renderer and reference fixtures were left unchanged.

## Reproduction and provenance

- Engine source commit: `748a17368a386b4f4b9dd4f4772d536051447ac3`.
- Engine source tree: `202fb1053a6beec2978c71cffccf89a6f54c4131`.
- Strands v21: `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`.
- Qwen base: `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`.
- Rust 1.98.0, release profile, four Rayon workers.
- Virtualized AMD EPYC 9V74 x86_64, 8 GiB container limit and eight-core CPU quota.
- INT8-compute used the scalar integer fallback, **not ARM SDOT**.
- One run per case/mode; model loaded once per mode. Timing fields are diagnostic
  samples, not warm latency quantiles or a controlled comparative benchmark.

Commands and metric definitions are in [validation.md](../validation.md).
The [environment record](adversarial-v1-environment.json) includes the corpus
SHA256 and exact toolchain. Raw reports are available for
[checkpoint](adversarial-v1-checkpoint.json), [INT8](adversarial-v1-int8.json),
[INT8-compute](adversarial-v1-int8-compute.json), and [INT4](adversarial-v1-int4.json).
The [machine-readable summary](adversarial-v1-summary.json) contains Brier, NLL,
ECE, relative drift and threshold diagnostics. These statistics on nine correlated
hand-written examples must not be read as representative population estimates.

## Remaining issue acceptance criteria

| Issue | Evidence supplied here | Still required |
| --- | --- | --- |
| #1 | Configurable request budgets; exact-boundary and zero-forward rejection tests | Review/merge |
| #4 | Checked shape arithmetic, load/workspace budgets, malformed metadata tests; v21 remains accepted | Review/merge |
| #5 | Adversarial binding suite, repeatable real-weight evaluation and explicit semantic limits | Review/merge |
| #2 | Opt-in/tag parity workflow; current three-question v21 parity | Expanded independently generated reference matrix and release publication dependency |
| #3 | All-mode evaluation harness, raw outputs and probability/threshold diagnostics | Representative held-out data, predeclared deployment gates, calibration and stable quality regressions |
| #6 | No unmeasured kernel changes | Full-model ARM allocation/operator profile and measured layout comparison |
| #7 | No AWS deployment or cost claim | Lambda host, reproducible deployment/teardown, real cold/warm latency, memory, cost and dispatch measurements |
