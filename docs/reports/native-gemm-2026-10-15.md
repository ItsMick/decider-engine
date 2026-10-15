# Native projection acceleration, 2026-10-15 UTC

The native engine now has an optional Rust GEMM path and cache-friendly output writes. The matched x86 experiment reduced the 1,024-token median from **52.80 to 17.07 seconds**, with effectively unchanged memory. On the ARM runner the accelerated native engine beats llama.cpp F32, while remaining slower than its F16 path. General parity with the fastest format is not claimed, and Lambda deployment remains paused.

## Changes

- `fast-gemm` uses safe `ndarray` views and the Rust `matrixmultiply` backend. Tasks convert bounded 128-column panels; model weights retain BF16 storage. Large base projections use GEMM; small projections and INT8-compute keep their existing kernels. This feature changes reduction order and remains opt-in.
- Large output transposes and bias/LoRA epilogues use 16-by-16 scratch tiles and disjoint parallel row writes. The arithmetic is bit-identical to the scalar output pass, including write, residual-add and SwiGLU modes. Small outputs keep the scalar path. This improvement also applies without GEMM.
- The optional `profile` feature records actual model/operator durations. It is excluded from latency binaries. Parent projection spans include their GEMM/write children, so those durations must not be added together.

No new unsafe code is introduced in this crate. The request schema, admission limits, checkpoint pinning and calibration parameters are unchanged.

GEMM's batch-shape-dependent reduction order can introduce small rounding differences between cached and uncached execution. Both paths must pass the unchanged reference tolerances. The default backend retains its bit-identical prefix-continuation behavior; GEMM remains an explicit feature.

## Matched x86 results

AMD EPYC 9V74 VM, four workers, `-C target-cpu=native`, Rust 1.98, checkpoint precision. Each question uses identical pinned token IDs and pointer positions. One warmup is excluded; entries are medians of three samples. Engines run serially, with model loading excluded.

| Probe | Tokens | Pinned baseline | Initial GEMM | Retained GEMM + tiled writes | Overall speedup |
|---|---:|---:|---:|---:|---:|
| Route | 87 | 3.704 s | 1.651 s | 1.720 s | 2.15x |
| Urgent | 86 | 3.759 s | 1.534 s | 1.613 s | 2.33x |
| Severity | 88 | 3.932 s | 1.595 s | 1.756 s | 2.24x |
| Long noul | 1,024 | 52.804 s | 21.129 s | 17.073 s | 3.09x |

Baseline source: `2e684576c5f0c44598e65f32d68f66970c419386`. Initial GEMM source: `6277850e57a9503782db01e83bb77e34bfaec1d5`. Tiled-output source: `cfa5abf6d140b5f19c2490cba13fd326a91e1640`, followed by cleanup removing the unused row-blocking experiment. Binary and retained core-file checksums are recorded in the environment manifest. These are exploratory measurements on one shared VM, without confidence intervals. Small differences between accelerated short-prompt runs should not be treated as reliable regressions or gains.

Peak RSS was **3.750 GiB baseline**, **3.761 GiB initial GEMM**, and **3.762 GiB retained GEMM plus tiled output**. Load timings are retained for diagnosis, not cold-start comparisons.

In separate instrumented 1,024-token runs, output conversion took **7.019 s before** and **1.630 s after** tiling. Total model time was 22.692 versus 17.446 s. GEMM itself took 11.840 versus 11.941 s, supporting attribution of the improvement to output processing. These are individual diagnostic samples; the uninstrumented table above is the latency evidence.

## Correctness

Both GEMM versions passed the complete independent v21 checkpoint test locally, including the 4,096-token context, all decision types and shared-prefix execution. The existing limits remain `1e-4` for logits and `1e-5` for probabilities, confidence and scores. The observed maximum reference logit error on the four timed probes is below `3e-6`. No tolerance was relaxed.

Packed-layout, partial-panel, bias/LoRA and one/four-worker tests cover GEMM. Signed-zero, boundary and epilogue tests compare the tiled output pass bit for bit with the original scalar pass. This numerical validation does not establish production confidence calibration on a representative deployment corpus.

## ARM measurements

The runner is a four-core Neoverse N2 with 128 MiB shared L3. Native binaries target Neoverse N1; llama uses ARMv8.2+FP16+dotprod, tuned for N1. These instruction restrictions do not reproduce Graviton2's cache and memory system, and no AWS resources were deployed.

The complete matched run passed: [workflow 38659917884](https://github.com/ItsMick/decider-engine/actions/runs/38659917884). Each entry below is the median of three warm samples after one excluded warmup. All four engines ran serially on the same runner with four workers and identical token IDs, LoRA and pointer weights.

| Probe | Native baseline | Native GEMM + tiled writes | llama F16 | llama F32 |
|---|---:|---:|---:|---:|
| Route, 87 tokens | 2.369 s | 1.603 s | 1.218 s | 2.175 s |
| Urgent, 86 tokens | 2.368 s | 1.599 s | 1.201 s | 2.145 s |
| Severity, 88 tokens | 2.403 s | 1.595 s | 1.231 s | 2.204 s |
| Long noul, 1,024 tokens | 34.653 s | 18.050 s | 14.298 s | 25.788 s |

The native improvement is **1.48-1.92x**. Relative to the controls, native is **1.34-1.43x faster than llama F32**, and **26-33% slower than llama F16**. F32 conversion preserves the original BF16 values exactly; F16 conversion and computation introduce different rounding. Native performs FP32 arithmetic while retaining packed BF16 storage. These format distinctions matter when interpreting performance parity.

| Peak RSS | Native baseline | Native accelerated | llama F16 | llama F32 |
|---|---:|---:|---:|---:|
| GiB | 3.750 | 3.763 | 3.936 | 7.428 |

The full ARM native checkpoint test passed all original tolerances through 4,096 tokens. Native maximum reference logit error on the four timed probes was below `3e-6`. All tested argmaxes match both llama formats. The maximum calibrated probability difference against native was **0.298 percentage points for F16** and **0.00216 percentage points for F32**. Neither llama control is assumed to meet every strict native tolerance; this is not a held-out quality or calibration study.

The separate instrumented ARM model pass took 18.069 s: 16.232 s in projection processing (including 14.552 s GEMM and 0.498 s output writes), 0.165 s convolution, 0.669 s recurrence and 0.706 s full attention. The parent projection duration includes its sub-spans. This identifies matrix multiplication as the main remaining cost after output tiling.

The measured candidate is commit `cfa5abf6d140b5f19c2490cba13fd326a91e1640`. Final cleanup removes the unused row-blocking experiment and restores the original small-projection loop; GEMM and output-tiling arithmetic are unchanged. The retained code was also rerun through the full local reference test. Full raw ARM records, numerical-gate output and environment details are in [arm-optimized](native-gemm-2026-10-15/arm-optimized/).

An earlier, bit-exact projection row-blocking experiment improved the ARM long median from 33.946 to 30.951 s (1.10x), while short medians were about 1% slower. That path was removed; the retained output tiling addresses a separately measured bottleneck. Both complete native result sets are preserved.

The initial llama BF16 ARM investigation was cancelled after source inspection showed a scalar BF16 fallback at the pinned revision. It produced an observed 1,024-token sample of about 454 s, but its sample set is incomplete and is **not used to claim performance parity**. Optimized F16 and F32 are the required controls. F16 can change weight/activation arithmetic; numerical differences are reported rather than assumed away.

## Evidence and reproduction

Raw samples, process RSS, revisions, quality logs and before/after operator traces are in [the evidence directory](native-gemm-2026-10-15/). See [the comparison guide](../llama-comparison.md) for commands and the opt-in ARM workflow.

These workloads measure independent backbone-plus-pointer-head calls. They exclude tokenization, HTTP, admission and response formatting, and do not measure multi-question prefix-cache savings, concurrency, INT8/INT4 quality, Lambda billing or production latency quantiles. Issues #6 and #7 remain open for their outstanding target-hardware requirements.
