# llama.cpp comparison, 2026-10-14

**The native engine is not at llama.cpp's speed on this x86-64 host.** On the matched single-question workloads, llama.cpp was 3.7-6.3 times faster with BF16 base weights across the initial and confirmation runs, and 3.0-4.0 times faster with F32 base weights. The F32 comparison reduces numerical differences substantially, but neither llama configuration passes every strict reference tolerance. This is a performance baseline with measured numerical differences, not proof of interchangeable outputs or an ARM Lambda result.

## Setup

- AMD EPYC 9V74 virtual CPU, x86-64, AVX2/AVX-512/BF16 advertised. Nine visible CPUs; cgroup quota eight CPU cores, memory limit 8 GiB.
- Four threads per engine. Native Rust release with `-C target-cpu=native`; llama Release with `GGML_NATIVE=ON`, CPU only. Neither engine ran concurrently with the other.
- Native engine commit `773fae82201b97f902fcaef76caee3e57c7ec3ad`; llama.cpp commit `6868a52cb337450658006ad2f9281c2eb63f1666`.
- Exact pinned v21 model and base revisions, separate LoRA, original FP32 pointer head. llama uses F32 attention K/V caches. GGUF F32 expands the original BF16 base values; it does not substitute a different model.
- Three short independent reference probes and the exact 1,024-token reference probe. Identical token IDs, option positions and final answer positions. One excluded warmup and three timed repetitions per case, within one loaded process per engine/configuration.
- Timings include the complete backbone, LoRA and pointer head. They exclude tokenization, admission, JSON response construction and model loading. Questions are independent; neither client uses shared-prefix caching.
- The llama public evaluation callback extracts hidden rows before LAST pooling, which suppresses vocabulary logits. This is an external adapter, not native Strands server support or a modification to llama.cpp.

See [methodology and reproduction commands](../llama-comparison.md). Raw results and environment metadata are in [the evidence directory](llama-comparison-2026-10-14/).

## Warm latency

Median seconds; three samples per cell. Ratios divide native time by llama time.

| Workload | Tokens | Native checkpoint | llama BF16 | Ratio | llama F32 | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| Choice: route | 87 | 3.896 | 0.919 | 4.24x | 1.161 | 3.36x |
| Noul: urgent | 86 | 3.772 | 0.872 | 4.33x | 1.152 | 3.28x |
| Score: severity | 88 | 3.894 | 0.935 | 4.16x | 1.290 | 3.02x |
| Long noul | 1,024 | 58.098 | 9.210 | 6.31x | 14.436 | 4.02x |

These are samples on a shared virtual machine, not confidence intervals or p95/p99 latency. The three long native warm samples were 58.098, 56.699 and 60.453 seconds, showing the host's variation. Do not multiply these ratios by the earlier prefix-cache or ARM projection improvements: this native commit already contains those changes.

A second BF16 run, after the native and F32 runs, returned identical logits. Its medians were 1.048, 0.961, 1.027 and 9.801 seconds in table order, corresponding to 3.72x, 3.92x, 3.79x and 5.93x versus the original native samples. This confirms the broad gap while demonstrating timing variation. The native run was not repeated, so this is not a fully interleaved statistical study.

## Numerical results

All four argmaxes match across both engines, and repeated logits were deterministic within each configuration. The score distribution and score expectation are not asserted equal merely because their argmax matches.

| Maximum across cases | Native checkpoint | llama BF16 | llama F32 |
|---|---:|---:|---:|
| Absolute logit error against independent FP32 reference | 0.00000224 | 0.004666 | 0.000142 |
| Calibrated probability difference against native | 0 | 0.001428 | 0.0000224 |

The BF16 maximum probability difference is **0.143 percentage points**; F32 reduces it to **0.00224 percentage points**. Probabilities use the pinned per-kind temperature. No held-out calibration assessment is implied.

Native results remain below the existing `1e-4` logit tolerance. The llama BF16 run exceeds that tolerance; the F32 route case also exceeds it. The route and severity F32 probability differences against native exceed `1e-5`. The comparison adapter includes its own FP32 head reduction order, so these figures should not be attributed solely to a particular llama backbone operator without further investigation. No tolerance was relaxed to label these runs equivalent.

## Memory and loading

The native process peaked at **3.765 GiB RSS**, the llama BF16 confirmation process at **3.934 GiB**, and the llama F32 process at **7.425 GiB**. These include loading, warmup and all measured cases. F32's larger weight storage is a material trade-off, and this host leaves little memory headroom for it. The cgroup recorded no OOM kills during the runs.

Reported load times were about 4.19 seconds native, 0.66 seconds llama BF16 and 2.36 seconds llama F32. **These are not cold-start comparisons:** llama memory maps weights, native loads eagerly, filesystem caches were warm, and deferred page faults can move work into the first forward. They are retained in the raw evidence for diagnosis only.

## What this establishes

There is a substantial measured x86 performance gap even with the same model, LoRA, prompts and decision head, and even when llama's base is stored as F32. It does not establish the gap on Graviton: the native engine's recent projection tiling applies only on ARM64. It also does not measure multi-question prefix reuse, INT8/INT4, concurrency or AWS billing.

This advances the evidence for #6 but does not satisfy its intended-hardware profiling requirement or #7's AWS measurements. The next useful measurement is this same comparison on ARM64, alongside operator profiling, before choosing a projection rewrite.
