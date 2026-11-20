# Performance Measurement Methodology

[Русский](../RU/performance.md)

This document sets uniform benchmarking rules across `Velri` and `Velri-lib` so kernel comparisons (including native C23/SIMD vs. **Slang CPU** and **Vulkan** kernels) remain honest and reproducible.

**What works today:** `velri-lib/` ships `bench_slang_cpu` (Slang AOT CPU over `velri_pool`) and `bench_vulkan` (SPIR-V 1.6 when `-DVELRI_ENABLE_VULKAN=ON`, see [kernels.md](../../velri-lib/docs/EN/kernels.md)). The earlier handwritten C23/SIMD suite is preserved for historical comparisons on branch `archive/c23-cpu-legacy`. Raw reports and profiles stay locally in gitignored `docs/benchmarks/`.

---

## 1. Three Measurement Levels

Every reported figure explicitly states its level:
1. **Microkernel** on pre-prepared buffers (excluding thread-pool dispatch).
2. **Full operation call**, including argument validation, dispatch selection, pool or GPU queue submission, and completion wait.
3. **Model step**, covering all layer operations, cross-device transfers, and state updates (`decode` latency, `prefill` throughput, TTFT, peak memory).

Speeding up an isolated microkernel does not prove an end-to-end model speedup.

### Cold start, cache, and DRAM
- Cold loading, weight packing, and `VkPipeline` creation are timed separately; steady-state measurements record their warmup scheme.
- Re-reading one matrix from L2/L3 (cached GEMV) and streaming distinct matrices from DRAM (streaming GEMV, e.g., weight rings in `bench_slang_cpu` / `bench_vulkan`) are distinct regimes and are always labeled explicitly.

---

## 2. Units and Ceiling References

For dense `GEMM` / `GEMV` with input `[N, K]` and weights `[M, K]`:

```text
TFLOPS (or TOPS for INT8) = 2 * N * M * K / seconds / 1e12
```

- Only useful arithmetic operations enter the numerator (zero padding, packing, and transfers never inflate FLOP/OP counts, even though their time is included in full-call timing).
- `GB/s` figures always state whether they count logical weight bytes (`weight_bytes / seconds / 1e9`, as in `bench_slang_cpu` / `bench_vulkan`) or measured hardware traffic at a specific memory level.
- **Conditional references rather than "absolute ceilings":** peak compute estimates depend on active cores, ISA width, dtype, and **sustained clock under the measured workload** (not marketing boost). A GEMV memory reference is `bandwidth * 2 / weight_element_bytes` adjusted for padding. Ratios to such numbers are labeled as **fraction of a reference ceiling** (theoretical, conditional, or measured); unknown physical limits remain unknown.

---

## 3. Profiling Before Optimizing

- On CPU, separately inspect call setup, task publication, worker wake-up, arithmetic, and completion wait. Overlapping wait time across threads must never be summed as single-call latency.
- On Vulkan, separately measure GPU execution (via timestamp queries only when supported by the queue), CPU waits, copies, and pipeline creation. CPU and GPU clocks cannot be subtracted directly without calibration.
- Profilers, sanitizers, and Vulkan validation layers distort timings: always confirm wins with a clean release run without them.
- State a falsifiable hypothesis before editing code (repeated packing, stray buffer, tile imbalance, register spilling). Changing compiler or language alone is not a root cause.

---

## 4. Comparison Tables and Reproducibility

Every comparison table (e.g., native C23/AVX-512 vs. Slang CPU vs. BLIS/OpenBLAS/Vulkan) records:
- source revision/hash, compiler (`clang`/`gcc`/`slangc`), and flags;
- CPU/GPU, driver, ISA, worker count, and `affinity` policy;
- exact dimensions (`rows`, `M`, `K`), `dtype`, layout (row-major / prepared), cache mode (`ring_count`), warmup, and iteration count;
- median and spread (`min`/`p50`/`p99`/`max`), including regressions;
- **numerical contract and verification status** (tolerances against an independent C23 reference for float kernels; exact/bitwise checks for integer kernels);
- interleaved execution (AB/BA) rather than cherry-picking best runs across separate sessions.

QEMU emulation (ARM64, RISC-V, x86 dispatch) establishes compilation and correctness only, never physical hardware performance.
