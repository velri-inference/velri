# CPU Operations (`VelriLib::cpu`, Current Implementation)

[Русский](../RU/cpu-kernels.md)

This document describes the **implemented and tested** C23 CPU operations exported by CMake target `VelriLib::cpu` (header `<velri_lib/kernels.h>`). It is a standalone working layer (Stage 0) without the graph executor or Vulkan backend; in the target design ([kernels.md](kernels.md)), it serves as the working baseline and correctness/performance bar for migrating kernels to **Slang AOT**.

The typed kernels were adapted from the user-supplied Gemini/Carbon implementation ([source provenance](../origins/README.md)). Shared x86 SIMD sources are documented in [src/backend/cpu/kernels/README.md](../../src/backend/cpu/kernels/README.md).

---

## 1. Linear (`GEMV` / `GEMM`): Formula and Formats

`Y[rows, M] = X[rows, K] * transpose(W[M, K]) + bias[M]`.

Here `rows` is the number of input vectors, `K` is their length, and `M` is the number of output features. `rows = 1` is GEMV; multiple rows are GEMM. Weights are supplied row-major (with optional row stride `w_stride >= K`); `bias` is optional. General BLAS `alpha`/`beta` and arbitrary transpose modes are not implemented.

| Inputs (`X`) / Weights (`W`) | Accumulator | Output (`Y`) | Scales |
| :--- | :--- | :--- | :--- |
| `F64` / `F64` | `F64` | `F64` | — |
| `FP32` / `FP32` | `FP32` | `FP32` | — |
| `FP16` / `FP16` | `FP32` | `FP16` or `FP32` | — |
| `BF16` / `BF16` | `FP32` | `BF16` or `FP32` | — |
| `INT8` / `INT8` | `INT32` | `INT32` | — |
| `INT8` / `INT8` (`W8A8`) | `INT32` | `FP32` | `x_scale[rows]`, `w_scale[M]` |
| `FP32` / `FP16` or `BF16` | `FP32` | `FP32` | — |
| `FP32` / `INT8` (`W8A32`) | `FP32` | `FP32` | `w_scale[M]` |

For `W8A8`, each output is `signed_dot * x_scale[r] * w_scale[c] + bias[c]`. Scales and floating biases are `FP32`, except `F64` (which uses `F64` bias). Integer outputs use `INT32` bias.

---

## 2. Preparation, Ownership, and Invocation

1. Select a descriptor, e.g. `velri_desc_f64()`, `velri_desc_bf16(VELRI_DTYPE_F32)`, or `velri_desc_w8a8_f32()`.
2. Call `velri_linear_typed_query` to validate dimensions and select an implementation. `VELRI_CPU_AUTO` allows fallback to narrower ISAs; requesting an unsupported ISA explicitly returns `VELRI_ERR_UNSUPPORTED`.
3. If `prepared_weights = true`, allocate `plan.packed_bytes` bytes and call `velri_linear_typed_pack` once. Otherwise supply raw row-major weights.
4. Populate `velri_linear_run_args` and call `velri_linear_typed_run` (single-threaded), `velri_linear_typed_run_parallel` (via `velri_pool`), or `velri_linear_typed_run_slice` (for an external scheduler).

**Memory and threading rules:**
- Plans and packed weights are immutable after preparation. Packing belongs to its producing plan: do not serialize `velri_linear_typed_plan`, edit its fields, or reuse a packed buffer with another plan. No stable binary ABI is promised yet.
- Capacities and strides count **elements**, except packed weights and `scratch`, which count **bytes**.
- Natural element alignment suffices for all caller buffers; 8-byte alignment suffices for any packed buffer; `scratch` may be unaligned.
- Read-only inputs may overlap each other. Output `y`, `scratch`, the plan, and the args struct must be disjoint from writable ranges. All buffers must outlive the synchronous call; `rows = 0` writes nothing.
- **Zero heap allocations during execution:** `rows <= 2` needs `0 B` scratch; `rows >= 3` needs `plan.scratch_bytes` per worker (owned by the pool or caller).
- Only one thread may submit to a given `velri_pool` at a time; nested dispatch on the same pool is unsupported. Separate pools with disjoint outputs may safely share immutable weights.

---

## 3. Kernel and Scheduler Internals

- **Prepared weights:** pack a small tile of output columns together so one vector instruction updates multiple outputs from one input broadcast. Incomplete tiles are zero-padded; INT8 correction terms (`-128 * sum(w)`) and `w_scale` sit adjacent to the tile. Packing pays off when weights are reused and is not guaranteed to beat row-major GEMV on every shape — always measure.
- **GEMV and small-batch path:**
  - `rows <= 2` runs the GEMV kernel directly (register accumulation, `0 B` scratch).
  - `rows = 3..16` uses the 1D output-tile scheduler (like GEMV) while calling the GEMM microkernel and requiring `plan.scratch_bytes`.
  - Small weight matrices assign one contiguous column range per worker; large matrices use per-worker queues grouped by L3 domain with work stealing (dot products along `K` are never split across workers).
- **GEMM path (`rows > 16`):** 2D tiling over rows and output columns preferring jobs in the worker's L3 domain. Packed panels are reused inside bounded `scratch`.
  - AVX2: `6×16` (`FP32`) and `6×8` (`F64`) register tiles.
  - AVX-512: `14×32` (`FP32`) and `14×16` (`F64`) register tiles.
  - AVX-512 BF16 (`linear_avx512_bf16.c`) and INT8 (`linear_avx512_int8.c`) translation units compile separately and gate on `has_avx512_bf16` and `has_avx512_vnni` (this also applies to mixed `F32×BF16` and `F32×I8` under explicit `VELRI_CPU_AVX512`).
- **Shared SIMD sources and codegen:**
  - `src/backend/cpu/kernels/gemv.c` + `gemv.inc` and `elementwise.c` compile for both AVX2 and AVX-512 atop `simd/x86.h` primitives.
  - `cmake/gemm.cmake` generates the shared FMA register-update macros for 6-row and 14-row GEMM tiles.
  - `linear_workspace.h` computes aligned offsets and exact `scratch_bytes` from each implementation's block dimensions (e.g., on AVX-512: `290,879` bytes/worker for packed `F64` and `537,663` bytes for row-major `INT8`; callers always query `plan.scratch_bytes`).

---

## 4. Other Operations (`ops.h` and `types.h`)

| Operation | Supported Formats |
| :--- | :--- |
| `velri_rmsnorm` (`w` or `1 + w`), `velri_rope` (`HALF` / `INTERLEAVED`, partial `rotary_dim`) | `FP32`, `FP16`, `BF16` (`FP32` accumulation) |
| `velri_fused_add_rmsnorm_f32`, `velri_layernorm_f32` | `FP32` |
| `velri_activation_f32` (`SiLU`, `GELU_tanh`, `ReLU2`), `velri_softmax_f32` | `FP32` |
| `velri_gated_mlp_f32` (`activation(gate) * up` with opt. `clamp`), `velri_vec_axpby_f32`, `velri_vec_mul_f32` | `FP32` |
| `velri_fwht_f32` (normalized block Hadamard: blocks `16, 32, 64, 128, 256`) | `FP32`; opt. `FP32` / `FP16` `suh` |
| Conversions (`velri_cvt_*`, `velri_cvt_parallel`) and per-row quantization (`velri_quantize_row_f32_i8*`) | `FP32 ↔ FP16/BF16`; `FP32 → INT8` with row scales |

`velri_gated_mlp_f32` performs only the element-wise part; matrix projections are separate Linear calls. Having `F64` and `INT8` Linear does not imply `F64` norms or integer variants of every element-wise op.

---

## 5. Arithmetic, Tests, and Benchmarks

- **Floating-point math:** uses FMA, multiple accumulators, blocked reductions, and polynomial approximations in SIMD activations/softmax. Low bits may differ across ISAs; tests validate against higher-precision references with tolerances.
- **INT8:** product accumulation and `INT32` bias addition wrap **modulo $2^{32}$** using explicit unsigned math (no signed-overflow UB); `W8A8` casts the wrapped signed `INT32` to `FP32` before scaling.
- **Dynamic `FP32 → INT8` quantization:** requires finite inputs, round-to-nearest-even, and subnormal flushing disabled. Zero rows get scale `1.0`; tiny scales clamp to `FLT_TRUE_MIN` using `double` division to avoid reciprocal overflow.
- **Tests (`ctest`):** `test_linear_typed`, `test_ops`, and `test_contract` cover all 11 Linear modes, all ops, tails, strides, `INT32` wrapping, worker-count changes, naturally aligned packing, and exact-size `scratch` buffers with canaries and `guard pages`.

### `bench_cpu` Benchmark
Build with `-DVELRI_BUILD_CPU_BENCHMARK=ON`:

```sh
./build/bench_cpu bf16 avx512 1 4096 4096 8 1 32 0
```

Arguments: `mode`, `isa`, `rows`, `M`, `K`, `workers`, `prepared (0|1)`, `ring_count` (weight-buffer ring size to defeat cache residency), `iters` (`0` = auto). CSV outputs input columns, median/best/worst call time (**ms**), throughput (**TFLOPS** for float / **TOPS** for integer), effective weight bandwidth (**GB/s** = `weight_bytes / time_s / 1e9`), weight size, `scratch` per worker, and setup time (**ms**). Setup is outside timed calls; validation and pool dispatch are included. Raw reports stay locally in gitignored `docs/benchmarks/`.

### Current OS and pool limits
`velri_pool` (`src/platform/pool.c`) currently uses POSIX `pthreads` with a 250 ms active spin before parking on a condition variable (favors back-to-back dispatches while briefly occupying CPU after the last call). Windows thread integration is separate work.

---

## 6. Portability and Cross-Compilation

Every operation keeps a portable C23 implementation (`VELRI_CPU_BASE`); public headers store `FP16`/`BF16` in `uint16_t` without x86 vector ABI dependencies (assuming IEEE binary32/binary64 floats). CMake inspects the compiler target and never passes x86 flags on non-x86 targets.

| Target | Implementation | CI Verification (`.github/workflows/kernels.yml`) |
| :--- | :--- | :--- |
| x86-64 baseline | C23 | Full suite with `VELRI_CPU_MULTIVERSION=OFF` and QEMU `Nehalem` |
| x86-64 AVX2 | C23 + AVX2/FMA/F16C | Native suite; AVX2 dispatch and arithmetic under QEMU `Haswell` |
| x86-64 AVX-512 | Separate FP/BF16/VNNI units | Native suite on compatible CPU (QEMU TCG lacks AVX-512) |
| ARM64 / AArch64 Linux | Portable C23 | GCC 14 cross-build + full `ctest` under QEMU |
| RISC-V 64 Linux (`rv64gc`, `lp64d`) | Portable C23 (no RVV required) | GCC 14 cross-build + full `ctest` under QEMU |

QEMU emulation verifies compilation and correctness, not native ARM64/RISC-V speed. Because QEMU TCG has masked-load fault-suppression bugs, SIMD guard-page tests run natively while emulated AVX2 uses ordinary buffers.

```sh
cmake -S velri-lib -B build-arm -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=velri-lib/cmake/toolchains/aarch64-linux-gnu.cmake
cmake --build build-arm && ctest --test-dir build-arm --output-on-failure

cmake -S velri-lib -B build-riscv -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=velri-lib/cmake/toolchains/riscv64-linux-gnu.cmake
cmake --build build-riscv && ctest --test-dir build-riscv --output-on-failure
```

---

## 7. Deprecated Regression Target `VelriLib::cpu_linear`

> **Status: deprecated, scheduled for removal.** Retained in the build for regression checks today and planned for removal once the new Slang CPU path stabilizes (Stage 4 in [roadmap.md](../../../docs/EN/roadmap.md)).

- **Interface:** `<velri_lib/linear.h>` (`velri_cpu_variant_available`, `velri_linear_query`, `velri_linear_pack`, `velri_linear_run`) and target `VelriLib::cpu_linear`.
- **Contract:** `FP32` inputs/weights only, mandatory sequential `FP64` accumulation from `-0.0` in increasing `K` order, canonical NaN `0x7fc00000`, and **bitwise** equality with `tests/cpu/reference.c` and the frozen fixture [tests/cpu/fixtures/README.md](../../tests/cpu/fixtures/README.md) (`rust-linear-v1.bin`).
- **Scope:** this blanket bitwise contract applies solely to this legacy target and does **not** carry over to `VelriLib::cpu` or future Slang kernels. Its diagnostic benchmarks `bench_linear` (`-DVELRI_BUILD_BENCHMARK=ON`) and Linux-only `bench_linear_parallel` (`-DVELRI_BUILD_PARALLEL_BENCHMARK=ON`) remain available alongside it.
