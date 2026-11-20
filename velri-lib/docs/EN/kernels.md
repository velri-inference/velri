# Kernel Architecture: Slang-First (`slangc` AOT for GPU and CPU)

[Русский](../RU/kernels.md)

Rather than maintaining separate GLSL shaders, `#ifdef` matrices, and duplicated handwritten SIMD files for every CPU ISA, `Velri-lib`'s target kernel system is built around **Khronos Slang** (`Apache-2.0 WITH LLVM-exception`) with **pure build-time AOT compilation via `slangc`**.

**Status:** target design plus modular Slang kernel prototype in `velri-lib/src/kernels/slang/`. The working native C23/SIMD suite in `VelriLib::cpu` ([cpu-kernels.md](cpu-kernels.md)) serves as the Stage 0 baseline and correctness/performance bar for migrating kernels to Slang CPU.

---

## 1. Core Principles: Reusable Components and Zero Duplication

1. **Small composable components instead of monolithic copies:** shared math is factored into Slang interfaces and generics (`IElement`, `IActivation`, `IEpilogue`, `IKVSource`, shared reductions). One `RMSNorm`, `RoPE`, `GatedMLP`, or `Linear` implementation reuses those building blocks instead of copy-pasting.
2. **Do not multiply near-identical kernels:**
   - **No separate kernel sources for AVX2 vs. AVX-512** (or per-dtype copy-paste): one parameterized `.slang` module compiles AOT via `slangc` for all target profiles (`base`, `avx2`, `avx512`, `aarch64`, `riscv64`, `spirv`).
   - Hardware differences (such as GPU subgroup reductions vs. CPU loops, or GPU `CoopMat`) stay isolated inside thin low-level primitives rather than duplicating whole kernels.
3. **Pure AOT (`slangc` at build time) — zero compiler at runtime:** the shipped Velri binary has no runtime dependency on `libslang`, LLVM, `shaderc`, Python, or PyTorch.

```text
src/kernels/slang/
  ├── common/ (numeric, activations, epilogue, reductions)
  └── ops/    (norm, rope, elementwise, linear, ...)
        │
        ├──► slangc -target spirv      ──► .spv (+ spirv-val)       ──► Vulkan backend
        ├──► slangc -target llvm-ir/o  ──► .o (CPU ISA variants)     ──► CPU backend (velri_pool)
        └──► slangc reflection         ──► generated CMake dispatch/binding headers
```

---

## 2. Modular Composition in Slang

Velri's composition invariants are expressed directly in Slang types with zero model-family branches:

- **`IElement` (`common/numeric.slang`):** uniform access to `F32`, `F64`, `FP16`, `BF16`, and `INT8` (load/convert to accumulator and store back) so operations are not duplicated per dtype.
- **`IActivation` (`common/activations.slang`):** `SiLU`, `GELU_tanh`, `ReLU2`, `Identity` (+ optional symmetric `clamp`) — reused simultaneously by standalone activations, `GatedMLP`, and `Linear` epilogues.
- **`IEpilogue` (`common/epilogue.slang`):** bias addition, scaling (`W8A8` / `W8A32`), and optional activation — shared by `GEMV` and `GEMM` across CPU and GPU.
- **`IKVSource`:** key/value source for attention — separate buffers, one shared buffer, or on-the-fly reconstruction `K = RoPE(V * w_k)` (Gemma 4) without allocating a temporary `K` buffer.
- **AOT fusions:** hot sequences (`Residual + RMSNorm`, `GatedMLP`, `Linear + Epilogue`, `K=V + Attention`) are instantiated as AOT specializations of the same interfaces from a controlled CMake manifest.
- **Autodiff (`bwd_diff`):** built-in Slang capability available when needed for future quantization/calibration tools without rewriting math.

---

## 3. GPU Path: Vulkan / SPIR-V and Multi-GPU Portability

- **Baseline profile:** Vulkan 1.1 compute with standard storage buffers and explicit device capability queries prior to kernel selection.
- **Accelerated paths (SPIR-V 1.5/1.6):** `linalg::CoopMat` (`SPV_KHR_cooperative_matrix`, FP16/BF16/INT8), `VK_KHR_shader_bfloat16`, `VK_KHR_shader_float16_int8`, `VK_KHR_shader_integer_dot_product`, and subgroup ops are enabled **only when confirmed supported by the device**.
- **Independent capability checks:** storage, arithmetic, and accumulation capabilities are queried separately; no fixed subgroup size (32 or 64) is assumed.
- **Build-time validation:** every emitted `.spv` is validated by `spirv-val` for the target Vulkan profile in CMake and CI.
- **Pipeline management:** AOT specialization selection and `VkPipeline` creation happen in `prepare`; on-disk pipeline cache compatibility is checked against device and driver identity.
- **Portability to future GPU APIs:** the same `.slang` modules can compile to **Metal (MSL)**, **CUDA**, **DXIL**, and **WGSL** behind thin device backends in `src/backend/<name>/` without rewriting math.

---

## 4. CPU Path: Slang-First and Slang CPU Backend Improvement Plan

### Role of Slang and C23 on CPU
The target CPU strategy is **Slang-first for all optimized kernels** from the shared modular codebase. Pure **C23** retains:
- the public API, graph, memory planner, and backend registry;
- the persistent worker pool `velri_pool` and L3-aware scheduler;
- **simple independent C23 reference implementations** for correctness checks without involving the Slang compiler.

### Out-of-the-box Slang CPU limits and improvement strategy
Slang supports CPU emission via LLVM (`slang-llvm`: LLVM IR / `.o`) and C++. We account for its practical limits explicitly:
1. The C++ backend lacks workgroup barriers (`GroupMemoryBarrierWithGroupSync`) and general user atomics — `groupshared`-synchronized GPU shaders do not port directly to CPU.
2. Default C++ output carries wrapper overhead unsuitable for short `velri_pool` dispatches; CPU targets need direct **LLVM IR / `.o` lowering with a clean C-ABI** and `0 B` heap allocations.
3. One Slang source compiles to target LLVM IR, and **compiler target flags** (`-march=x86-64`, `-mavx2 -mfma -mf16c`, `-mavx512f -mavx512bw ...`, `aarch64`, `riscv64`) emit ISA variants — **without duplicating kernel code for AVX2 vs. AVX-512**.
4. Hot CPU `GEMM` / `GEMV` requires register tiling, bounded `scratch` panel packing, and hardware dot instructions (`vdpbf16ps`, `vpdpbusd`). Wherever stock Slang CPU codegen falls short, we measure the exact gap and apply targeted Slang CPU backend improvements (clean C-ABI, `fp16`/`bf16`, vector primitives/intrinsics).

### Verified Slang `2026.22.1` findings and limits (prototype measurements)
The opt-in prototype is built with `-DVELRI_ENABLE_SLANG_CPU=ON` and `-DVELRI_ENABLE_VULKAN=ON` (`entry_cpu.slang`, `entry_vulkan.slang`, benchmarks `bench_slang_cpu` and `bench_vulkan`). With `slangc 2026.22.1`, both targets are verified from shared `common/` and `ops/` modules:
- **CPU (`-target host-object-code -O3 -fp-mode fast`):** zero-runtime C-ABI emission (`[DllExport] export __extern_cpp`); one parameterized `linear_cpu.slang` compiles across `base`/`avx2`/`avx512` via `-DVELRI_MR/NR_F32/NR_F64` with zero per-ISA source duplication. Key codegen practices for `slang-llvm`:
  - `pack_b` via `IElement::load4F32` + aligned `Ptr<float4>`/`Ptr<double4>` stores (eliminates `vgather`/`vscatter`), plus zero-copy `A` reads for native `F32`/`F64` (eliminates `LoopIdiomRecognize` `call memcpy`);
  - exact modulo-$2^{32}$ `W8A8` reusing the `F32` outer-product microkernel at `KC=128` ($|\sum| \le 2^{21} \ll 2^{24}$);
  - single-loop `select(v > m, v, m)` reductions (scalar `max()` and `?:` emit `vucomiss` even under `-fp-mode fast` and block vectorization; `select` yields 4-way ZMM `vmaxps`), Radix-4 Stockham ping-pong FWHT without in-place aliasing, and writing local arrays through `Ptr` (avoiding Slang local-array SSA pitfalls).
- **Vulkan (`-target spirv -profile spirv_1_6 -fvk-use-c-layout -O3`):** under C layout Slang defaults BDA accesses to `Aligned 4` (`4× LDG.32`), so explicit `OpLoad`/`OpStore Aligned 16` helpers (`LDG.128`/`STG.128`), `ks=0 / BAR / STS.128 + ks=1` overlap on single-buffered `CoopMat` (`20 KiB` -> `2 WGs/SM`), and `Split-K` (`vk_splitk_reduce`) for small-batch `F32` are essential.
- **Two remaining upstream `slang-llvm` gaps (scope for future Slang CPU backend work):**
  1. **`BFloat16` is unsupported on the LLVM target** (`InternalError: Unsupported type for LLVM target!`) — bit-shift `ElemBF16` (`uint16_t`) is used instead, preventing `vdpbf16ps` (`AVX512_BF16`) emission.
  2. **`dot4add_i8packed` and integer dot products do not lower to `vpdpbusd` (`AVX512_VNNI`) on CPU** — capping W8A8 at the FP32 FMA ceiling (`~4.0 TOPS` vs. `~8.7 TOPS` for `VelriLib::cpu` with VNNI).

### Replacement rule for existing C23 code
The working native kernels in [VelriLib::cpu](cpu-kernels.md) are **never removed prematurely**. A Slang CPU kernel replaces its native counterpart only after it:
- passes all contract tests (tails, strides, guard pages, ASan/UBSan, AArch64/RISC-V QEMU);
- matches or beats full-call throughput under [performance.md](../../../docs/EN/performance.md).

---

## 5. Numerical Contracts and Verification

- **Validate before the first write:** dimensions, strides, overflow, alignment, and forbidden overlaps are rejected before touching outputs; `rows == 0` performs zero writes.
- **Zero heap allocations on the hot path:** `scratch` is queried during `prepare`.
- **Floating-point ops (`F64`, `FP32`, `FP16`, `BF16`):** FMA, multiple accumulators, blocked reductions, and polynomial approximations are permitted; bitwise equality across ISAs or CPU/GPU is not required — explicit tolerances are verified against C23 references.
- **Integer and exact ops (`INT8`, indexing):** exact deterministic contracts (`INT8 Linear` accumulates modulo $2^{32}$ without signed-overflow UB).
- **Memory safety and validation:** independent C23 references + ASan/UBSan + `guard pages` on CPU; `spirv-val` + Vulkan API/sync validation layers on GPU.
