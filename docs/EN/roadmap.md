# Roadmap

[Русский](../RU/roadmap.md)

The goal is a verifiable, maintainable, fast native C23/C++23 engine (with zero Python or PyTorch at runtime) powered by a unified library of composable **Khronos Slang (`slangc` AOT)** kernels and **CPU + Vulkan** as the first backends. Every new or migrated piece is accepted only after checking correctness, memory cost, and full execution time.

Stages express dependencies; independent tracks (e.g., weight container formats and Slang PoC) may progress in parallel.

---

### Stage 0 *(Completed, archived on `archive/c23-cpu-legacy`)* — Initial C23/SIMD Baseline
This stage built the initial handwritten C23/SIMD CPU suite (`VelriLib::cpu` and legacy `cpu_linear`), the persistent `velri_pool` (pthreads), independent FP64/INT64 references, and root `Velri` integration (subdirectory and installed `VelriLib` CMake package). The handwritten SIMD suite served as the accuracy and throughput benchmark bar for Slang CPU and is preserved on branch `archive/c23-cpu-legacy` now that `main` is pure Slang-first.

---

### Stage 1 *(Completed)* — Minimal `velri-lib` Core and Backend Contract
Implemented the `VelriLib::core` library core ([velri-lib/docs/EN/architecture.md](../../velri-lib/docs/EN/architecture.md)):
- `Storage` / `TensorView`, explicit K/V roles (`VELRI_KV_SEPARATE` / `SHARED` / `DERIVED`), central operation registry, graph builder, and liveness-based scratch arena memory planner with UMA-aware reporting;
- 6-group C backend contract (`query_operation`, `prepare_op`, `submit`, `poll`, `wait`), static registration, and automatic `Device A → RAM staging → Device B` transfers;
- thin CPU adapter over `VelriLib::cpu`, optional Vulkan adapter (`-DVELRI_ENABLE_VULKAN=ON`), and controllable `mock_async` backend (`tests/core/test_core_contract.c`: 2 concurrent SwiGLU MLP slots, pre-accept rejection, post-accept/partial-accept drain, 3-window cancellation, RAM bus transfers, and hybrid CPU↔Vulkan).

---

### Stage 2 *(Completed)* — Slang-First MatOps AOT (CPU + Vulkan)
`slangc` is integrated into CMake as a required build-time AOT compiler ([velri-lib/docs/EN/kernels.md](../../velri-lib/docs/EN/kernels.md)):
- authored shared Slang modules (`common/` + `ops/`) covering core MatOps: `GEMV`/`GEMM` (`F64`, `FP32`, `FP16`, `BF16`, `W8A8`, `W8A32`, mixed modes), `RMSNorm`, `Add+RMSNorm`, `LayerNorm`, `RoPE`, activations, `GatedMLP`, `Softmax`, `FWHT`, `QuantRow`, and conversions;
- compiled the shared `.slang` sources AOT to **SPIR-V 1.6** (35 `spirv-val`-verified shaders including `CoopMat` and `Split-K`) and **CPU `.o`** (`base`/`avx2`/`avx512` on x86-64 with zero per-ISA source duplication);
- integrated CPU `.o` dispatch directly over `velri_pool` (`VelriLib::cpu`) with `0 B` heap allocation; verified numerical accuracy against independent C23 references (`bench_slang_cpu --verify-only`, `bench_vulkan --verify-only`) and measured against CPU and GPU physical ceilings; recorded remaining `slang-llvm` gaps (`BF16`/`VNNI`).

---

### Stage 3 *(Completed)* — End-to-End Synthetic Decoder & Engine Skeleton (CPU + Vulkan + Hybrid)
Assembled reusable blocks (`velri-lib/include/velri_lib/blocks.h`), streaming C23 JSON & cascading placement (`src/format/json.{h,c}`, `src/config/placement.{h,c}`), declarative GPT-2 and Llama/Qwen2 architectures (`src/model/arch/{gpt2,llama}.c`), DAG barrier elision + persistent Vulkan decode graph capture + FlashDecoding Split-KV, and the public C23 runtime (`include/velri/velri.h`: `Context`, `Model`, single/dual-plan zero-duplication `Program`, `Session`, zero-alloc sampler).

**Verified:** synthetic `prefill` and `decode` match independent fixtures across chunked/single-token splits, CPU, Vulkan, and hybrid `prefill-compute != compute` dual-plan placements with 0 weight/KV duplication, `0 B` hot-path allocations, multi-session isolation, and safe cancellation/reset (`gpt2_blocks`, `config_placement`, `engine_runtime`, `core_contract`).

---

### Stage 4 — Slang CPU Expansion, `slang-llvm` Improvements, and Portability
Add block-level and quantized Slang CPU kernels, close the recorded `slang-llvm` gaps (`AVX512_BF16`, `AVX512_VNNI`), and expand guard-page contract testing plus AArch64/RISC-V cross-compilation (via LLVM IR).

**Acceptance:** all CPU kernels pass contract tests (guard pages, ASan/UBSan, x86/AArch64/RISC-V QEMU); Windows (Clang + native worker pool) and Android NDK builds are verified.

---

### Stage 5 — Real Model Loading
Implement `.vtq` weight-format loading (using the built-in streaming C23 JSON footer scanner), real weight binding for dense family adapters (`GPT-2` / `Llama` / `Qwen`), tokenizer, chat templates, and CLI `inspect` / `run`.

**Acceptance:** full-model `token IDs` and logits match an independent reference; `inspect` accurately reports devices, resolved placement, and memory budgets.

---

### Stage 6 — First Vulkan Backend
Integrate `Vulkan-Headers`, `volk`, `VMA`, and Slang SPIR-V kernels (Vulkan 1.1 baseline + capability-gated `CoopMat`/subgroups/FP16/BF16/INT8).

**Acceptance:** end-to-end Vulkan decoder matches CPU within tolerances, passes Vulkan API and synchronization validation layers, and is measured across multiple GPU vendors and Android.

---

### Stage 7 — Hybrid CPU + Vulkan Execution
Implement mixed CPU + GPU plans over RAM staging, explicit layout conversions, cross-queue/pool dependencies ([velri-lib/docs/EN/execution.md](../../velri-lib/docs/EN/execution.md)), followed by disk-backed tables (PLE).

**Acceptance:** changing placement (`vulkan:0` / `cpu:0` / `ram:0` / `disk`) preserves accuracy within tolerances; peak memory (including UMA), cancellation, errors, and transfer p50/p99 latencies are verified.

---

### Stage 8 — Advanced Architectures and Quantized Paths
Add `GatedDeltaNet`, `QSA`, `MoE`, `Hyper-Connections`, `PLE`, fused Slang paths such as bufferless `K = RoPE(V * w_k)` (`Gemma 4`), plus the active low-bit quantization format and `velri-quantize`.

**Acceptance:** each mechanism and full model match independent reference implementations; shared blocks remain free of family-name branches.

---

### Stage 9 — Applications and Multi-Request Serving
Build out CLI/chat, optional HTTP/SSE server (`serve`), multi-request batching, `prefix reuse`, speculative decoding (`MTP` / `Gemma4Assistant`), and multimodal encoders (`ViT`, `Conformer`, direct patch projection).

**Acceptance:** strict session isolation, bounded queues with backpressure, clean cancellation on client disconnect, and verified speculative commit/rollback.
