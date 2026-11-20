# Roadmap

[Русский](../RU/roadmap.md)

The goal is a verifiable, maintainable, fast native C23/C++23 engine (with zero Python or PyTorch at runtime) powered by a unified library of composable **Khronos Slang (`slangc` AOT)** kernels and **CPU + Vulkan** as the first backends. Every new or migrated piece is accepted only after checking correctness, memory cost, and full execution time.

Stages express dependencies; independent tracks (e.g., weight container formats and Slang PoC) may progress in parallel.

---

### Stage 0 *(Completed)* — Baseline C23 CPU Library (`VelriLib::cpu`)
Implemented in `velri-lib/` and verified in CI (`x86-64` baseline/AVX2/AVX-512, `AArch64` and `RISC-V 64` under QEMU):
- typed `Linear` (11 modes: `F64`, `FP32`, `FP16`, `BF16`, `INT8`, `W8A8`, `W8A32`, row-major and prepared);
- `RMSNorm`, `LayerNorm`, `RoPE`, activations, `gated MLP`, `softmax`, `FWHT`, format conversions, and per-row `INT8` quantization;
- persistent `velri_pool` (pthreads), guard-page contract tests, and `bench_cpu`;
- root `Velri` integration build (both subdirectory and installed `VelriLib` CMake package).

This suite serves as the working baseline and correctness/speed bar for subsequent stages.

---

### Stage 1 — Minimal `velri-lib` Core and Backend Contract
Build the library core ([velri-lib/docs/EN/architecture.md](../../velri-lib/docs/EN/architecture.md)):
- `TensorView` / `Storage`, operation registry, graph builder, and memory planner (with explicit K/V roles);
- internal C backend contract (`query_operation`, `prepare`, `submit`, `poll`, `wait`), static registration, and RAM transfer bus;
- thin CPU adapter wrapping `VelriLib::cpu` plus a small async mock backend (one op + controllable completion delay).

**Acceptance:** a standalone client builds a graph, prepares a plan, and runs two independent execution slots with zero engine code; shared contract tests verify pre-accept rejection, post-accept failure, cancellation, RAM transfers, and slot isolation.

---

### Stage 2 — Slang AOT PoC (Vulkan + CPU) and Slang CPU Assessment
Integrate `slangc` into CMake for pure build-time AOT ([velri-lib/docs/EN/kernels.md](../../velri-lib/docs/EN/kernels.md)):
- author reusable Slang components (`numeric`, `activations`, `epilogue`, `reductions`) and initial real operations (`RMSNorm`, `RoPE`, `GatedMLP`, `Softmax`, `GEMV` / `GEMM`);
- compile the same `.slang` source AOT to **SPIR-V** (validated with `spirv-val`) and **CPU `.o`** (via LLVM IR with target ISA flags, with zero AVX2 vs. AVX-512 source duplication);
- dispatch CPU `.o` directly from `velri_pool` with `0 B` heap allocation;
- benchmark Slang CPU head-to-head against native `VelriLib::cpu`, and Slang Vulkan against measured and physical GPU ceilings; record concrete Slang CPU backend improvements needed.

**Acceptance:** shared `.slang` modules compile AOT with zero runtime compiler, pass numerical verification against C23 references, and produce reproducible CPU and Vulkan benchmarks.

---

### Stage 3 — End-to-End Synthetic CPU Decoder
Assemble initial reusable blocks, `Model`, `Program`, and `Session` in `Velri` ([blocks-and-config.md](blocks-and-config.md), [api.md](api.md)).

**Acceptance:** synthetic `prefill` and `decode` match independent fixtures across arbitrary step splits; shared storage and derived K/V allocate no redundant buffers; session isolation and cancellation are verified.

---

### Stage 4 — Migrate CPU Kernels to Slang AOT and Remove Legacy Target
Deliver the Slang CPU backend improvements identified in Stage 2 and progressively migrate optimized CPU ops (including `Linear` GEMM/GEMV) to Slang AOT, keeping thin runtime, `velri_pool`, and independent references in C23.

**Acceptance:** every migrated kernel passes contract tests (guard pages, ASan/UBSan, AArch64/RISC-V QEMU) and matches or beats native `VelriLib::cpu` full-call throughput; remove deprecated `VelriLib::cpu_linear` once regressions are retired. Verify Windows (Clang + native worker pool) and Android NDK builds.

---

### Stage 5 — Real Model Loading
Implement JSON configuration (`yyjson`), weight-format loading, initial dense family adapters (`Llama` / `Qwen3`), tokenizer, chat templates, and CLI `inspect` / `run`.

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
