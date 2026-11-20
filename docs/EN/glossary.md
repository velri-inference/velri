# Velri Glossary

[Русский](../RU/glossary.md)

Concise reference of terms specific to Velri, `velri-lib`, Slang, and target model architectures.

---

## Architecture and API

- **`Velri-lib`** — C23/C++23 computation library in `velri-lib/`. Owns tensors, storage, operation/launch graphs, Slang kernels, and device backends.
- **`Context` / `Model` / `Program` / `Session`** — target opaque C API handles in the engine:
  - `Context` — devices, thread pool, and platform resources;
  - `Model` — configuration, weights, and immutable prepared representations;
  - `Program` — validated immutable execution plan for bounded workload limits;
  - `Session` — mutable per-sequence state and execution slots.
- **`Storage` and `TensorView`** — separation between physical allocation (owner, device, alignment, lifetime) and data view (shape, `dtype`, offset, strides).
- **Preparation (`prepare`)** — one-time graph validation, AOT kernel selection, weight packing, pipeline creation, and memory planning before generation.
- **Execution slot** — pre-allocated bundle of mutable parameters, `scratch` memory, and completion objects for one active step.
- **`y_stride` / `x_stride` / `w_stride`** — row-to-row stride in logical elements (not bytes), enabling zero-copy sub-buffer writes.
- **UMA (Unified Memory Architecture)** — shared physical RAM between CPU and GPU (integrated GPUs, Android); requires single memory accounting without double-counting and still requires explicit Vulkan synchronization.

---

## Kernels, Slang, and Vulkan

- **Khronos Slang** — modular shader and compute kernel language featuring generics, interfaces, and automatic differentiation (`bwd_diff`). Used in `Velri-lib` as the unified language for optimized GPU and CPU kernels.
- **AOT (Ahead-Of-Time)** — compiling `.slang` to `.spv` and CPU `.o` on the build host via `slangc`; no compiler ships in the runtime binary.
- **`slang-llvm`** — Slang's LLVM backend for emitting CPU LLVM IR and native object code (`.o`) for target triples (`x86_64`, `aarch64`, `riscv64`).
- **`CoopMat` (`SPV_KHR_cooperative_matrix`)** — Vulkan/SPIR-V cooperative matrices (`linalg::CoopMat` in Slang) executing subgroup-wide matrix multiply-accumulate on GPU tensor hardware.
- **Subgroup (warp / wavefront)** — lockstep GPU thread group with fast register exchange; its size (32, 64, etc.) is queried from the device rather than hardcoded.
- **Prepared weights (`prepared` / `packed`)** — immutable reordered weight layout (typically output-column tiles with zero padding) enabling efficient vector loads across repeated calls.
- **`velri_pool`** — persistent CPU worker pool in `velri-lib` with pre-allocated 64-byte-aligned per-worker `scratch` (`0 B` hot-path allocations).
- **FWHT (Fast Walsh-Hadamard Transform)** — normalized block Walsh-Hadamard rotation (`velri_fwht_f32`) spreading coordinate outliers across blocks.
- **Guard pages** — inaccessible virtual-memory pages placed flush against test buffers to catch out-of-bounds reads/writes immediately, including inside assembly or lowered object code.

---

## Model Architecture Mechanisms

- **K/V roles** — attention input roles that never mandate two distinct buffers (e.g., storing `V` only and deriving $K = \text{RoPE}_p(V \odot w_k)$ in `Gemma 4`).
- **GDN (`GatedDeltaNet`)** — linear recurrent attention using a fixed-size state matrix and short causal convolution instead of a growing KV cache.
- **QSA / DSA** — specific sparse-attention mechanisms with indexers (`Qwen Sparse Attention`, `DeepSeek Sparse Attention`); described by explicit operations rather than a boolean `sparse` flag.
- **PLE (`Per-Layer Embeddings`)** — per-layer embeddings or large n-gram lookup tables indexed by token ID (may reside in RAM or on NVMe).
- **Hyper-Connections** — multi-stream residual architecture (e.g., 4 parallel residual streams with mix and inject steps in `Qwen3.8-Flash-Next`).
- **MTP (`Multi-Token Prediction`)** — auxiliary draft layer predicting multiple future tokens for speculative decoding.
- **M-RoPE / p-RoPE** — positional rotation variants: multidimensional (text/image/video) and proportional (rotating a subset of head dimensions).
