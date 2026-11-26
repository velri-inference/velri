# Velri — Native C23 & Khronos Slang LLM Inference Engine

**[Website](https://velri-inference.github.io/)** · **[English Docs](docs/EN/README.md)** · **[Русский](docs/RU/README.md)** · **[Velri-lib](https://github.com/velri-inference/velri-lib)** · **[VTQ Quantization](https://github.com/velri-inference/vtq-research)** · **[llms.txt](llms.txt)** · **License: [GPL-3.0](LICENSE)**

**Velri** is an open-source, native neural-network and LLM inference engine for **Linux, Windows, and Android**, designed to run large models whose weights exceed GPU VRAM through explicit memory planning and CPU+GPU offloading. Written in **C23 and C++23** behind a public **C23 API**, Velri compiles composable **Khronos Slang (`slangc`)** kernels ahead of time (AOT) for both **CPU** (x86-64 AVX2, AVX-512 VNNI, ARM64, RISC-V 64) and **Vulkan 1.3 / SPIR-V 1.6** — with **zero Python, PyTorch, CUDA, LLVM, or Slang compiler dependencies at runtime**.

---

## Velri Ecosystem

| Repository | Role | Key Technologies |
| :--- | :--- | :--- |
| **[velri-inference/velri](https://github.com/velri-inference/velri)** *(this repo)* | Inference engine, model composition, placement planner, and C23 runtime (`Context`, `Model`, `Program`, `Session`) | C23/C++23, cascading JSON placement, GPT-2 / Llama / Qwen2 runtime |
| **[velri-inference/velri-lib](https://github.com/velri-inference/velri-lib)** *(also in [`velri-lib/`](velri-lib/))* | Standalone C23 computation library (`VelriLib::velri`) — tensor views, liveness scratch planner, graphs, and backends | 23 Slang AOT CPU kernels, 29 Vulkan 1.3 SPIR-V 1.6 shaders (`CoopMat`, FlashAttention) |
| **[velri-inference/vtq-research](https://github.com/velri-inference/vtq-research)** | **VTQ** (*Velri Trellis Quant*, `.vtq` v2) hardware-co-designed `16×256` trellis weight quantization (`1.5–6.0 bpw`) | Tail-biting Ungerboeck TCQ, Outer-DCT rotation, AVX2 `vpshufb` & GPU DP4A (`36/36` KLD wins vs EXL3 & GGUF) |

---

## Verified Benchmarks

### End-to-End 64-Token Decode (`TinyLlama-1.1B-Chat-v1.0`, W8A8, `batch=1`)

Single request after a 128-token prompt, 64 dependent greedy decode steps through the public `velri_session_step` API (mean of 3 interleaved processes per backend; full methodology and SHA-256 artifacts in **[English](docs/performance/decode64/README.EN.md)** · **[Русский](docs/performance/decode64/README.md)**):

| Backend | Hardware | 64-Token Time | Decode Speed | W8A8 Projection Ceiling Share |
| :--- | :--- | :---: | :---: | :---: |
| **CPU (AVX2)** | AMD Ryzen 9 9950X (16 cores, DDR5-6000) | `1048.04 ms` | **`61.07 tok/s`** | `91.57%` of streaming DRAM ceiling |
| **CPU (AVX-512)** | AMD Ryzen 9 9950X (16 cores, DDR5-6000) | `1064.62 ms` | **`60.12 tok/s`** | `91.14%` of streaming DRAM ceiling |
| **Vulkan 1.3** | NVIDIA RTX 5070 Ti (GDDR7) | `113.64 ms` | **`563.16 tok/s`** | `76.69%` weighted (`95.84%` on `32000×2048`) |

### Dense CPU Kernel Roofline (`Velri-lib`, Ryzen 9 9950X, 16 Physical Cores)

Measured via `bench_slang_cpu --workers 16` (`1024×4096×4096` GEMM and cold-DRAM single-token GEMV, `0 B` hot-path allocations):

| Precision / Mode | `1024×4096×4096` GEMM | Cold-DRAM GEMV | Notes |
| :--- | :---: | :---: | :--- |
| **FP32 (`F32`)** | **`4.02 TFLOPS`** | `66.4–71.3 GB/s` | 5-loop L2-resident BLIS in portable Slang AOT |
| **FP16 → FP32 (`F16`)** | **`4.07 TFLOPS`** (`4.17 TF` at 256 rows) | `66.4–71.3 GB/s` | Shared `IElement` Slang component |
| **BF16 → FP32 (`BF16`)** | **`4.20 TFLOPS`** | `66.4–71.3 GB/s` | Shared `IElement` Slang component |
| **F32 × BF16 (`F32xBF16`)** | **`4.16 TFLOPS`** (`4.29 TF` at 256 rows) | `66.4–71.3 GB/s` | Mixed-precision linear |
| **FP64 (`F64`)** | **`2.02 TFLOPS`** (`2.05 TF` at 256 rows) | `66.4–71.3 GB/s` | Zero-spill AVX-512 BLIS microkernel |
| **INT8 W8A8 (packed)** | **`17.87 TOPS`** (`80.3%` VNNI peak) | `66.4–71.3 GB/s` | Exact deterministic mod $2^{32}$ arithmetic |

---

## Current Status

- **`VelriLib::velri` (alias `VelriLib::core`, v0.7.0) is implemented and verified in [`velri-lib/`](velri-lib/):**
  - Single consolidated C23 library (`<velri_lib/velri_lib.h>` — public headers for types, backends, graphs, and reusable blocks): `Storage` / `TensorView`, operation registry, graph builder, liveness scratch memory planner, UMA-aware memory report, plan-shared weights (`velri_plan_bind_weight_host`), device-resident state with explicit host sync, refcounted lifetimes, 6-group Backend Contract v2 with command-buffer batching, cross-backend RAM transfer bus, and independent concurrent execution slots (`core_contract` passes).
  - MatOps kernels: 23 modular CPU kernels (`GEMV`/`GEMM` incl. FP64, norms, RoPE, activations, `GatedMLP`, `Softmax`, `FWHT`, `QuantRow`) compiled AOT from shared Slang modules (`common/` + `ops/`) alongside 29 `spirv-val`-verified modular Vulkan 1.3 / SPIR-V 1.6 shaders (`src/kernels/slang/vk/`, `CoopMat`, FlashAttention, `Split-K`, `Split-KV`), verified across every CPU ISA and Vulkan GPU by `test_kernels` ([Kernels](velri-lib/docs/EN/kernels.md)).
- The earlier handwritten C23/SIMD CPU backend and deprecated `VelriLib::cpu_linear` target are preserved for reference on branch `archive/c23-cpu-legacy`.
- The public runtime builds GPT-2 and Llama/Qwen2 from descriptors and caller-bound weights, supporting chunked prefill, decode, sampling, and independent concurrent sessions. `.vtq` weight loading, tokenizers, and CLI/HTTP apps follow in [Roadmap](docs/EN/roadmap.md).

---

## Quick Build & Test

Requires CMake 3.21+, Ninja, Clang 18+ or GCC 14+ (C23 mode), and **Khronos Slang (`slangc`)** on `PATH` (or `-DVELRI_SLANGC=/path/to/slangc`):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

`slangc` is a build-time AOT tool only (no Slang or LLVM runtime dependency). Default CPU builds require no Vulkan SDK or network access; pass `-DVELRI_ENABLE_VULKAN=ON` to enable the built-in Vulkan 1.3 backend and `-DVELRI_BUILD_BENCHMARKS=ON` to build `bench_session`, `bench_slang_cpu`, and `bench_vulkan`. For standalone library builds, cross-compilation (Android, Windows, ARM64, RISC-V 64), and package installation, see [velri-lib/README.md](velri-lib/README.md).

With benchmarks enabled, `build/bench_session cpu 16` or `build/bench_session vulkan` times 64 consecutive greedy decode steps on one synthetic FP32 Llama session following a 128-token prompt. To benchmark a real checkpoint, convert a plain-RoPE Llama safetensors snapshot with `python3 tools/prepare_session_weights.py SNAPSHOT weights.w8` (NumPy required), then run `build/bench_session cpu 16 --isa avx2 --weights weights.w8` or `build/bench_session vulkan 1 --weights weights.w8`. See the [measurement protocol](tools/decode64-method.md) and the 9950X / RTX 5070 Ti results ([English](docs/performance/decode64/README.EN.md) · [Русский](docs/performance/decode64/README.md)).

---

## Design Principles

- **Composable kernels without copy-paste:** share Slang components (`IElement`, `IActivation`, `IEpilogue`, `IKVSource`) across operations and compile one source across target ISAs and GPU APIs via `slangc` AOT — no duplicate AVX2 vs. AVX-512 kernel sources and zero compiler at runtime.
- **Blocks without family branches:** assemble models from reusable blocks and operations; adding a model never hardcodes family names into shared kernels or backends.
- **Values are not allocations:** `K` and `V` are input roles that may share storage or derive from other state; allocators reserve only declared storage.
- **Prepare once, zero heap allocations per step:** pack weights, select AOT kernels, build pipelines, and allocate slots during `prepare`.
- **Verify honestly:** every kernel has an independent C23 reference, memory bounds checks (`guard pages` + ASan/UBSan), and full-call benchmarks compared against measured physical hardware ceilings.

---

## Documentation

English and Russian pages stay in 1:1 sync with reciprocal links on every page.

| English | Русский | Contents |
| :--- | :--- | :--- |
| [Reading Guide](docs/EN/README.md) | [Карта документации](docs/RU/README.md) | Overview and navigation |
| [Architecture & Stack](docs/EN/architecture.md) | [Архитектура и стек](docs/RU/architecture.md) | Engine pipeline, build, and dependencies |
| [Blocks & Configuration](docs/EN/blocks-and-config.md) | [Блоки и конфигурация](docs/RU/blocks-and-config.md) | Model composition and JSON placement schema |
| [Public C API](docs/EN/api.md) | [Внешний C API](docs/RU/api.md) | `Context`, `Model`, `Program`, `Session`, CLI/HTTP |
| [Performance Methodology](docs/EN/performance.md) | [Методика измерений](docs/RU/performance.md) | TFLOPS/TOPS, GB/s, ceilings, and tables |
| [TinyLlama Decode64 Report](docs/performance/decode64/README.EN.md) | [Отчёт Decode64](docs/performance/decode64/README.md) | 9950X CPU & RTX 5070 Ti Vulkan W8A8 decode results |
| [Roadmap](docs/EN/roadmap.md) | [План работ](docs/RU/roadmap.md) | Stages and acceptance criteria |
| [Glossary](docs/EN/glossary.md) | [Словарь](docs/RU/glossary.md) | Velri, `velri-lib`, Slang, and model terms |
| [Model Architectures](docs/EN/models/README.md) | [Архитектуры моделей](docs/RU/models/README.md) | Family references (`Gemma 4`, `Qwen3.5`, `Qwen3.8`, ...) |
| [Velri-lib Architecture](velri-lib/docs/EN/architecture.md) | [Архитектура Velri-lib](velri-lib/docs/RU/architecture.md) | Library boundary, graph, storage/K-V, backend registry |
| [Slang-First Kernels](velri-lib/docs/EN/kernels.md) | [Кернелы на Slang](velri-lib/docs/RU/kernels.md) | Composable Slang AOT for Vulkan and CPU |
| [Execution & Sync](velri-lib/docs/EN/execution.md) | [Исполнение и синхронизация](velri-lib/docs/RU/execution.md) | `prepare`, UMA budgets, hybrid CPU+Vulkan, cancellation |

---

## FAQ

- **What language is Velri written in?** C23 and C++23 behind a pure C23 public API, with compute kernels authored in Khronos Slang and compiled ahead of time (`slangc` AOT). Velri is **not** written in Rust and does **not** require CUDA.
- **Does Velri need `slangc`, LLVM, or Python at runtime?** No. `slangc` is a build-time tool only; shipped binaries have zero runtime compiler, Python, or PyTorch dependencies.
- **What quantization format does Velri target?** **VTQ** (*Velri Trellis Quant*, `.vtq` v2) developed in [velri-inference/vtq-research](https://github.com/velri-inference/vtq-research). VTQ supersedes earlier `LTQ` and `.liqk` research prototypes, delivering strictly lower teacher-forced KLD than EXL3 and GGUF across `1.5–6.0 bpw` with `16×256` supertiles shared across CPU (AVX2) and GPU.
- **Can I use the compute library without the full engine?** Yes — [`velri-lib`](https://github.com/velri-inference/velri-lib) builds and installs standalone as `VelriLib::velri` (`find_package(VelriLib 0.7.0 EXACT CONFIG REQUIRED)`).

---

## License & Citation

Licensed under the **[GNU General Public License v3.0](LICENSE)**. If you reference Velri in code, benchmarks, or papers, see [`CITATION.cff`](CITATION.cff).
