# Velri

**Velri** is a native neural-network inference engine for Windows, Linux, and Android, designed to run models whose weights may exceed GPU memory. Implementation languages are **C23** and **C++23** behind a public C23 API (no Python or PyTorch at runtime). **CPU and Vulkan** are the first target backends, and **Khronos Slang (`slangc` AOT)** is the unified language for composable GPU and CPU kernels.

The **[Velri-lib](velri-lib/)** computation library lives in this repository and owns tensor views, memory planning, kernel launch graphs, Slang kernels, and device backends. Velri uses `velri-lib` to compose reusable model blocks and architectures and to manage model loading and inference sessions.

Documentation: **[English](docs/EN/README.md)** · **[Русский](docs/RU/README.md)**

---

## Current Status

- **`VelriLib::cpu` (Slang AOT) and `VelriLib::vulkan` (SPIR-V 1.6, opt-in) are implemented and verified in `velri-lib/`:** 23 modular CPU MatOps kernels (`GEMV`/`GEMM` across `F64`/`F32`/`F16`/`BF16`/`W8A8`/`W8A32`, `RMSNorm`, `Add+RMSNorm`, `LayerNorm`, `RoPE`, activations, `GatedMLP`, `Softmax`, `FWHT`, `QuantRow`, format conversions) compiled from shared Slang modules (`common/` + `ops/`) alongside 35 `spirv-val`-verified Vulkan shaders (including `CoopMat` and `Split-K`) and persistent `velri_pool` ([Kernels](velri-lib/docs/EN/kernels.md)).
- The earlier handwritten C23/SIMD CPU backend and deprecated `VelriLib::cpu_linear` target are preserved for reference on branch `archive/c23-cpu-legacy`.
- The compute graph, backend registry, model loaders, and full inference engine are active development work ([Roadmap](docs/EN/roadmap.md)).

## Quick Build & Test

Requires CMake 3.21+, Ninja, Clang 18+ or GCC 14+ (C23 mode), and **Khronos Slang (`slangc`)** on `PATH` (or `-DVELRI_SLANGC=/path/to/slangc`):

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

`slangc` is a build-time AOT tool only (no Slang or LLVM runtime dependency). Default CPU builds require no Vulkan SDK or network access; pass `-DVELRI_ENABLE_VULKAN=ON` to also build `VelriLib::vulkan` and `bench_vulkan`. For standalone library builds and package installation, see [velri-lib/README.md](velri-lib/README.md).

## Design Principles

- **Composable kernels without copy-paste:** share Slang components (`IElement`, `IActivation`, `IEpilogue`, `IKVSource`) across operations and compile one source across target ISAs and GPU APIs via `slangc` AOT — no duplicate AVX2 vs. AVX-512 kernel sources and zero compiler at runtime.
- **Blocks without family branches:** assemble models from reusable blocks and operations; adding a model never hardcodes family names into shared kernels or backends.
- **Values are not allocations:** `K` and `V` are input roles that may share storage or derive from other state; allocators reserve only declared storage.
- **Prepare once, zero heap allocations per step:** pack weights, select AOT kernels, build pipelines, and allocate slots during `prepare`.
- **Verify honestly:** every kernel has an independent C23 reference, memory bounds checks (`guard pages` + sanitizers), and full-call benchmarks compared against measured ceilings.

## Documentation

English and Russian pages stay in 1:1 sync with reciprocal links on every page.

| English | Русский | Contents |
| :--- | :--- | :--- |
| [Reading Guide](docs/EN/README.md) | [Карта документации](docs/RU/README.md) | Overview and navigation |
| [Architecture & Stack](docs/EN/architecture.md) | [Архитектура и стек](docs/RU/architecture.md) | Engine pipeline, build, and dependencies |
| [Blocks & Configuration](docs/EN/blocks-and-config.md) | [Блоки и конфигурация](docs/RU/blocks-and-config.md) | Model composition and JSON placement schema |
| [Public C API](docs/EN/api.md) | [Внешний C API](docs/RU/api.md) | `Context`, `Model`, `Program`, `Session`, CLI/HTTP |
| [Performance Methodology](docs/EN/performance.md) | [Методика измерений](docs/RU/performance.md) | TFLOPS/TOPS, GB/s, ceilings, and tables |
| [Roadmap](docs/EN/roadmap.md) | [План работ](docs/RU/roadmap.md) | Stages and acceptance criteria |
| [Glossary](docs/EN/glossary.md) | [Словарь](docs/RU/glossary.md) | Velri, `velri-lib`, Slang, and model terms |
| [Model Architectures](docs/EN/models/README.md) | [Архитектуры моделей](docs/RU/models/README.md) | Family references (`Gemma 4`, `Qwen3.5`, `Qwen3.8`, ...) |
| [Velri-lib Architecture](velri-lib/docs/EN/architecture.md) | [Архитектура Velri-lib](velri-lib/docs/RU/architecture.md) | Library boundary, graph, storage/K-V, backend registry |
| [Slang-First Kernels](velri-lib/docs/EN/kernels.md) | [Кернелы на Slang](velri-lib/docs/RU/kernels.md) | Composable Slang AOT for Vulkan and CPU |
| [Execution & Sync](velri-lib/docs/EN/execution.md) | [Исполнение и синхронизация](velri-lib/docs/RU/execution.md) | `prepare`, UMA budgets, hybrid CPU+Vulkan, cancellation |
