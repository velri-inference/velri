# Velri Documentation

[Русский](../RU/README.md)

**Velri** is a native neural-network inference engine for Windows, Linux, and Android, designed to run models whose weights may exceed GPU memory. Implementation languages are **C23** and **C++23** (behind a public C23 API, with no Python or PyTorch at runtime). The first target backends are **CPU and Vulkan**; the unified language for optimized GPU and CPU kernels is **Khronos Slang** (`slangc`, build-time AOT compilation).

Computation, kernel launch graphs, and device backends live in **`velri-lib/`** inside this repository. Documentation is split without duplication: library internals live in `velri-lib/docs/EN/`, while engine, model, and benchmark docs live here.

---

## What Works Today

- The engine builds GPT-2 and Llama/Qwen2 from model descriptors and caller-bound weights. It supports prefill, decode, sampling, and independent sessions. Model file loading and tokenization are planned. [Runtime API](api.md)
- Consumers link `VelriLib::velri` and include `<velri_lib/velri_lib.h>`. The library provides reusable blocks, graph and memory planning, Slang AOT CPU kernels and an optional Vulkan backend. The integration test uses the same public API with an in-tree or installed library. [Architecture and build](architecture.md)

---

## Documentation Map

### Velri Engine (`docs/EN/`)
| Document | Contents |
| :--- | :--- |
| [Architecture & Stack](architecture.md) | Engine pipeline, `velri-lib` boundary, build instructions, and dependencies (C23/C++23, Slang AOT, Vulkan, VMA, yyjson) |
| [Blocks & Configuration](blocks-and-config.md) | Model composition from reusable blocks, K/V roles, and target JSON placement schema |
| [Public C API & Applications](api.md) | `Context`, `Model`, `Program`, `Session`, errors/cancellation, CLI, and HTTP/SSE |
| [Performance Methodology](performance.md) | How we measure speed (TFLOPS/TOPS, GB/s), ceiling references, and comparison tables |
| [Roadmap](roadmap.md) | Milestones from `VelriLib::cpu` and Slang PoC to hybrid CPU+Vulkan and apps |
| [Glossary](glossary.md) | Terminology specific to Velri, `velri-lib`, Slang, and target model architectures |
| [Model Architecture Reference](models/README.md) | Family specifications (`Gemma 4`, `Qwen3.5 / Next`, `Qwen3.8-Flash-Next`, etc.; not a list of working models) |

### Computation Library (`velri-lib/docs/EN/`)
| Document | Contents |
| :--- | :--- |
| [Velri-lib Architecture](../../velri-lib/docs/EN/architecture.md) | Three-layer design, storage/K-V roles, operation graph, backend registry, and RAM transfer bus |
| [Kernels: Slang-First AOT](../../velri-lib/docs/EN/kernels.md) | Composable Slang components with no per-ISA copies, Vulkan SPIR-V/`CoopMat`, Slang CPU, and backend improvement plan |
| [Execution & Synchronization](../../velri-lib/docs/EN/execution.md) | `prepare`, memory budgets & UMA, hybrid GPU↔RAM↔CPU, Vulkan sync, cancellation |
