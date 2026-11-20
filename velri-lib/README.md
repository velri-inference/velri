# Velri-lib

`velri-lib/` is the computation library inside the [Velri](../README.md) repository. It owns tensor views, storage and memory planning, kernel launch graphs, composable **Khronos Slang (`slangc` AOT)** kernels, and device backends (**CPU** and **Vulkan** first) behind a public **C23 API** (`include/velri_lib/`). Internals may use C23 and C++23, with no Python or PyTorch dependency.

Documentation: **[English](docs/EN/architecture.md)** · **[Русский](docs/RU/architecture.md)** · **[Kernels (Slang AOT)](docs/EN/kernels.md)** · **[Current CPU Ops](docs/EN/cpu-kernels.md)**

---

## What Works Today

- **`VelriLib::cpu`** (`<velri_lib/kernels.h>`): typed GEMV/GEMM across 11 modes (`F64`, `FP32`, `FP16`, `BF16`, `INT8`, `W8A8`, `W8A32`, row-major and prepared weights), persistent `velri_pool`, `RMSNorm`, `fused Add+RMSNorm`, `LayerNorm`, `RoPE`, activations (`SiLU`, `GELU_tanh`, `ReLU2`), `gated MLP`, `softmax`, `FWHT`, format conversions, and per-row `INT8` quantization. Includes portable C23 (`VELRI_CPU_BASE`) plus x86 AVX2 and AVX-512 paths; CI tests x86-64, AArch64, and RISC-V 64.
- **`VelriLib::cpu_linear`** (`<velri_lib/linear.h>`): deprecated FP32→ordered-FP64 regression target, scheduled for removal once the Slang CPU path stabilizes.
- Modular **Slang AOT** kernels (CPU + Vulkan), the compute graph, backend registry, and Vulkan backend are active work ([kernels.md](docs/EN/kernels.md)). Public C headers are experimental and do not promise a stable binary ABI yet.

---

## Build and Test

Requires CMake 3.21+, Ninja, and Clang 18+ or GCC 14+ with C23 support:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang
cmake --build build
ctest --test-dir build --output-on-failure
```

Standalone builds enable tests by default; as a subproject, tests are opt-in via `VELRI_LIB_BUILD_TESTS`. Existing CMake options:
- `VELRI_CPU_MULTIVERSION` (`ON`): build x86 AVX2/AVX-512 variants alongside portable C23.
- `VELRI_SANITIZE` (`OFF`): enable ASan + UBSan.
- `VELRI_BUILD_CPU_BENCHMARK` (`OFF`): build `bench_cpu` for `VelriLib::cpu`.
- `VELRI_BUILD_BENCHMARK` / `VELRI_BUILD_PARALLEL_BENCHMARK` (`OFF`): legacy `cpu_linear` benchmarks.

---

## Use from Another CMake Project

From source:

```cmake
add_subdirectory(path/to/velri-lib velri-lib-build)
target_link_libraries(my_application PRIVATE VelriLib::cpu)
```

Or from an installed package (`VelriLib 0.1.0 EXACT`):

```sh
cmake --install build --prefix /path/to/velri-lib-install
```

```cmake
find_package(VelriLib 0.1.0 EXACT CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE VelriLib::cpu)
```

Consumers include `<velri_lib/kernels.h>` and link `VelriLib::cpu`. Baseline CPU builds require no Slang compiler, Vulkan SDK, or network downloads.
