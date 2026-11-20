# Shared CPU kernels

Start here when changing the common arithmetic, rather than editing one copy
per instruction set.

| Source | Responsibility |
| --- | --- |
| `gemv.c` / `gemv.inc` | FMA GEMV traversal for F64 and FP32 accumulators, with raw or prepared weights |
| `elementwise.c` | Weighted addition and multiplication, including in-place and streaming paths |
| `../simd/x86.h` | Vector primitives and tuning constants for AVX2 and AVX-512 |
| `../../../../cmake/gemm.cmake` | Generates the FMA register update for the current GEMM tiles |

`cmake/cpu.cmake` compiles the same C sources for each supported instruction set.
GEMV selects the input and weight types before its reduction loop. The compiler
then specializes vector loads and arithmetic; there are no per-element function
pointer calls. `gemv.inc` is instantiated for float and double accumulators.

GEMM's generated register update has the same instruction order as the previous
handwritten sequences. Panel packing, cache traversal and fused output handling
still live beside the target microkernels in `linear_avx*.c`. Native BF16 and INT8
dot products keep their own arithmetic and packed layouts. This is not a claim
that every CPU operation is already shared.

The portable C implementation remains the path for ARM64 and RISC-V. A future
vector adapter must supply the operations needed by a shared kernel; x86 vector
types and flags must not enter the portable build or public headers.

See [English](../../../../docs/EN/cpu-kernels.md) or
[Русский](../../../../docs/RU/cpu-kernels.md) for the supported formats and tests.
