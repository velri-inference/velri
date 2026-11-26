# TinyLlama-1.1B W8A8 64-Token Single-Request Decode Benchmark

[Русский](README.md)

Measured on November 24, 2026: single request (`batch = 1`), 128-token prompt, 64 dependent greedy decode steps through the public `velri_session_step` C23 API on `TinyLlama-1.1B-Chat-v1.0` (`W8A8` linear weights, FP32 embeddings/norms/KV cache).

On an **AMD Ryzen 9 9950X** (16 physical cores), W8A8 projections reached **91.57%** of the measured streaming memory ceiling with **AVX2** and **91.14%** with **AVX-512**. On an **NVIDIA RTX 5070 Ti** over **Vulkan 1.3**, W8A8 projections reached **76.69%** of the streaming ceiling (and **95.84%** on the `32000×2048` output projection), achieving **563.16 tok/s** end-to-end.

- [Per-operator profile tables (RU)](OPERATORS.md) · [Full operator CSV](operators.csv) · [Measurement protocol](../../../tools/decode64-method.md)

---

## End-to-End Generation Speed (64 Tokens)

| Backend | Before (3 runs, ms) | After (3 runs, ms) | Mean Before → After (ms) | Throughput Before → After (tok/s) | Time Reduction | Weight-Only Ceiling (tok/s) | Δ After vs. Ceiling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ryzen 9 9950X, 16 cores, AVX2** | 1127.36 / 1138.86 / 1128.11 | 1044.26 / 1052.13 / 1047.72 | 1131.44 → **1048.04** | 56.57 → **61.07** | −7.37% | 66.98 | −8.83% |
| **Ryzen 9 9950X, 16 cores, AVX-512** | 1126.37 / 1144.36 / 1148.54 | 1069.18 / 1066.35 / 1058.31 | 1139.76 → **1064.62** | 56.15 → **60.12** | −6.59% | 68.04 | −11.65% |
| **RTX 5070 Ti, Vulkan 1.3** | 124.79 / 126.32 / 120.12 | 113.85 / 112.13 / 114.96 | 123.75 → **113.64** | 517.18 → **563.16** | −8.16% | 808.97 | −30.39% |

Each cell reports three separate processes run in interleaved AB/BA/AB order (`before/after`, `after/before`, `before/after`) after two full warm-up runs. Timing covers all 64 calls to `velri_session_step`, device synchronization, logit readback, and greedy token selection. CPU runs pinned one logical CPU per physical core (`0–15`).

---

## Measured Hardware Ceilings

| Backend | Measured INT8 Peak (TOPS) | Measured FP32 Peak (TFLOPS) | Measured Streaming Read (GB/s) | Nominal Interface Bandwidth (GB/s) |
| :--- | :---: | :---: | :---: | :---: |
| **Ryzen 9 9950X, AVX2** | 4.779 | 2.760 | 69.400 | 96 (2× DDR5-6000) |
| **Ryzen 9 9950X, AVX-512** | 22.193 | 5.540 | 70.500 | 96 (2× DDR5-6000) |
| **RTX 5070 Ti, Vulkan** | 393.425 (Tensor) / 97.670 (DP4A) | 47.410 | 838.196 | 896 (256-bit GDDR7 28 Gbps) |

### Weighted W8A8 Projections (All 155 Projections per Token)

| Backend | Actual Before → After (TOPS) | Streaming Ceiling (TOPS) | Share of Ceiling Before → After | Δ After vs. Ceiling |
| :--- | :---: | :---: | :---: | :---: |
| **Ryzen 9 9950X, AVX2** | 0.1273 → **0.1266** | 0.1383 | 92.08% → **91.57%** | −8.43% |
| **Ryzen 9 9950X, AVX-512** | 0.1252 → **0.1280** | 0.1405 | 89.10% → **91.14%** | −8.86% |
| **RTX 5070 Ti, Vulkan** | 1.1628 → **1.2799** | 1.6690 | 69.63% → **76.69%** | −23.31% |

---

## Numerical & Portability Verification

- **Bit-exact before/after parity:** On every backend, before and after builds produced identical `65 × 32,000` FP32 logits; all 18 runs generated the exact same token sequence (`FNV-1a64 = 8a5348dd784b27aa`), matching a PyTorch / Hugging Face reference on **65 / 65** argmax tokens.
- **Independent W8A8 reference parity:** Every single one of `1,015,713` CPU dot-product outputs and `48,899,584` activation codes matched the elementwise integer reference across ISAs.
- **Cross-platform checks:** All 6 tests passed on Release Linux (Clang/GCC), `llvmpipe` Vulkan, base/non-C23 CPU builds, QEMU (`AArch64 Cortex-A53`, `RV64 SiFive U54`, `AVX2 Haswell` without VNNI), Windows x64 (Wine 10), installed `VelriLib 0.7.0`, and Android NDK r29 (`arm64-v8a`, `x86_64`) builds.

For full reproduction commands, SHA-256 digests, and detailed per-operator analysis, see the [complete Russian report](README.md) and [measurement protocol](../../../tools/decode64-method.md).
