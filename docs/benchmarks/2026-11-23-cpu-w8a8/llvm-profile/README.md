# Paired full-graph W8A8 comparison of two portable Slang lowerings

On the Ryzen 9 9950X the LLVM IR path was faster in all 30 paired full-graph GEMM measurements across five shapes, two layouts and three processes per case. The twelve-matrix GEMV remains sensitive to run-to-run variation. Its raw-weight median was similar and its prepared-weight median declined by 4.3%.

Each value in the table is the median of three independently launched processes. The two irregular shapes exercise different row, column and K tails in the existing 4×5 tiling.

| Rows × output channels × K | Unit | GCC raw | LLVM raw | GCC prepared | LLVM prepared |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 × 2048 × 2048 | TOPS | 10.573 | 11.560 | 10.701 | 11.409 |
| 64 × 4096 × 4096 | TOPS | 10.256 | 12.850 | 10.963 | 12.982 |
| 256 × 4096 × 4096 | TOPS | 10.622 | 11.779 | 10.239 | 12.654 |
| 127 × 2051 × 2049 | TOPS | 8.378 | 8.928 | 7.817 | 10.019 |
| 65 × 4099 × 4093 | TOPS | 6.711 | 9.353 | 6.749 | 8.821 |
| 1 × 4096 × 4096, ring 12 | effective weight GB/s | 74.25 | 74.56 | 77.29 | 73.98 |

The median paired speed changes for the three standard GEMMs were +8.8%, +14.1%, +12.3% raw and +9.2%, +12.7%, +15.8% prepared. These use the median of the three within-round LLVM/GCC ratios. The medians in the table are computed independently for each implementation, so their ratios can differ. For GEMV the median paired changes were −2.9% raw and −5.3% prepared. LLVM won one of three pairs in each GEMV layout. Prepared binding of a 16 MiB matrix in the decode configuration took a median of 2.216 ms on GCC and 2.417 ms on LLVM, separately from execution.

## Separating Slang output from x86 instruction selection

A control experiment compiled the same target-neutral LLVM IR from Slang with Clang 21.1.8 and a Clang 24 snapshot. It changed the downstream compiler and vectorization flags while keeping the public graph, source, and IR fixed. The following are full-graph TOPS for 256 × 4096 × 4096, medians of three independent processes. Both modes request a 512-bit preferred vector width. The explicit mode also sets a loop vectorization factor of 64 input bytes and an epilogue vectorization factor of eight.

| Compiler | Automatic, raw | Automatic, prepared | Explicit, raw | Explicit, prepared |
| --- | ---: | ---: | ---: | ---: |
| Clang 21.1.8 | 4.110 | 3.825 | 1.374 | 1.418 |
| Clang 24 snapshot | 5.003 | 5.009 | 13.017 | 13.093 |

The forced width improved this kernel when LLVM could lower its partial reductions to native byte dots. It degraded the earlier compiler. The CMake probe checks the generated instruction before selecting this path. The control links the compiled Slang object to the same library archive for every case. [All control runs](ir-controls.csv)

On 16 cores with private 36 KiB input sets, the integer portion of the 4 × 5 tile produced all twenty accumulator sums at 15.93 TOPS with GCC and 17.91 TOPS with Clang 24 for prepared weights. This isolated experiment includes reading the arrays and horizontally summing the accumulators. Row corrections, floating-point scales and graph dispatch are outside its timed loop. Repeating the archived register-only calibration on the same CPU gave 22.144 TOPS. The existing comparison target remains 92% of the original 22.2739 TOPS register-only median, or 20.492 TOPS. [Tile measurements](isolated-tile.csv) · [Register calibration](register-check.csv)

An independent correctness probe also found a signedness issue in the Slang 2026.22.1 CPU implementation of `coopVecMatMulPacked`. With unsigned packed input bytes `[128, 1, 255, 4]` and signed weights, all sixteen outputs disagreed with the scalar reference on three compiler paths. With a matrix of ones, a minimal C++-target reproducer returns −124 in every lane. The expected unsigned sum is 388. The CPU implementation casts each extracted byte to `int8_t` for both packed input types. Velri's measured kernel uses ordinary integer multiplication and unsigned modular accumulation. [Slang implementation](https://github.com/shader-slang/slang/blob/v2026.22.1/source/slang/hlsl.meta.slang#L34825-L34865)

Run the packed-input probe on a host with Slang and G++. The checker returns status 1 when the compiler produces the signed result.

```sh
slangc packed-input.slang -target cpp -O3 -o /tmp/packed-input.cpp
g++ -O3 -fPIC -shared /tmp/packed-input.cpp -o /tmp/packed-input.so
python3 check_packed_input.py /tmp/packed-input.so
```

## What was measured

The untouched archives came from the existing standalone GCC Release and aggregate LLVM Release builds. Among their 22 archive members, 21 have identical SHA-256 values. Only slang_cpu_dot.o differs. Both use the same Slang arithmetic, SHA-256 1929c67649c20524c1187e598a64519c31a63eaad5467d23777523bba4ae93c6. Rebuilding each route from the final working tree reproduced the full archive SHA-256 in the manifest. Input library, source and linked executable hashes were checked before and after all 72 timing processes.

Slang 2026.22.1 emitted C++ for GCC 15.2.0 and ordinary LLVM IR for Clang 24.0.0 snapshot 51d7f08b16b6. Both production code generators used the same x86 instruction features, -O3 and a 512-bit vector-width preference. The LLVM build forces a 64-iteration main vector width and uses an eight-iteration vectorized epilogue. The full 4×5 reduction in both actually timed executables was disassembled. GCC emits nineteen 512-bit VPDPBUSD and two VPDPWSSD. LLVM emits twenty 512-bit VPDPBUSD. The LLVM tail loops are recorded separately in dot-loops.csv.

Both timing binaries link the identical committed graph.c with Clang 21.1.8, using the public AUTO CPU path and identical link flags. Additional diagnostic executables used a link wrapper to count calls to both byte-dot exports on a small ragged shape. Both libraries dispatched each weight layout to its corresponding entry point. The two executables that produced timing data were built without these wrappers. Every graph process validates 32 scalar-reference FP32 output cells before measurement, checks the requested weight layout, and applies per-row and per-channel scales.

Linux CPUs 0–15 represent one thread from each of 16 physical cores. Each graph warms for 250 ms, then reports the median of nine timed batches of full public slot calls. Each GEMM graph contains one operation. The GEMV graph contains twelve 16 MiB matrices, so it touches 192 MiB of weights per slot call. The table gives the rate and latency per GEMV. The weight binding, runtime preparation and output tests precede all timed batches.

A seeded random permutation assigns twelve shape/layout configurations to pairs within each repetition. The two implementations are run consecutively within each pair, alternating first position by pair and round. Every individual process uses an exclusive benchmark lock. CPU frequency followed the host governor. The raw per-process file retains execution order, min/max can be calculated from any three values, and the paired summary preserves individual speedup range. Effective weight bandwidth uses original weight bytes, including possible cache hits.

## Reproduce

Build Velri-lib twice with Ninja in Release mode. Use `-DVELRI_SLANG_DOT_CLANG=OFF` for the GCC build and `-DVELRI_SLANG_DOT_CLANG=/path/to/clang` for the LLVM build. Verify that the latter reports `W8A8 byte-dot AOT: LLVM IR` at configure time. The script reads their archives and links new uninstrumented drivers. It also validates the chosen paths before timing. The output directory should be new or empty.

    python3 paired.py --repo /path/to/Velri --gcc /path/to/gcc/libvelri.a --llvm /path/to/llvm/libvelri.a --out /tmp/paired
    python3 summarize.py /tmp/paired
    python3 instructions.py /tmp/paired
    python3 metadata.py /tmp/paired /path/to/gcc/libvelri.a /path/to/llvm/libvelri.a

The raw observations include elapsed time, throughput and binding time. The paired table reports within-round speed changes and ratios of separate medians. [72 process results](process.csv) · [Medians and ranges](medians.csv) · [Paired comparisons](paired.csv) · [Toolchain and flags](tools.csv) · [Archive members](archive-members.csv) · [Dispatch checks](dispatch.csv)
