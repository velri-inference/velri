# Portable Slang W8A8 on the Ryzen 9 9950X

The measured Release library was compiled directly from the Velri working tree containing the joined-panel change. The SHA-256 of `velri-lib/src/kernels/slang/ops/linear_cpu.slang` was `5b8e43b495ae6cdd439801ac49aa3c24da689a8894807834404157376d6e78f6`. The AVX-512 AOT object's machine code had SHA-256 `f9b6986ec105e05231a9e29ce22fe80f9df1621cc8299ad8487c214910b4b99d` for both Release and the independently compiled DWARF copy. The host ran Linux on 16 physical cores with Slang 2026.22.1 and Clang 21.1.8. [Measured Slang source](https://github.com/velri-inference/velri-lib/blob/e002dccdbc037a1778889d136595fbb998f8c554/src/kernels/slang/ops/linear_cpu.slang#L425)

## Full graph

| Projection (rows × M × K) | Row-major | Packed | Row-major ms | Packed ms |
| --- | ---: | ---: | ---: | ---: |
| 128 × 2048 × 2048 | 3.743 TOPS | 3.874 TOPS | 0.287 | 0.277 |
| 64 × 4096 × 4096 | 3.458 TOPS | 3.679 TOPS | 0.621 | 0.584 |
| 256 × 4096 × 4096 | 4.035 TOPS | 4.145 TOPS | 2.129 | 2.072 |
| 1 × 4096 × 4096, 192 MiB weight ring | 75.55 GB/s | 62.97 GB/s | 0.222 | 0.266 |

All rates include the public graph dispatch. Inputs and weights are signed INT8, with per-row and per-channel scales and FP32 output. The GEMMs reuse one weight matrix. For GEMV, each graph invocation performs twelve projections against separate 16 MiB matrices. The reported GEMV time and effective bandwidth use one projection as the unit. The driver checks against scalar arithmetic, warms up for at least 250 ms and takes the median of nine batches. Each table entry is the median of three independent processes. The timing executable runs without the sampler. Times and effective GB/s in this capture were recovered from the benchmark's nine-significant-digit TOPS output. [Graph driver](../graph.c) · [Individual measurements](data/timings.csv)

## Where the CPU time goes

| Rows | Layout | GEMM microkernel or GEMV dot | Expand weights | Expand activations | Merge INT32 partials | Worker pool |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Row-major | 76.0% | 6.7% | 3.7% | 2.7% | 8.8% |
| 128 | Packed | 78.9% | 3.8% | 4.3% | 3.0% | 7.4% |
| 64 | Row-major | 67.5% | 13.4% | 3.3% | 2.4% | 12.3% |
| 64 | Packed | 73.2% | 7.5% | 4.4% | 2.9% | 10.6% |
| 256 | Row-major | 80.0% | 4.5% | 5.2% | 3.2% | 5.8% |
| 256 | Packed | 81.7% | 2.4% | 5.4% | 3.1% | 6.0% |
| 1 | Row-major | 92.3% | — | — | — | 7.5% |
| 1 | Packed | 85.1% | — | — | — | 14.8% |

Across these eight cases, three windows each yielded 572,010 instruction-pointer samples with 3,263 missed timer expirations. The largest packed GEMM spent 77.2–82.8% of its running CPU samples in the FP32 microkernel. For one-token decode the hot loop uses integer operations. The packed loop computes four of the 32 output channels on each traversal of K. Its K stride is 32 bytes, while the row-major K stride is one byte. The Release code assembles the packed input vectors with 120 byte insertions and eight individual byte loads per 32 K values and four outputs. The row-major loop widens contiguous bytes in vectors. The packed decoder reaches 83.3% of row-major effective weight throughput in this run. [Hot loop offsets, instructions and sample counts](data/hot-loops.csv)

The Linux sampler interrupts each running thread on its own CPU clock at approximately 1 ms intervals for 1.5 s of warmed graph calls per process. Percentages are medians of three per-process sample shares. The inner GEMM category includes broadcasts, loads and loop control inside the FP32 microkernel. The reported shares identify where running threads were executing. A stalled thread can still sample inside an arithmetic loop. Hardware PMU access was denied on this host (`perf_event_paranoid=4`), so there is no measured attribution to cache misses, memory stalls or DRAM traffic. [Per-run sample counts](data/categories.csv) · [Median, minimum and maximum shares](data/summary.csv) · [Original IP histograms](data/sample-ips.tar.gz)

In an independent worker trace of the current 256 × 4096 × 4096 graph, the packed path reached 0.305, 1.194 and 2.141 TOPS on one, four and eight physical cores. At 16 cores the three instrumented runs gave 1.802, 1.432 and 3.888 TOPS. In the two slow runs, the gap between the first and last kernel return was 2.89 and 3.03 ms. The three ordinary 16-core timings for the same layout gave 3.936, 4.145 and 4.253 TOPS. Use the ordinary timings for throughput. The traces expose variation in when workers finish. [Per-process worker traces](data/worker-runs.csv) · [Scaling medians](data/worker-scaling.csv)

In the slowest trace all 36 calls placed worker 0 and worker 14 on CPU 8 at kernel entry and exit. A worker entered roughly 3 ms later than its peer. In a separate series of three paired runs on the same release, the ordinary graph reached a median 4.290 TOPS and the traced graph reached 4.280. With an individual CPU affinity mask per worker these medians were 4.301 and 4.311. This identifies occasional waiting for workers as a second source of latency. The available traces do not establish why Linux chose that placement. [Paired control runs](data/worker-control.csv) · [Placement observations](data/worker-placement.csv)

## What packing changed

The stored weight format is `[ceil(M / 32)][K][32]`. The AVX-512 GEMM produces a 6 × 64 tile. Previously the unpacker stopped at the end of each 32-column storage panel, zero-filled half the scratch tile and ran the full 64-column loop twice. The current Slang unpacker gathers successive storage panels into one compute tile. It pads only the final incomplete output tile. The hot K loop still uses 24 packed FP32 FMAs, six activation broadcasts and four 64-byte weight loads per K step. There are no accumulator spills inside this loop.

In a matched A/B run with the same inputs, the packed GEMMs at 128, 64 and 256 rows went from **2.14 to 4.04**, **2.04 to 3.92** and **2.21 to 4.25 TOPS**, respectively. In the A/B CSV, `graph` is the previous commit `4885f8a9` and `joined-graph` has the same machine code as the measured working tree. A separate linker counter recorded 131072, 524288 and 524288 calls that each filled 128 bytes of the local weight scratch before the change. All three shapes now record zero calls at that site. The remaining zero fills come from padding the last *activation* row tile, totaling 1, 2 and 2 MiB. The counter measures library calls and bytes written into cacheable scratch. [Three-run A/B measurements](data/packed-comparison.csv) · [Current counts](data/zero-fills.csv) · [Previous counts](data/before-zero-fills.csv)

## Arithmetic ceiling

The current release produced a median 5.515 TFLOP/s in the register-only FP32 calibration. The largest packed graph delivers **75.2% of that measured FP32 rate** in useful multiply/add operations. The separate register-only INT8 VNNI calibration on these cores is **22.274 TOPS**. This graph delivers **18.6% of the INT8 reference**. Reaching 92% would require 20.492 TOPS, 3.72 times the measured FP32 peak. Input storage is already INT8. The operation to change for that target is the FP32 dot product in GEMM. Enabling better lowering of a portable byte dot would address that arithmetic gap. [FP32 reference](data/fp32-ceiling.csv) · [INT8 reference](../ceilings.csv)

## Reproduce from the working tree

Use Linux x86-64 with AVX-512 and a checkout containing the graph driver. The instruction sampler uses GNU binutils, the LLVM_IR Release build and a non-PIE benchmark executable. The selected CPUs must be one hardware thread per physical core. Run from the Velri repository root and use fresh output directories for each source revision.

```sh
profile=docs/benchmarks/2026-11-23-cpu-w8a8/portable-profile
build=$(mktemp -d)
results=$(mktemp -d)
cmake -G Ninja -S velri-lib -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DVELRI_SLANG_CPU_MODE=LLVM_IR -DVELRI_BUILD_BENCHMARKS=ON
cmake --build "$build" --target velri bench_slang_cpu
python3 "$profile/profile_cpu.py" --build "$build" --graph "$profile/../graph.c" --output "$results/ip"
python3 "$profile/profile_scaling.py" --repo . --lib "$build/libvelri.a" --out "$results/scaling"
python3 "$profile/count_memset.py" . "$build/libvelri.a" "$results/zero-fills.csv"
for rep in 1 2 3; do taskset -c 0-15 "$build/bench_slang_cpu" --workers 16 > "$results/ceiling-$rep.txt"; done
```

The profiler disassembles the matching Release executable and resolves inlined functions against the DWARF object. Raw IP samples, symbol offsets, timer overruns and process timings are written to the requested output directory. Worker scaling uses per-thread clocks around the exported Slang calls and the normal Linux process affinity mask. A separate counter wraps `memset` for one graph invocation. Its timing has no role in the reported throughput.
