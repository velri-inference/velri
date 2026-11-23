# CPU W8A8 output panel benchmark

Velri-lib `a5c9bb7` accumulates up to 64 rows by 40 channels before applying scales to adjacent outputs. The baseline is Velri-lib `918477d` (Velri `44cac9ac`). The table compares the two public graphs on a Ryzen 9 9950X. Each cell shows the previous and new median over three independent processes. The first five rows use integer TOPS. The last row uses effective weight bandwidth in decimal GB/s.

| Rows × channels × K | Clang raw | Clang prepared | GCC raw | GCC prepared |
|---|---:|---:|---:|---:|
| 128 × 2048 × 2048 | 12.119 → 13.815 | 12.332 → 13.806 | 10.184 → 11.432 | 10.870 → 11.443 |
| 64 × 4096 × 4096 | 13.251 → 13.812 | 13.231 → 14.176 | 11.485 → 12.259 | 11.822 → 12.422 |
| 256 × 4096 × 4096 | 13.300 → 13.911 | 13.522 → 13.906 | 11.570 → 12.249 | 11.780 → 12.400 |
| 127 × 2051 × 2049 | 10.564 → 11.137 | 10.513 → 11.664 | 8.457 → 10.090 | 8.577 → 10.262 |
| 65 × 4099 × 4093 | 9.112 → 9.840 | 9.640 → 9.744 | 6.752 → 6.967 | 6.926 → 7.025 |
| 1 × 4096 × 4096, ring 12 | 79.91 → 80.38 | 79.45 → 79.55 | 80.58 → 82.64 | 76.72 → 76.52 |

The new kernel was faster in 56 of the 60 individual GEMM pairs. The three raw LLVM pairs for 256 × 4096 × 4096 ranged from −8.3% to +25.9%. The median paired gain there was 4.5%. The fastest regular GEMM reached 14.176 TOPS, or 63.6% of the measured 22.2739 TOPS register-only ceiling. The 92% target for the public graph is 20.492 TOPS. [Register calibration](../ceilings.csv) · [All 144 process results](process.csv) · [Medians and ranges](medians.csv) · [Per-pair changes](paired.csv)

An independent replay of all 144 processes also improved the median in each of the 20 GEMM configurations. The three new prepared LLVM rates were 14.11, 13.53 and 13.90 TOPS for 128, 64 and 256 rows. The 64-row speedup varied from 7.1% in the first series to 2.0% in the replay. [Second series](replay.csv)

## Timing and profile

Each process used Linux CPUs 0–15 and 16 workers, checked 32 output cells, warmed up for at least 250 ms and measured nine batches of `velri_slot_run`. The two revisions ran consecutively within each pair, with alternating order. The GEMV ring rotated through twelve 16 MiB matrices. Binding and packing were timed separately. The runtime selected the native Slang byte-dot path in all eight dispatch checks across the two revisions, compilers and layouts. [Compiler versions and flags](tools.csv) · [Binary hashes](build.csv) · [Dispatch checks](dispatch.csv)

A separate LLVM profile sampled the actual release code on the 128- and 256-row shapes. Each shape and layout used three fresh processes, sampled for 1.5 s each. The new kernel supplied 284,156 instruction-pointer samples. Thread CPU time divided by 16 and normalized per graph call estimates the time in each instruction range. On the 256-row prepared graph, scale and output stores went from 25.00 to 7.39 µs, horizontal sums from 42.41 to 35.22 µs and the main 4 × 5 dot loop from 491.7 to approximately 487 µs. At 128 rows, scale and output stores went from 5.80 to 0.85 µs. These are sampled CPU-time estimates. The throughput table uses elapsed wall time. [Profile by instruction range](profile.csv)

## Reproduce

Build the previous and new Velri-lib revisions in Release mode. For each one, use `-DVELRI_SLANG_DOT_CLANG=/path/to/clang` for the Clang 24 snapshot and `-DVELRI_SLANG_DOT_CLANG=OFF` for the G++ 15 route. Check the W8A8 AOT line in the CMake output. Run the following from this directory with the four resulting archives. The replay links the public graph driver and follows the execution order, shapes and iteration counts recorded in the original series.

```sh
python3 replay.py \
    --production-clang /path/to/old-clang/libvelri.a --production-gcc /path/to/old-gcc/libvelri.a \
    --candidate-clang /path/to/new-clang/libvelri.a --candidate-gcc /path/to/new-gcc/libvelri.a \
    --output /tmp/w8-panel-replay.csv
```

The default affinity is `0-15`. Pass `--cpus` to choose one logical CPU per physical core on a different 16-core host.
