# W8A8 CPU graph with automatic GCC byte-dot generation

On an AMD Ryzen 9 9950X with 16 workers, all three GEMMs spend most of their active thread CPU time in the integer dot loop. For 256 × 4096 × 4096, the full graph reaches 11.36 TOPS with raw weights and 11.45 TOPS with immutable prepared weights. These rates reach 51.0% and 51.4% of the same host's measured 22.2739 TOPS register-only VNNI ceiling. Within the dot loop, GCC widens one of the 20 output reductions to signed word products.

| Rows × M × K | Raw TOPS | Prepared TOPS | Raw ms | Prepared ms |
| --- | ---: | ---: | ---: | ---: |
| 128 × 2048 × 2048 | 10.440 | 10.532 | 0.10285 | 0.10195 |
| 64 × 4096 × 4096 | 11.234 | 11.696 | 0.19116 | 0.18361 |
| 256 × 4096 × 4096 | 11.365 | 11.455 | 0.75585 | 0.74990 |
| 1 × 4096 × 4096, ring of 12 | 0.144 | 0.152 | 0.23338 | 0.22080 |

The GEMV reads 16 MiB of weights per operation from a 192 MiB working set across 12 distinct weight tensors. Effective weight rates are 71.89 GB/s raw and 75.98 GB/s prepared. The graph runs 12 GEMVs per call. Table times are per GEMV. Packed binding takes a separate median of 2.13 ms per GEMV weight tensor. The 256-row prepared throughput ranges from 11.08 to 11.70 TOPS across the three timing processes. Full min/max and binding times are in [timing-summary.csv](data/timing-summary.csv).

## Paired comparison with the previous release

An additional run alternated the graph executable linked against the working tree (`worktree`) and the measured graph from Velri `9dbcae17` (`timing`). Each of the six GEMM medians improved. The ranges in the two timing series reflect variation between fresh processes on this workstation.

| Rows × M × K | Previous raw | Current raw | Previous prepared | Current prepared |
| --- | ---: | ---: | ---: | ---: |
| 128 × 2048 × 2048 | 3.961 TOPS | 10.190 TOPS | 3.985 TOPS | 10.169 TOPS |
| 64 × 4096 × 4096 | 3.642 TOPS | 10.410 TOPS | 3.952 TOPS | 11.726 TOPS |
| 256 × 4096 × 4096 | 4.187 TOPS | 10.393 TOPS | 4.222 TOPS | 10.576 TOPS |
| 1 × 4096 × 4096, 12 matrices | 81.32 GB/s | 74.71 GB/s | 71.92 GB/s | 79.79 GB/s |

In a second, decode-only paired run, raw changed from 77.36 to 76.08 GB/s and prepared from 65.39 to 71.91 GB/s. Both comparisons use the full 192 MiB weight ring. [Paired process medians](data/paired-graph.csv) · [Decode control](data/decode-control.csv)

## CPU-clock sampling

The following entries are median percentages across three processes, listed as raw / prepared.

| Rows | Dot-product inner loops | Horizontal reductions | Activation correction | Scale/bias/store | Pool and synchronization |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 65.2 / 64.8 | 10.3 / 10.1 | 3.6 / 3.3 | 5.1 / 5.1 | 10.7 / 12.1 |
| 64 | 77.6 / 76.6 | 5.8 / 6.0 | 1.6 / 1.3 | 3.0 / 3.0 | 9.3 / 10.1 |
| 256 | 79.1 / 78.8 | 5.9 / 5.9 | 2.4 / 2.3 | 2.9 / 2.9 | 6.5 / 7.0 |
| 1 | 90.4 / 90.5 | 0.1 / 0.1 | 0.03 / 0.05 | 0.15 / 0.15 | 9.1 / 9.0 |

The remaining IPs occur in kernel control/setup or incomplete source mappings, other executable symbols and shared libraries. These median percentages are computed independently. Per-process counts, elapsed and CPU times, overruns, and min/max percentages are in [categories.csv](data/categories.csv) and [summary.csv](data/summary.csv). On the 256-row shape, the share of raw IPs inside the dot loop ranged from 75.0% to 79.8% and the prepared share from 74.9% to 79.0%.

On GEMM the hottest K-loop is at symbol-relative offsets +0x600…+0x70d raw and +0x5e0…+0x6ec prepared. For each 64 K entries, the 4×5 tile executes 19 ZMM VPDPBUSD and two ZMM VPDPWSSD with signed and unsigned byte widening. These cover 19×64 + 2×32 = 1,280 products. The prepared loop uses nine vector loads. The raw loop also biases incoming signed weight bytes. Its five XOR instructions have memory operands. Immediately after the K-loop GCC emits 180 instructions to fold the 20 accumulators into scalar cell sums. Ring GEMV uses a separate 1×5 block with five VPDPBUSD on 64 K values and no word dots or widening. [Static instruction counts and observed loop samples](data/hot-loops.csv)

On the 256-row prepared run, median shares of all sampled IPs landing on VPDPBUSD, VPDPWSSD and widening/extraction instructions inside dot loops were 32.23%, 2.30% and 11.20%. CPU-clock samples describe interrupted instruction addresses. Instruction overlap, signal delivery and the lack of hardware counters prevent inferring latency or stall causes from those percentages. [Within-loop samples by opcode family](data/inner-loop-ip.csv)

## Direct calls with cache-sized working sets

A second harness calls the raw and prepared entry points from the same Release archive. Each worker repeatedly computes its private panel with K = 4096. The timed entry includes activation correction, row and channel scales, and the output store. Preparation and the public graph dispatcher are outside the timed loop. The harness checks every output against a 64-bit scalar sum before and after timing.

| Rows × output channels | Panels per worker | Active bytes per worker | Raw TOPS, 16 cores | Prepared TOPS, 16 cores |
| --- | ---: | ---: | ---: | ---: |
| 4 × 5 | 1 | 36,980 | 6.167 | 6.164 |
| 4 × 5 | 8 | 295,840 | 5.744 | 5.741 |
| 16 × 20 | 1 | 148,880 | 9.212 | 9.280 |
| 16 × 80 | 1 | 398,720 | 11.211 | 11.445 |

At fixed 4 × 5 geometry, increasing the active data from one L1-sized panel to eight panels reduced throughput by about 7%. On the wider 16 × 80 prepared panel the isolated entry reached 11.445 TOPS, close to 11.455 for the full 256-row graph. The host has 48 KiB L1D and 1 MiB L2 per core. Working-set sizes include the active inputs, scales, and output. Cache occupancy and misses were not measured. [One-core and 16-core process medians](data/resident-summary.csv) · [Individual batches](data/resident-batches.csv)

To reproduce the wider prepared case, build the harness on the benchmark host and run three fresh processes. Each warms for 250 ms, pins one worker to each physical core 0–15 and times five batches. Use the median batch within each process and then the median across processes. The five arguments select workers, rows, output channels, panel count and prepared weights (1) or raw weights (0).

```sh
cc -O3 resident.c /path/to/native-build/libvelri.a -pthread -lm -o /tmp/w8-resident
for rep in 1 2 3
do
    /tmp/w8-resident 16 16 80 1 1
done
```

## Measurement and replay

The native Release build used Slang 2026.22.1 emitting ordinary C++, GCC 15.2.0 at -O3 with AVX-512 VNNI enabled and 512-bit preferred vector width, Clang 21.1.8 for the C graph, under Linux 7.0.0-44-generic. Both graph binaries link the unmodified CPU archive from this build. Affinity 0–15 covers one hardware thread from each physical core. Both layouts use the public automatic variant. The graph checks 32 output cells, applies FP32 per-row and per-channel scales and warms for 250 ms. Use of the prepared layout changes immutable weights to biased contiguous rows during binding.

Timing uses three fresh processes per shape/layout, each returning the median of nine full graph batches. An independent set of three processes per configuration samples the 1.5 s loop of warmed public graph calls. Setup and binding precede this window. Layouts and shapes use seeded shuffles, and measurement processes take an exclusive lock. The standard graph counts two operations per int8 multiply-add. Effective GB/s uses original weight bytes per operation and includes cache hits.

Here perf_event_paranoid=4 prevents hardware PMU access. The existing Linux sampler uses CLOCK_THREAD_CPUTIME_ID with a 1,000,003 ns period per pthread and staggered starting phases. Across 24 processes it recorded 572,933 IPs and 3,028 overruns. The denominator for attribution is delivered samples per process. Pool symbols account for scheduled CPU time in worker/spin/synchronization, while time spent asleep does not advance the thread CPU clock.

The profiler extracts the production GCC archive member, recompiles its generated C++ with the identical flags plus -g2, and requires byte-for-byte equality of its entire 24,420-byte .text before using debug line tables. The Slang source SHA-256 is 1929c67649c20524c1187e598a64519c31a63eaad5467d23777523bba4ae93c6 and the GCC .text SHA-256 is 812b52688da04d94ce5351344a243e8c56c722c0f9384b1d7d5016b9e72906fc. Source, generated C++, executables and machine code were rechecked after the series. [Recorded hashes](data/SHA256SUMS)

Symbol-relative offsets come from the measured executable. Attribution identifies the full backward-edge straight-line dot loops and recognizes explicit horizontal-fold instructions immediately following loop exit. The rest uses GCC's Slang source line tables, including epilogue inlining. Some template locations map to their callers, so the remaining kernel category includes unmatched control and epilogue work. The arithmetic calibration uses three independent register-only 16-core measurements from the same CPU. [Ceiling measurements](../ceilings.csv)

Run from this directory with a standalone velri-lib Release Ninja build and automatic GCC byte-dot AOT enabled. Choose a new or empty output directory. The profiler writes a SHA-256 manifest with paths relative to that directory and packages the IP samples and symbol tables in `samples-and-symbols.tar.gz`.

```sh
python3 profile_dot.py --build /path/to/native-build --graph ../graph.c --out /tmp/w8-profile --all
python3 summarize_timing.py /tmp/w8-profile/timings.csv ../ceilings.csv /tmp/w8-profile/timing-summary.csv
```

To replay the recorded IP attribution independently of the host CPU, the archive contains all 24 IP and call-count logs, the symbol tables, instruction listings, DWARF line mappings and the exact kernel source.

```sh
replay_dir=$(mktemp -d)
tar -xzf data/samples-and-symbols.tar.gz -C "$replay_dir"
python3 attribute_dot.py "$replay_dir" --recorded
python3 summarize_timing.py data/timings.csv ../ceilings.csv "$replay_dir/timing-summary.csv"
for name in categories summary hot-loops top-ips inner-loop-ip timing-summary
do
    cmp "$replay_dir/$name.csv" "data/$name.csv" || exit 1
done
```
