# W8A8 CPU kernel comparison

The benchmark times signed int8 activations and weights, with a per-row activation scale, a per-output-channel weight scale, and float32 output. It uses one AMD Ryzen 9 9950X, 16 workers and Linux logical CPUs 0–15, one logical CPU from each physical core. The toolchain was Clang 21.1.8, Slang 2026.22.1, CMake 4.2.3 and Ninja 1.13.2. The graph builds used the Slang LLVM IR pipeline and Release flags.

| Series label | Implementation | Timed call |
| --- | --- | --- |
| graph-base | Velri commit `bf3fcc9dca2f9fe44dbf441132e0348f7740db3d` | `velri_slot_run` |
| graph-new | Candidate Velri commit selected with `--candidate` | `velri_slot_run` |
| legacy-raw | Commit `dd1ed3cbad86d102c87f2fa17200d5ba55f014fa`, row-major weights | `velri_linear_typed_run_parallel` |
| legacy-prepared | Same archived code, with weights prepacked once before timing | `velri_linear_typed_run_parallel` |

The archived benchmark is the repository's unmodified `velri-lib/tools/bench_cpu.c`. The graph driver adapts the repository's `bench_slang_cpu.c` graph setup to the archived benchmark's datatype, byte generator, scales and timing. All cases use the automatic ISA selection. The graph driver independently checks 32 output cells; the archived code checks eight. The timed graph includes dispatch.

| Rows | Output channels M | Input channels K | Distinct weight buffers | Operations per timed sample |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 2048 | 2048 | 1 | 64 |
| 64 | 4096 | 4096 | 1 | 32 |
| 256 | 4096 | 4096 | 1 | 16 |
| 1 | 4096 | 4096 | 12 | 48 |

Each process first verifies one result and warms the weight buffers. Each GEMM sample reuses one weight matrix. The GEMV rotates through twelve separately allocated 16 MiB weight matrices, a 192 MiB working set on a machine with 64 MiB total L3. The graph contains all twelve GEMV operations in this case. Each process reports the median of nine timed samples. The runner starts three independent processes for each shape and implementation, rotating implementation order between rounds. The reported result is the median of those three process medians. Allocation, binding, graph setup, archive weight preparation and result checks precede the timed samples.

GEMM throughput counts a multiply and an add as two integer operations. GEMV throughput in the summary uses raw weight bytes divided by elapsed time for all implementations. The `gbps` column of the raw CSV preserves each binary's own byte count, including packed metadata in the legacy prepared case. Rates use decimal GB/s and TOPS.

## Recorded results

The timings below describe the archived AVX-512 VNNI implementation at Velri `782d72b2`. The [byte-dot profile](dot-profile/README.md) measures the current W8A8 CPU graph, including dispatch, source-level samples and paired comparisons. The [earlier FP32 microkernel profile](portable-profile/README.md) records the preceding portable implementation.

Each entry is the median of three independent processes in `final.csv`. The production kernel uses 14 × 32 accumulators, 140 rows per panel, 192 K bytes per panel and four column tiles. The SHA-256 of the measured `velri-lib/src/kernels/slang/w8a8_vnni.slang` is `4063c78f37eff59f3d4871199c30726cbd2d08ceb522e09d6cbed45119fdced9`.

| Shape (rows × M × K) | Unit | Previous graph | New graph | Archived raw | Archived prepared |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 × 2048 × 2048 | TOPS | 4.142 | 13.093 | 12.415 | 13.608 |
| 64 × 4096 × 4096 | TOPS | 3.805 | 11.408 | 7.045 | 14.825 |
| 256 × 4096 × 4096 | TOPS | 4.278 | 13.812 | 8.958 | 17.002 |
| 1 × 4096 × 4096 | effective GB/s | 86.47 | 85.54 | 76.35 | 91.67 |

`baseline.csv` contains the earlier three-way series. `final.csv` contains all four implementations under the same schedule.

The register-only AVX-512 VNNI calibration reached **22.27 TOPS** on 16 physical cores. Each of three fresh `bench_slang_cpu --workers 16` processes ran one warmup and five measurements with CPU affinity 0–15. The reference is the median of the three process peaks recorded in `ceilings.csv`. The full row-major GEMMs reached 51–62% of this measured arithmetic rate, including panel packing, scaling, and dispatch.

The measured library source is Velri-lib commit `0ed62f4631c58d257bcbbe1c8b5222e1d9e563d2`.

## Reproduction

Use Linux, a full checkout of `velri-inference/velri` containing `origin/archive/c23-cpu-legacy`, and a Clang/Slang toolchain on PATH. Run these commands from this directory. The output CSV must be a new path. The script builds source archives in a temporary directory.

```sh
python3 reproduce.py /path/to/Velri --output /tmp/w8-baseline.csv
python3 reproduce.py /path/to/Velri --candidate HEAD --output /tmp/w8-final.csv
python3 summarize.py /tmp/w8-baseline.csv /tmp/w8-final.csv
```

Replace `HEAD` with the candidate's commit to select that exact implementation. On a different CPU, pass `--cpus` and `--workers` for the desired physical cores.

Use `--compare-packed --candidate HEAD` on a revision with the packing API to compare packed and row-major graph calls against the previous graph and archived prepared C. This mode adds a 250 ms warmup for each implementation and reports binding or packing time separately.
