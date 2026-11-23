# Prepared W8A8 on the public CPU graph

Velri-lib `73e278e` uses 4 × 6 tiles for batched prepared weights, with up to 48 channels per output panel. The panel uses 12 KiB per active worker. A weight row can be extended to a multiple of 64 bytes if the additional bytes fit within 1/32 of K. The planner reuses space in the existing channel allocation when available. On 2051 × 2049 weights the allocation grew by 69,792 bytes. On 4099 × 4093 and the regular benchmark matrices the size stayed the same. The kernel reads contiguous activations directly. The final tile in a slice uses the logical K.

## Whole-graph throughput

The baseline is Velri `d242720e`, Velri-lib `a5c9bb7`. Clang 24 compiles Slang LLVM IR and G++ 15.2 compiles Slang-generated C++. The table shows previous and new process medians on a Ryzen 9 9950X with 16 workers. GEMM uses integer TOPS and the last row uses effective weight GB/s. Each entry uses three independent processes, each with a 250 ms warmup and nine timed batches. Timing includes dispatch and output scaling. Binding and weight preparation are timed separately. Both revisions and both compilers used the same public graph driver. [144 process observations](process.csv)

| Rows × M × K | LLVM raw | LLVM packed | GCC raw | GCC packed |
|---|---:|---:|---:|---:|
| 128 × 2048 × 2048 | 13.59 → 13.31 | 13.87 → 13.85 | 11.32 → 11.49 | 11.29 → 11.56 |
| 64 × 4096 × 4096 | 13.72 → 13.37 | 13.53 → 14.57 | 12.17 → 12.32 | 12.10 → 12.67 |
| 256 × 4096 × 4096 | 13.35 → 13.50 | 14.17 → 14.35 | 12.21 → 12.23 | 12.37 → 12.81 |
| 127 × 2051 × 2049 | 11.40 → 11.55 | 11.62 → 12.99 | 10.11 → 10.00 | 10.28 → 11.08 |
| 65 × 4099 × 4093 | 9.30 → 9.53 | 9.78 → 12.29 | 7.00 → 6.70 | 6.50 → 11.78 |
| 1 × 4096 × 4096 (12 matrices) | 75.31 → 78.87 | 75.22 → 78.45 | 77.07 → 77.00 | 79.53 → 78.15 |

The best median reached 14.57 TOPS, or 65.4% of the measured 22.2739 TOPS register calibration. The 92% target corresponds to 20.492 TOPS on the full graph. Reproduce using the four Release archives and the existing panel replay script. Its checked-in execution order matches this series. [Register measurements](../ceilings.csv) · [Replay instructions](../panel-profile/README.md#reproduce)

## Why the register and graph rates differ

A separate profile of the **preceding 4 × 5 packed kernel** sampled its compiled inner loop with real inputs. Three processes per configuration used 16 pinned workers. With 4 activation rows and 5 weight rows at K=4096, the 36 KiB inputs per worker yielded 21.42 TOPS inside the sampled loop. The complete isolated entry in that experiment reached 8.47 TOPS with corrections, reduction and output work included. At 256 × 256 × 4096 per worker, the same loop reached 17.21 TOPS. These loop rates are estimates based on sampled thread CPU time. They help distinguish the effects of operand reuse and the additional work in a full operation. [Resident working sets and sampled ranges](resident-loop.csv)

## Slang specialization check

On Slang 2026.22.1, passing a generic function's local static constant as the width of a generic interface implementation produced an incorrect row stride. The expected write indices for raw, packed and explicit-width control are 40, 48 and 48. Both Slang host targets produced 40, 40 and 48 on the probe. The production kernel passes its tile width as an explicit generic parameter. Run `python3 check_stride.py` on Linux with Slang, CMake, Clang and G++ available. Exit code 1 identifies a mismatched write. [Independent probe](generic-stride.slang) · [Checker](check_stride.py)
