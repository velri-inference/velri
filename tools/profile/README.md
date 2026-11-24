# Profiling a 64-token session

A Linux Release benchmark build produces `bench_session` and `profile_session`. Both run the same model and public session API. Use `bench_session` for wall time. The profiler observes a single host session, device and queue with GNU linker wrappers. It needs Vulkan compute queue timestamps for the GPU backend.

```sh
for trial in 1 2 3; do
    VELRI_TRACE_FILE="avx2-$trial.trace" VELRI_TRACE_RUN=3 taskset -c 0-15 \
        build/profile_session cpu 16 --isa avx2 --weights model.w8 --warmups 2 --runs 1
done
python3 tools/profile/summarize.py avx2 avx2-{1,2,3}.trace > avx2.csv
```

Repeat with `--isa avx512` or `profile_session vulkan 1`. The CPU list in the example covers the 16 physical cores of the measured Ryzen 9 9950X. Select the intended Vulkan ICD when several are present. Supply `--vk-header path/to/vk_internal.h` to the summarizer when reading traces from another checkout. Keep one independent process and one measured sequence per trace.

CPU time spans each synchronous operator submission, including worker rendezvous. The sampler has a separate host timer. Vulkan records a pair of device timestamps for each actual dispatch in the captured graph, plus one pair for the logits copy and one empty bracket per command buffer. A shader-write to transfer-read dependency before the closing timestamp forces completion of the measured shader on the RTX 5070 Ti. Without a cross-stage dependency, queries on this driver can attribute work to a later dispatch. The profiler serializes otherwise independent work. The reported time includes that boundary. It can be substantial for small tensors. Device timestamps are converted with the physical device's timestamp period and queue-family valid-bit mask. Use the ordinary executable for full-session performance.

## Rates and work

For each operator shape, rates divide total useful operations or logical bytes by summed elapsed time over all calls and trials. Counts are per 64-token sequence. A mixed fused dispatch reports its INT8 dot-product TOPS, with its additional FP32 arithmetic listed separately. A GPU projection fused with RoPE or SwiGLU preserves both operations in its shape label. Arithmetic gaps divide rates by the sustained compute and streaming references in the fixed measurement protocol. Pure data movement uses GB/s. The readback reference is the independently calibrated 57.8066 GB/s sustained GPU-to-pinned-host link.

- One dot product with M outputs and K inputs counts 2MK arithmetic operations. Minimum W8A8 bytes count the signed matrix, quantized input, per-channel and per-row scales, and FP32 output. Fused projections count their shared inputs once and all published outputs.
- RMSNorm counts 4D+3 basic FP32 operations and row quantization counts D+2. The reductions, absolute values, comparisons, conversions and transcendental approximations execute within the measured interval. SwiGLU counts 3D basic operations. A residual add counts D.
- RoPE counts three arithmetic operations per rotated element. For attention, each head counts two operations per key/value component per visible position. Logical traffic includes Q, the FP32 output, the KV cache and the appended rows. The visible sequence length advances through 129–192. These are algorithmic counts; actual memory traffic depends on cache reuse and kernel partitioning.
- Embedding and last-row copy count the input and output bytes. Embedding also reads a token ID. The host greedy sampler counts its logical logits input and output ID. Its implementation can revisit part of the logits array in cache. Readback counts the payload sent across the device/host link once.
