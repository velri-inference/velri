#!/usr/bin/env python3
"""Aggregate in-session, single-request dispatch traces. Reads matching Vulkan pipeline IDs from source."""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# GB/s, INT8 TOPS, FP32 TFLOPS; independent sustained calibrations, not fitted to these traces.
REFERENCE = {"avx2": (69.40, 4.779, 2.760), "avx512": (70.50, 22.193, 5.540),
             "vulkan": (838.196, 393.425, 47.410)}


@dataclass
class Work:
    calls: int = 0
    bytes: int = 0
    f32: int = 0
    int8: int = 0
    ns: float = 0.0


def account(operations, pipe, position):
    """Minimum logical tensor bytes, not DRAM transactions. Sum useful arithmetic by precision."""
    ops = [(s.split(":")[0], [int(v) for v in s.split(":")[1].split("x")])
           for s in operations.split("+")]
    b = f = i = 0
    for name, dims in ops:
        d = dims[0]
        if name == "W8A8":
            m, k = dims
            b += m * k + k + 8 * m + 4
            i += 2 * m * k
        elif name == "linear":
            m, k = dims
            b += 4 * (m * k + k + m)
            f += 2 * m * k
        elif name == "quant":
            b += 5 * d + 4
            f += d + 2
        elif name == "rms":
            b += 12 * d
            f += 4 * d + 3
        elif name == "residual-rms":
            b += 20 * d
            f += 5 * d + 3
        elif name == "gated":
            b += 12 * d
            f += 3 * d
        elif name == "rope":
            h, hd, rot = dims
            b += 8 * h * hd + 4 * rot
            f += 3 * h * rot
        elif name == "residual":
            b += 12 * d
            f += d
        elif name == "embedding":
            b += 8 * d + 4
        elif name == "last":
            b += 8 * d
        elif name == "attention":
            h, g, dk, dv, item = dims
            # One new K/V row read as FP32 and stored at the cache's precision.
            append = (4 + item) * g * (dk + dv)
            if pipe in ("KV_APPEND_F32", "KV_APPEND_F16KV"):
                b += append
            else:
                f += 2 * h * (dk + dv) * (position + 1)
                b += 4 * h * (dk + dv) + item * g * (dk + dv) * (position + 1)
                if pipe in ("CPU", "ATTENTION_DECODE_F32_INLINE_APPEND"):
                    b += append
        elif name == "sampler":
            b += 4 * d + 4
        elif name == "readback":
            b += d  # payload crossing the device/host link once
        elif name != "empty":
            raise ValueError(f"missing cost model for {name}")
    names = [name for name, _ in ops]
    if len(ops) > 1 and names[-1] == "residual" and "W8A8" in names:
        b -= 4 * ops[-1][1][0]  # residual reads the projection directly from the accumulator
        names = names[:-1]
        ops = ops[:-1]
    if len(ops) > 1:
        if names == ["rms", "quant"]:
            b -= 4 * ops[0][1][0]  # normalized output is published but consumed in registers
        elif names == ["quant", "W8A8"]:
            b -= ops[0][1][0] + 4  # group zero still publishes the I8 input and scale
        elif names == ["W8A8", "W8A8", "gated"]:
            m, k = ops[0][1]
            b -= k + 4 + 12 * m  # shared activation, SiLU uses register outputs
        elif names == ["W8A8", "W8A8", "W8A8", "rope", "rope"]:
            q, k = ops[0][1]
            kv = ops[1][1][0]
            b -= 2 * (k + 4) + 8 * (q + kv) + 4 * ops[3][1][-1]
        elif all(n == "W8A8" for n in names):
            b -= (len(ops) - 1) * (ops[0][1][1] + 4)
        else:
            raise ValueError(f"missing fused cost model for {operations}")
    return b, f, i


def summarize_file(path, table, pipelines):
    counts = Counter()
    with path.open() as fd:
        records = list(csv.DictReader(fd))
    for n, row in enumerate(records):
        pos = int(row["position"])
        counts[int(row["sequence"]), pos] += 1
        pipe = pipelines[int(row["pipeline"])]
        if pipe == "ATTENTION_DECODE_F32" and int(row["gy"]) > 1:
            # Standalone sliced decode integrates KV append unless preceded by its own append stage.
            prev = records[n-1] if n else None
            if prev is None or "KV_APPEND" not in pipelines[int(prev["pipeline"])]:
                if row["ops"].startswith("attention:"):
                    pipe += "_INLINE_APPEND"
        key = (pipe, row["ops"])
        b, f, i = account(key[1], pipe, pos)
        w = table[key]
        w.ns += float(row["ns"])
        w.bytes += b
        w.int8 += i
        w.f32 += f
        w.calls += 1
    sequences = {seq for seq, _ in counts}
    if len(sequences) != 1 or sorted(pos for _, pos in counts) != list(range(128, 192)):
        raise ValueError(f"{path} must contain exactly one tg64 with prompt128")


def percentage(measured, reference):
    return 100 * (measured / reference - 1) if reference else ""


def metrics(w, device, kernel):
    bw, int_peak, fp_peak = REFERENCE[device]
    # Host sampler always runs on CPU. This stream bandwidth is a system-wide reference.
    if kernel == "host_sampler":
        bw = REFERENCE["avx2"][0]
    if kernel == "logit_readback":
        # Separate link limit: host-device transfer does not use GPU DRAM bandwidth alone.
        bw = 57.8066  # independent 64 MiB pinned device-to-host copies on this PCIe link
    amount = w.int8 or w.f32
    throughput = amount / w.ns / 1e3 if amount else w.bytes / w.ns
    peak = int_peak if w.int8 else fp_peak if w.f32 else None
    roof = min(peak, bw * amount / w.bytes / 1e3) if peak else bw
    return ("TOPS" if w.int8 else "TFLOPS" if w.f32 else "GB/s", throughput, peak,
            percentage(throughput, peak), roof, percentage(throughput, roof))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("device", choices=REFERENCE)
    parser.add_argument("traces", nargs="+", type=Path, help="one trace per independent process")
    parser.add_argument("--vk-header", type=Path, default=Path(__file__).resolve().parents[2]
                        / "velri-lib/src/backend/vulkan/vk_internal.h")
    args = parser.parse_args()
    pipeline_names = re.findall(r"^    PIPE_(\w+)(?: = 0)?,", args.vk_header.read_text(), re.M)
    pipelines = dict(enumerate(pipeline_names))
    pipelines.update({-1: "CPU", -2: "empty_bracket", -3: "logit_readback", -4: "host_sampler"})
    totals = defaultdict(Work)
    for path in args.traces:
        summarize_file(path, totals, pipelines)
    rep = len(args.traces)
    out = csv.writer(sys.stdout)
    out.writerow(("kernel", "operations_and_shape", "calls_per_tg64", "us_per_call", "ms_per_tg64",
                  "unit", "rate", "compute_peak", "delta_compute_pct", "streaming_roof", "delta_roof_pct",
                  "logical_GB_s", "logical_bytes_per_call", "int8_ops_per_call", "fp32_ops_per_call"))
    for (kernel, operations), w in sorted(totals.items(), key=lambda pair: -pair[1].ns):
        prefix = [kernel, operations, w.calls / rep, w.ns / w.calls / 1000, w.ns / rep / 1e6]
        rates = metrics(w, args.device, kernel) if w.bytes else ("", "", "", "", "", "")
        fields = prefix + list(rates) + [w.bytes / w.ns, w.bytes / w.calls, w.int8 / w.calls, w.f32 / w.calls]
        out.writerow([round(v, 6) if isinstance(v, float) else v for v in fields])


if __name__ == "__main__":
    main()
