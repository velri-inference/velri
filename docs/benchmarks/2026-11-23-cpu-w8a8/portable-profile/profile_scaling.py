"""Measure full 256x4096x4096 W8A8 graphs and per-worker kernel occupancy on physical cores."""
import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import tempfile

CASES = (1, 4, 8, 16)
LAYOUTS = ("row-major", "packed")
SAMPLES = 9
REPEATS = 3
PROCESS_METRICS = ("graph_ms", "tops", "kernel_wall_pct", "kernel_cpu_pct", "worker_balance_pct",
                   "completion_spread_us", "outside_kernel_envelope_us", "endpoint_migrations")


def compile_profiler(repo, lib, cc, build):
    driver = repo / "docs/benchmarks/2026-11-23-cpu-w8a8/graph.c"
    source = driver.read_text()
    edits = {
        "    double times[9];": "    profile_init(m, 9, iterations);\n    double times[9];",
        '            check_ok(velri_slot_run(slot, nullptr, nullptr), "run");':
            '            profile_graph_begin();\n            check_ok(velri_slot_run(slot, nullptr, nullptr), "run");\n            profile_graph_end();',
        "    velri_slot_destroy(slot);": "    profile_save();\n    velri_slot_destroy(slot);",
    }
    for old, new in edits.items():
        if source.count(old) != 1:
            raise RuntimeError("Graph driver's timed loop changed")
        source = source.replace(old, new)
    declarations = ("#include <stddef.h>\nvoid profile_init(size_t,size_t,size_t);\n"
                    "void profile_graph_begin(void);\nvoid profile_graph_end(void);\nvoid profile_save(void);\n")
    driver_copy = build / "graph.c"
    driver_copy.write_text(declarations + source)
    symbols = subprocess.check_output(["nm", "-g", "--defined-only", lib], text=True)
    flags = []
    for variant in ("base", "avx2", "avx512"):
        name = f"velri_slang_{variant}_linear_w8a8_f32"
        if f" T {name}\n" in symbols:
            flags += [f"-DPROFILE_{variant.upper()}", f"-Wl,--wrap={name}", f"-Wl,--wrap={name}_packed"]
    executable = build / "graph-profile"
    subprocess.run([cc, "-std=c23", "-O3", "-DVELRI_BENCH_PACKED_API", "-DVELRI_BENCH_FIXED_WARMUP",
                    "-I", repo / "velri-lib/include", "-I", repo / "velri-lib/src/platform",
                    driver_copy, Path(__file__).with_name("profile_workers.c"), *flags, lib,
                    "-pthread", "-ldl", "-lm", "-o", executable], check=True)
    return executable


def validate_affinity(cpus):
    if len(cpus) != max(CASES) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Specify 16 available logical CPUs from distinct physical cores")
    roots = [Path(f"/sys/devices/system/cpu/cpu{cpu}") for cpu in cpus]
    ids = [(p.joinpath("topology/physical_package_id").read_text(),
            p.joinpath("topology/core_id").read_text()) for p in roots]
    if len(set(ids)) != len(roots):
        raise ValueError("The supplied CPU list contains SMT siblings")
    def l3_id(p):
        for cache in (p / "cache").glob("index*"):
            if (cache / "level").read_text().strip() == "3":
                return (cache / "shared_cpu_list").read_text().strip()
        raise ValueError("L3 topology unavailable")
    if len({l3_id(p) for p in roots[:8]}) != 1:
        raise ValueError("First 8 requested physical cores must share L3")
    return [l3_id(p) for p in roots]


def reduce_process(rows, n, iterations):
    batches = []
    migrations = 0
    for _, batch in itertools.groupby(rows, lambda r: int(r["call"]) // iterations):
        total_wall = total_cpu = capacity = total_slowest = spread = edges = 0
        calls = 0
        for _, records in itertools.groupby(batch, lambda r: r["call"]):
            w = list(records)
            assert len(w) == n
            graph = int(w[0]["graph_ns"])
            begins = [int(v["kernel_start_ns"]) for v in w]
            ends = [int(v["kernel_end_ns"]) for v in w]
            durations = [end - start for start, end in zip(begins, ends)]
            total_wall += sum(durations)
            total_cpu += sum(int(v["thread_cpu_ns"]) for v in w)
            capacity += n * graph
            total_slowest += n * max(durations)
            spread += max(ends) - min(ends)
            edges += min(begins) + graph - max(ends)
            migrations += sum(v["first_cpu"] != v["last_cpu"] for v in w)
            calls += 1
        assert calls == iterations
        batches.append((100 * total_wall / capacity, 100 * total_cpu / capacity,
                        100 * total_wall / total_slowest, spread / calls / 1e3, edges / calls / 1e3))
    assert len(batches) == SAMPLES
    return [statistics.median(items) for items in zip(*batches)] + [migrations]


def write_worker_summary(out):
    by_process = defaultdict(list)
    with (out / "spans.csv").open() as f:
        for row in csv.DictReader(f):
            group = (row["layout"], int(row["workers"]), int(row["worker"]), int(row["rep"]))
            start = int(row["kernel_start_ns"])
            end = int(row["kernel_end_ns"])
            wall = int(row["graph_ns"])
            by_process[group].append(((end - start) / 1e3, int(row["thread_cpu_ns"]) / 1e3,
                                      start / 1e3, (wall - end) / 1e3))
    by_worker = defaultdict(list)
    for key, spans in by_process.items():
        by_worker[key[:3]].append([statistics.median(col) for col in zip(*spans)])
    with (out / "worker_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "workers", "worker", "kernel_wall_us", "kernel_thread_cpu_us",
                    "before_kernel_us", "after_kernel_us"])
        for key, medians in sorted(by_worker.items()):
            w.writerow([*key, *(statistics.median(col) for col in zip(*medians))])

def tool_metadata(lib, cc):
    return {"model": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                          if line.startswith("model name")),
            "kernel": platform.release(), "lib_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
            "cc": subprocess.check_output([cc, "--version"], text=True).splitlines()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--lib", type=Path, required=True, help="Release libvelri.a from the same checkout")
    parser.add_argument("--out", type=Path, required=True, help="New directory for data")
    parser.add_argument("--cc", default="clang")
    parser.add_argument("--cpus", default=",".join(str(x) for x in range(16)),
                        help="Ordered physical CPU IDs; put cores of one CCD first")
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    cpus = [int(c) for c in args.cpus.split(",")]
    caches = validate_affinity(cpus)
    out = args.out
    out.mkdir(parents=True)
    repo = args.repo.resolve()
    revision = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    kernel_source = repo / "velri-lib/src/kernels/slang/ops/linear_cpu.slang"
    meta = {"checkout_head": revision, "kernel_source_sha256": hashlib.sha256(kernel_source.read_bytes()).hexdigest(),
            "shape": [256, 4096, 4096], "ring": 1,
            "ordered_cpus": cpus, "cpu_l3_shared_list": caches, "iterations_per_sample": args.iterations,
            "samples_per_process": SAMPLES, "processes_per_case": REPEATS,
            "affinity": "process mask; original scheduler and thread pool; one hardware thread per core"}
    meta.update(tool_metadata(args.lib, args.cc))
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    result = []
    prefix = ["rep", "layout", "workers", "cpus"]
    with tempfile.TemporaryDirectory() as tmp, (out / "process.csv").open("w", newline="") as proc_file, \
         (out / "spans.csv").open("w", newline="") as raw, \
         (Path(tempfile.gettempdir()) / "velri-w8-perf.lock").open("a") as lock:
        build = Path(tmp)
        exe = compile_profiler(repo, args.lib.resolve(), args.cc, build)
        trace = build / "span.csv"
        proc = csv.writer(proc_file)
        proc.writerow(prefix + list(PROCESS_METRICS))
        span_writer = csv.writer(raw)
        spans_header = False
        for rep in range(REPEATS):
            for n in CASES[rep:] + CASES[:rep]:
                mask = ",".join(str(x) for x in cpus[:n])
                for layout in (LAYOUTS if (rep + CASES.index(n)) % 2 == 0 else LAYOUTS[::-1]):
                    cmd = ["taskset", "-c", mask, exe, "256", "4096", "4096", str(n), "1", str(args.iterations)]
                    if layout == "packed":
                        cmd.append("packed")
                    env = os.environ | {"VELRI_PROFILE_WORKERS": str(n), "VELRI_PROFILE_TRACES": str(trace)}
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:
                        line = subprocess.check_output(cmd, env=env, text=True).strip()
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
                    fields = next(csv.reader([line]))
                    with trace.open() as f:
                        reader = csv.DictReader(f)
                        if not spans_header:
                            span_writer.writerow(prefix + reader.fieldnames)
                            spans_header = True
                        rows = list(reader)
                        for row in rows:
                            span_writer.writerow([rep, layout, n, mask, *row.values()])
                    values = [float(fields[8]), float(fields[11]), *reduce_process(rows, n, args.iterations)]
                    proc.writerow([rep, layout, n, mask, *values])
                    result.append((layout, n, values))
                    proc_file.flush()
                    print(f"rep {rep + 1}/{REPEATS}, {layout}, {n} cores: {values[1]:.3f} TOPS", flush=True)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layout", "workers", *PROCESS_METRICS, "speedup", "parallel_efficiency_pct"])
        for layout in LAYOUTS:
            one = statistics.median(v[1] for label, n, v in result if label == layout and n == 1)
            for n in CASES:
                matrix = [v for label, workers, v in result if label == layout and workers == n]
                med = [statistics.median(z) for z in zip(*matrix)]
                speedup = med[1] / one
                w.writerow([layout, n, *med, speedup, speedup * 100 / n])
    write_worker_summary(out)


if __name__ == "__main__":
    main()
