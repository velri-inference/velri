#!/usr/bin/env python3
"""Compare graph W8A8 against the archived C23 implementation on Linux."""
import argparse
import csv
import fcntl
import shlex
import subprocess
import tempfile
from pathlib import Path

BASE = "bf3fcc9dca2f9fe44dbf441132e0348f7740db3d"
ROW_MAJOR = "782d72b22c7e7fa0a3a8ea4fd6fdbb87f16260ed"
LEGACY = "dd1ed3cbad86d102c87f2fa17200d5ba55f014fa"
CASES = [
    # rows, output channels, input channels, distinct weight buffers, timed operations
    (128, 2048, 2048, 1, 64),
    (64, 4096, 4096, 1, 32),
    (256, 4096, 4096, 1, 16),
    (1, 4096, 4096, 12, 48),
]
HEADER = ("impl", "rep", "rows", "m", "k", "threads", "ring", "iterations", "ms", "tops", "gbps")


def build(repo, revision, dest, cc, jobs, archived=False, packed_api=False, fixed_warmup=False):
    dest.mkdir()
    tar = dest / "source.tar"
    subprocess.run(["git", "-C", repo, "archive", "--output", tar, revision, "velri-lib"], check=True)
    subprocess.run(["tar", "-xf", tar, "-C", dest], check=True)
    tar.unlink()
    src = dest / "velri-lib"
    if archived and fixed_warmup:
        # Reuse the archive's numerical checks and timed harness; match the graph warmup.
        driver = src / "tools/bench_cpu.c"
        source = driver.read_text()
        marker = "    double samples[9];"
        assert source.count(marker) == 1
        driver.write_text(source.replace(marker, """    double warm = now();
    do {
        for (size_t i = 0; i < ring; ++i) {
            a.weights = weights[i];
            check(velri_linear_typed_run_parallel(&plan, &a, pool, threads));
        }
    } while (now() - warm < 0.25);
""" + marker))
    out = dest / "build"
    options = ["-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_C_COMPILER={cc}", "-DVELRI_LIB_BUILD_TESTS=OFF"]
    if archived:
        options.append("-DVELRI_BUILD_CPU_BENCHMARK=ON")
    subprocess.run(["cmake", "-G", "Ninja", "-S", src, "-B", out, *options], check=True)
    target = "bench_cpu" if archived else "velri"
    subprocess.run(["cmake", "--build", out, "--target", target, "-j", str(jobs)], check=True)
    if archived:
        return out / target
    exe = dest / "graph"
    driver = Path(__file__).resolve().with_name("graph.c")
    defines = []
    if packed_api:
        defines.append("-DVELRI_BENCH_PACKED_API")
    if fixed_warmup:
        defines.append("-DVELRI_BENCH_FIXED_WARMUP")
    subprocess.run([cc, *defines, "-std=c23", "-O3", "-I", src / "include", "-I", src / "src/platform",
                    driver, out / "libvelri.a", "-pthread", "-ldl", "-lm", "-o", exe], check=True)
    return exe


def measure(binaries, labels, output, cpus="0-15", workers=16, include_setup=False):
    """Rotate execution order over three independent processes per implementation/shape."""
    lockpath = Path(tempfile.gettempdir()) / "velri-w8-perf.lock"
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with output.open("x", newline="", buffering=1) as stream:
            results = csv.writer(stream, lineterminator="\n")
            results.writerow((*HEADER, "bind_or_pack_ms") if include_setup else HEADER)
            for rep in range(3):
                for rows, m, k, ring, iterations in CASES:
                    for label in labels[rep % len(labels):] + labels[:rep % len(labels)]:
                        command = ["taskset", "-c", cpus, binaries[label]]
                        if label.startswith("legacy-"):
                            params = ["w8a8", "auto", rows, m, k, workers, int(label == "legacy-prepared"),
                                      ring, iterations]
                        else:
                            params = [rows, m, k, workers, ring, iterations]
                            if label == "graph-packed":
                                params.append("packed")
                        command = list(map(str, [*command, *params]))
                        print(shlex.join(command), flush=True)
                        raw = subprocess.check_output(command, text=True)
                        row = next(csv.reader([raw.strip()]))
                        fields = [label, rep, rows, m, k, workers, ring, iterations, row[8], row[11], row[12]]
                        if include_setup:
                            fields.append(row[15])
                        results.writerow(fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="local velri-inference/velri checkout with the archive branch")
    parser.add_argument("--candidate", help="new graph git revision; omit for the original three-way series")
    parser.add_argument("--compare-packed", action="store_true",
                        help="compare candidate packed and raw graphs, previous graph and archived prepared C")
    parser.add_argument("--output", required=True, type=Path, help="new CSV file")
    parser.add_argument("--cpus", default="0-15", help="Linux taskset CPU list")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=6, help="build parallelism")
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="velri-w8a8-") as tmp:
        root = Path(tmp)
        old = build(args.repo, LEGACY, root / "legacy", args.cc, args.jobs,
                    archived=True, fixed_warmup=args.compare_packed)
        if args.compare_packed:
            previous = build(args.repo, ROW_MAJOR, root / "previous", args.cc, args.jobs, fixed_warmup=True)
            candidate = build(args.repo, args.candidate or "HEAD", root / "new", args.cc, args.jobs,
                              packed_api=True, fixed_warmup=True)
            binaries = {"graph-packed": candidate, "legacy-prepared": old,
                        "graph-new": candidate, "graph-previous": previous}
            measure(binaries, list(binaries), args.output, args.cpus, args.workers, include_setup=True)
            return
        base = build(args.repo, BASE, root / "base", args.cc, args.jobs)
        binaries = {"legacy-raw": old, "graph-base": base, "legacy-prepared": old}
        if args.candidate:
            candidate = build(args.repo, args.candidate, root / "new", args.cc, args.jobs)
            binaries = {"graph-new": candidate, "graph-base": base, "legacy-raw": old, "legacy-prepared": old}
        measure(binaries, list(binaries), args.output, args.cpus, args.workers)


if __name__ == "__main__":
    main()
