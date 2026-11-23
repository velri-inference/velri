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
LEGACY = "dd1ed3cbad86d102c87f2fa17200d5ba55f014fa"
CASES = [
    # rows, output channels, input channels, distinct weight buffers, timed operations
    (128, 2048, 2048, 1, 64),
    (64, 4096, 4096, 1, 32),
    (256, 4096, 4096, 1, 16),
    (1, 4096, 4096, 12, 48),
]
HEADER = ("impl", "rep", "rows", "m", "k", "threads", "ring", "iterations", "ms", "tops", "gbps")


def build(repo, revision, dest, cc, jobs, archived=False):
    dest.mkdir()
    tar = dest / "source.tar"
    subprocess.run(["git", "-C", repo, "archive", "--output", tar, revision, "velri-lib"], check=True)
    subprocess.run(["tar", "-xf", tar, "-C", dest], check=True)
    tar.unlink()
    src = dest / "velri-lib"
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
    subprocess.run([cc, "-std=c23", "-O3", "-I", src / "include", "-I", src / "src/platform",
                    driver, out / "libvelri.a", "-pthread", "-ldl", "-lm", "-o", exe], check=True)
    return exe


def measure(binaries, labels, output, cpus="0-15", workers=16):
    """Rotate execution order over three independent processes per implementation/shape."""
    lockpath = Path(tempfile.gettempdir()) / "velri-w8-perf.lock"
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with output.open("x", newline="", buffering=1) as stream:
            results = csv.writer(stream, lineterminator="\n")
            results.writerow(HEADER)
            for rep in range(3):
                for rows, m, k, ring, iterations in CASES:
                    for label in labels[rep % len(labels):] + labels[:rep % len(labels)]:
                        command = ["taskset", "-c", cpus, binaries[label]]
                        if label.startswith("legacy-"):
                            params = ["w8a8", "auto", rows, m, k, workers, int(label == "legacy-prepared"),
                                      ring, iterations]
                        else:
                            params = [rows, m, k, workers, ring, iterations]
                        command = list(map(str, [*command, *params]))
                        print(shlex.join(command), flush=True)
                        raw = subprocess.check_output(command, text=True)
                        row = next(csv.reader([raw.strip()]))
                        results.writerow((label, rep, rows, m, k, workers, ring, iterations,
                                          row[8], row[11], row[12]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="local velri-inference/velri checkout with the archive branch")
    parser.add_argument("--candidate", help="new graph git revision; omit for the original three-way series")
    parser.add_argument("--output", required=True, type=Path, help="new CSV file")
    parser.add_argument("--cpus", default="0-15", help="Linux taskset CPU list")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--jobs", type=int, default=6, help="build parallelism")
    parser.add_argument("--cc", default="clang")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="velri-w8a8-") as tmp:
        root = Path(tmp)
        old = build(args.repo, LEGACY, root / "legacy", args.cc, args.jobs, archived=True)
        base = build(args.repo, BASE, root / "base", args.cc, args.jobs)
        binaries = {"legacy-raw": old, "graph-base": base, "legacy-prepared": old}
        if args.candidate:
            candidate = build(args.repo, args.candidate, root / "new", args.cc, args.jobs)
            binaries = {"graph-new": candidate, "graph-base": base, "legacy-raw": old, "legacy-prepared": old}
        measure(binaries, list(binaries), args.output, args.cpus, args.workers)


if __name__ == "__main__":
    main()
