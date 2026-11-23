"""Replay the published W8A8 public-graph schedule using four Release archives."""
import argparse
import csv
import fcntl
import subprocess
import tempfile
from itertools import groupby
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    cli = argparse.ArgumentParser(description=__doc__)
    for revision in ("production", "candidate"):
        for aot in ("clang", "gcc"):
            cli.add_argument(f"--{revision}-{aot}", type=Path, required=True)
    cli.add_argument("--cpus", default="0-15", help="Linux affinity mask for 16 physical cores")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    lib = here.parents[3] / "velri-lib"
    with tempfile.TemporaryDirectory() as tmp:
        drivers = {}
        for revision in ("production", "candidate"):
            for aot in ("clang", "gcc"):
                archive = getattr(args, f"{revision}_{aot}")
                driver = Path(tmp) / f"{revision}-{aot}"
                with open("/tmp/velri-w8-perf.lock", "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    subprocess.run([
                        "clang", "-std=c23", "-O3", "-DVELRI_BENCH_FIXED_WARMUP",
                        "-DVELRI_BENCH_PACKED_API", "-I", lib / "include", "-I", lib / "src/platform",
                        here.parent / "graph.c", archive, "-ldl", "-lm", "-pthread", "-o", driver,
                    ], check=True)
                drivers[revision, aot] = driver
        with (here / "process.csv").open() as schedule, args.output.open("x") as out:
            runs = csv.DictReader(schedule)
            writer = csv.DictWriter(out, fieldnames=runs.fieldnames, lineterminator="\n")
            writer.writeheader()
            for _, pair in groupby(runs, lambda r: (r["rep"], r["pair"], r["aot"])):
                with open("/tmp/velri-w8-perf.lock", "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    for run in pair:
                        command = ["taskset", "-c", args.cpus, drivers[run["revision"], run["aot"]]]
                        command.extend(run[k] for k in ("rows", "m", "k", "workers", "ring", "iterations"))
                        if run["layout"] == "packed":
                            command.append("packed")
                        rec = next(csv.reader(subprocess.check_output(command, text=True).splitlines()))
                        if len(rec) != 16 or rec[:2] != ["w8a8", "auto"]:
                            raise ValueError(f"Unexpected graph output: {rec}")
                        run.update(ms=rec[8], tops=rec[11], effective_W_GB_s=rec[12], bind_ms=rec[15])
                        writer.writerow(run)
                    out.flush()


if __name__ == "__main__":
    main()
