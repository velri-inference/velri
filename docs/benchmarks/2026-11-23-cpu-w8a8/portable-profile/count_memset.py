"""Count zero fills during one graph invocation. Requires Linux x86-64, Clang, nm, taskset."""
import argparse
from bisect import bisect_right
from collections import Counter
import csv
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("repo", type=Path)
p.add_argument("libvelri", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--cpus", default="0-15")
p.add_argument("--workers", default="16")
p.add_argument("--cc", default="clang")
args = p.parse_args()
work = args.out.resolve().parent
work.mkdir(parents=True, exist_ok=True)
exe = work / "count-memset"
lib = args.repo / "velri-lib"
src = args.repo / "docs/benchmarks/2026-11-23-cpu-w8a8/graph.c"
subprocess.run([
    args.cc, "-std=c23", "-O2", "-fno-pie", "-no-pie", "-DVELRI_BENCH_PACKED_API",
    f"-I{lib / 'include'}", f"-I{lib / 'src/platform'}", str(src),
    str(Path(__file__).with_name("memset_counter.c")), str(args.libvelri),
    "-lm", "-lpthread", "-Wl,--wrap=velri_slot_run,--wrap=memset", "-o", str(exe)
], check=True)
symbols = []
for line in subprocess.check_output(["nm", "-n", str(exe)], text=True).splitlines():
    fields = line.split()
    if len(fields) == 3 and fields[1] in {"T","t"}:
        symbols.append((int(fields[0], 16), fields[2]))
addresses = [addr for addr, _ in symbols]
with args.out.open("w", newline="") as f:
    table = csv.writer(f)
    table.writerow(["rows","n","k","layout","symbol","return_offset","calls","bytes"])
    for rows, n, k in [(128,2048,2048), (64,4096,4096), (256,4096,4096)]:
        for layout in ["raw", "packed"]:
            cmd = ["taskset", "-c", args.cpus, str(exe), str(rows), str(n), str(k),
                   args.workers, "1", "1"]
            if layout == "packed":
                cmd.append("packed")
            run = subprocess.run(cmd, capture_output=True, text=True, check=True)
            calls = Counter()
            sizes = Counter()
            for line in run.stderr.splitlines():
                if line.startswith("site "):
                    _, text_addr, _, count, _, size = line.split()
                    addr = int(text_addr, 16)
                    calls[addr] += int(count)
                    sizes[addr] += int(size)
            for addr in sorted(calls):
                start, name = symbols[bisect_right(addresses, addr) - 1]
                table.writerow([rows, n, k, layout, name, hex(addr-start), calls[addr], sizes[addr]])
