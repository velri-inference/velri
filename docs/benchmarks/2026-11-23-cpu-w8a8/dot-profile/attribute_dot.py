"""Map Release CPU-clock samples into the actual GCC byte-dot path."""

import argparse
import csv
import re
import subprocess
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
parser.add_argument(
    "--recorded", action="store_true", help="Use archived binutils symbol/line tables"
)
args = parser.parse_args()
out = args.out.resolve()
profiles_on_disk = sorted(out.glob("profile-*.ip"))
if not profiles_on_disk:
    parser.error(
        f"No profile-*.ip samples in {out}. Extract the recorded archive or collect a profile first."
    )


def lines(report, *cmd):
    path = out / report
    if args.recorded:
        if not path.is_file():
            parser.error(f"Missing recorded table: {path}")
        text = path.read_text()
    else:
        text = subprocess.check_output(list(map(str, cmd)), cwd=out, text=True)
        path.write_text(text)
    return text.splitlines()


def symbols(obj):
    table = {}
    for line in lines(f"{obj.name}.nm", "nm", "-S", "-n", obj):
        fields = line.split()
        if len(fields) == 4 and fields[2] in ("t", "T"):
            table[fields[3]] = int(fields[0], 16), int(fields[1], 16)
    return table


prod = symbols(out / "graph")
dwarf = symbols(out / "dwarf.o")
targets = ["velri_slang_dot_linear_w8a8_f32", "velri_slang_dot_linear_w8a8_f32_packed"]
syms = sorted((address, size, name) for name, (address, size) in prod.items())
starts = [a for a, s, n in syms]


def lookup(pc):
    i = bisect_right(starts, pc) - 1
    if i >= 0:
        a, s, n = syms[i]
        if pc < a + s:
            return n, pc - a
    return "plt/unknown", pc


def disassembly(name):
    ops = {}
    for ln in lines(
        f"{name}.asm",
        "objdump",
        "-dw",
        "--no-show-raw-insn",
        f"--disassemble={name}",
        "graph",
    ):
        m = re.match(r"\s*([0-9a-f]+):\s+(\w+)\s*(.*)", ln)
        if m:
            ops[int(m[1], 16) - prod[name][0]] = m[2], m[3]
    return ops


operations = {sym: disassembly(sym) for sym in targets}


def blocks(sym):
    # Each leaf backward edge visits a single straight-line inner loop.
    ops = operations[sym]
    ans = []
    for end, (op, args) in ops.items():
        target = re.match(r"([0-9a-f]+) <", args)
        if not op.startswith("j") or not target:
            continue
        start = int(target[1], 16) - prod[sym][0]
        if not 0 < end - start < 8192:
            continue
        body = Counter(op for pc, (op, _) in ops.items() if start <= pc < end)
        if body and not any(op.startswith("j") for op in body):
            ans.append((start, end, body))
    return ans


inner = {sym: blocks(sym) for sym in targets}
dot_ranges = {
    sym: [
        (a, b) for a, b, ops in inner[sym] if any(op.startswith("vpdp") for op in ops)
    ]
    for sym in targets
}

profiles = {}
metadata = {}
addresses = set()
for f in profiles_on_disk:
    if not f.with_suffix(".log").is_file():
        parser.error(f"Missing call-count log for {f.name}")
    data = f.read_text().splitlines()
    if not data or not data[0].startswith("# sample="):
        parser.error(f"Missing sample header in {f.name}")
    metadata[f] = dict(re.findall(r"(\w+)=([\d.]+)", data[0]))
    counts = Counter()
    for row in data[1:]:
        obj, pc, n = row.split()
        if obj == "thread":
            continue
        sym, off = lookup(int(pc, 16)) if obj == "graph" else ("external", 0)
        counts[sym, off] += int(n)
        if sym in targets:
            addresses.add(dwarf[sym][0] + off)
    profiles[f] = counts
if not addresses:
    raise RuntimeError(
        "No samples in the GCC byte-dot path. Check auto-dispatch and the input build."
    )
frames = {}
for row in lines(
    "sample-locations.txt",
    "addr2line",
    "-afi",
    "-s",
    "-e",
    out / "dwarf.o",
    *map(hex, sorted(addresses)),
):
    if row.startswith("0x"):
        ip = int(row, 16)
        frames[ip] = []
    else:
        frames[ip].append(row)
source = (out / "dot.snapshot.slang").read_text().splitlines()


def find_line(sub, start=0):
    for i, line in enumerate(source, 1):
        if i > start and sub in line:
            return i
    parser.error(
        f"Kernel snapshot no longer contains {sub!r}. Update its source attribution."
    )


# Distinct Slang lines can collapse into one template-call location in generated C++.
# Keep such remainder/loop-setup samples in kernel_other.
correction_first = find_line("uint sum = 0")
correction_last = find_line("correction[r] =", correction_first)
ep_first = find_line("float sum = float(")
ep_last = find_line("EpQuantScale::apply", ep_first)


def horizontal_sum_offsets(sym):
    # Once the K-loop finishes GCC folds packed i32 lanes into one sum per cell.
    # Exclude scalar address bookkeeping interleaved with folds in narrow tiles.
    result = set()
    for _, edge in dot_ranges[sym]:
        for pc, (op, operands) in operations[sym].items():
            if pc <= edge:
                continue
            if op.startswith(("test", "cmp", "j", "call", "ret")):
                break
            if op.startswith("vextract") or op in (
                "vpaddd",
                "vpsrldq",
                "vphaddd",
                "vmovd",
            ):
                result.add(pc)
    return result


horizontal_sums = {sym: horizontal_sum_offsets(sym) for sym in targets}


def match_group(sym, off):
    if sym == "external":
        return "DSO"
    if sym in (
        "velri_os_cpu_relax",
        "worker_loop",
        "velri_pool_parallel",
        "velri_os_cond_wait",
    ):
        return "pool_spin_sync"
    if sym not in targets:
        return "dispatch_other"
    if any(a <= off <= b for a, b in dot_ranges[sym]):
        return "dot_reduction_loop"
    if off in horizontal_sums[sym]:
        return "horizontal_sum"
    source_lines = frames[dwarf[sym][0] + off][1::2]
    if any("epilogue.slang:" in ln for ln in source_lines):
        return "scale_bias_store"
    for ln in source_lines:
        loc = re.search(r"w8a8_dot\.slang:(\d+)", ln)
        if not loc:
            continue
        no = int(loc[1])
        if correction_first <= no <= correction_last:
            return "X_correction"
        if ep_first <= no <= ep_last:
            return "scale_bias_store"
    return "kernel_other"


categories = [
    "dot_reduction_loop",
    "horizontal_sum",
    "X_correction",
    "scale_bias_store",
    "kernel_other",
    "pool_spin_sync",
    "dispatch_other",
    "DSO",
]
groups = defaultdict(list)
hot = defaultdict(Counter)
with (out / "categories.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        [
            "rep",
            "rows",
            "layout",
            "samples",
            "overruns",
            "wall_s",
            "user_s",
            "sys_s",
            "calls",
            *categories,
        ]
    )
    for file, ticks in profiles.items():
        _, rep, r, layout = file.stem.split("-")
        key = int(r), layout
        counts = Counter()
        for (s, off), n in ticks.items():
            counts[match_group(s, off)] += n
            hot[key][s, off] += n
        meta = metadata[file]
        total = int(meta["sample"])
        if total <= 0 or total != sum(counts.values()):
            parser.error(f"Inconsistent delivered sample count in {file.name}")
        call_count = re.search(
            r"profiled (\d+) hot calls", file.with_suffix(".log").read_text()
        )
        if not call_count:
            parser.error(f"Missing hot-call count in {file.with_suffix('.log').name}")
        writer.writerow(
            [
                rep,
                r,
                layout,
                total,
                meta["overruns"],
                meta["wall"],
                meta["user"],
                meta["system"],
                call_count[1],
                *(counts[c] for c in categories),
            ]
        )
        groups[key].append((total, counts))
with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        [
            "rows",
            "layout",
            "runs",
            "samples",
            "category",
            "median_pct",
            "min_pct",
            "max_pct",
        ]
    )
    for (rows, layout), runs in sorted(groups.items()):
        for c in categories:
            shares = [100 * cnt[c] / n for n, cnt in runs]
            writer.writerow(
                [
                    rows,
                    layout,
                    len(runs),
                    sum(n for n, _ in runs),
                    c,
                    *(f"{fn(shares):.3f}" for fn in (median, min, max)),
                ]
            )
with (out / "hot-loops.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        [
            "rows",
            "layout",
            "symbol",
            "first",
            "back_edge",
            "loop_samples",
            "all_samples",
            "static_byte_dots",
            "static_word_dots",
            "static_body",
        ]
    )
    for key, ticks in sorted(hot.items()):
        rows, layout = key
        target = targets[int(layout == "packed")]
        candidates = []
        for start, end, body in inner[target]:
            n = sum(
                n
                for (s, off), n in ticks.items()
                if s == target and start <= off <= end
            )
            candidates.append((n, start, end, body))
        total = sum(ticks.values())
        for n, start, end, body in sorted(candidates, key=lambda b: b[0], reverse=True)[
            :3
        ]:
            if not n:
                continue
            writer.writerow(
                [
                    rows,
                    layout,
                    target,
                    hex(start),
                    hex(end),
                    n,
                    total,
                    body["vpdpbusd"],
                    body["vpdpwssd"],
                    " ".join(f"{op}={cnt}" for op, cnt in body.items()),
                ]
            )
with (out / "top-ips.csv").open("w", newline="") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(
        [
            "rows",
            "layout",
            "symbol",
            "offset",
            "instruction",
            "operands",
            "samples",
            "category",
        ]
    )
    for key, ticks in sorted(hot.items()):
        for (s, off), n in ticks.most_common(16):
            op, operands = operations.get(s, {}).get(off, ("", ""))
            writer.writerow([*key, s, hex(off), op, operands, n, match_group(s, off)])


# A sampled address identifies where the thread was interrupted. Superscalar
# execution and interrupt skid preclude charging latency to the instruction.
def opcode_family(op):
    if op.startswith("vpdpb"):
        return "byte_dot"
    if op.startswith("vpdpw"):
        return "word_dot"
    if op.startswith(("vpmovsx", "vpmovzx", "vextract")):
        return "widen_extract"
    if op.startswith("vpxor"):
        return "raw_bias"
    if op.startswith("vmov"):
        return "vector_move"
    return "other"


families = ("byte_dot", "word_dot", "widen_extract", "raw_bias", "vector_move", "other")
with (out / "inner-loop-ip.csv").open("w", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        ["rep", "rows", "layout", "dot_loop_samples", "all_samples", *families]
    )
    for file, ticks in profiles.items():
        _, rep, rows, layout = file.stem.split("-")
        counts = Counter()
        for (sym, offset), n in ticks.items():
            if sym in targets and any(
                start <= offset <= end for start, end in dot_ranges[sym]
            ):
                counts[opcode_family(operations[sym][offset][0])] += n
        writer.writerow(
            [
                rep,
                rows,
                layout,
                sum(counts.values()),
                sum(ticks.values()),
                *(counts[group] for group in families),
            ]
        )
