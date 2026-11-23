"""Compare full, untouched Velri W8A8 graph calls on existing AOT archives."""

from pathlib import Path
import argparse, csv, fcntl, hashlib, os, random, re, subprocess

here = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", required=True, type=Path)
parser.add_argument("--gcc", required=True, type=Path)
parser.add_argument("--llvm", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
p = args.out.resolve()
if p.exists() and any(p.iterdir()):
    parser.error("Output directory must be new or empty")
p.mkdir(parents=True, exist_ok=True)
repo = args.repo.resolve()
archives = {"gcc": args.gcc.resolve(), "llvm": args.llvm.resolve()}
source = repo / "velri-lib/src/kernels/slang/ops/w8a8_dot.slang"
graph = repo / "docs/benchmarks/2026-11-23-cpu-w8a8/graph.c"


def hashfile(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exec_cmd(*cmd, **kwargs):
    return subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def output(*cmd):
    return exec_cmd(*cmd, capture_output=True, text=True).stdout


flags = [
    "-O3",
    "-std=c23",
    "-DVELRI_BENCH_FIXED_WARMUP",
    "-DVELRI_BENCH_PACKED_API",
    "-I",
    repo / "velri-lib/include",
    "-I",
    repo / "velri-lib/src/platform",
]


def runlocked(*cmd):
    with open("/tmp/velri-w8-perf.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return exec_cmd("taskset", "-c", "0-15", *cmd, text=True, capture_output=True)


snapshots = {"graph.c": hashfile(graph), "w8a8_dot.slang": hashfile(source)}
members = {}
dispatch_counts = []
with (p / "verification.txt").open("w", buffering=1) as log:
    for tag, lib in archives.items():
        archive_members = output("ar", "t", lib).splitlines()
        if "slang_cpu_dot.o" not in archive_members:
            raise ValueError(f"{tag}: dot AOT unavailable")
        members[tag] = {
            m: hashlib.sha256(subprocess.check_output(["ar", "p", lib, m])).hexdigest()
            for m in archive_members
        }
        with (p / (tag + ".o")).open("wb") as f:
            exec_cmd("ar", "p", lib, "slang_cpu_dot.o", stdout=f)
        obj = p / (tag + ".o")
        text = p / (tag + ".text")
        exec_cmd("objcopy", "--dump-section", f".text={text}", obj, os.devnull)
        snapshots[f"{tag}/libvelri.a"] = hashfile(lib)
        snapshots[f"{tag}.text"] = hashfile(text)
        for wrapped in (False, True):
            exe = p / (tag + ("-verify" if wrapped else ".graph"))
            link = (
                [
                    "-Wl,--wrap=velri_slang_dot_linear_w8a8_f32,--wrap=velri_slang_dot_linear_w8a8_f32_packed"
                ]
                if wrapped
                else []
            )
            exec_cmd(
                "clang",
                *flags,
                graph,
                *([here / "verify.c"] if wrapped else []),
                lib,
                *link,
                "-lm",
                "-ldl",
                "-pthread",
                "-o",
                exe,
            )
            if not wrapped:
                snapshots[exe.name] = hashfile(exe)
            else:
                for layout in ("raw", "packed"):
                    result = runlocked(
                        exe,
                        5,
                        37,
                        67,
                        16,
                        1,
                        1,
                        *(["packed"] if layout == "packed" else []),
                    )
                    counts = re.search(
                        r"actual dot: raw=(\d+) packed=(\d+)", result.stderr
                    )
                    assert (
                        counts is not None
                        and int(counts[1 if layout == "raw" else 2]) > 0
                        and int(counts[2 if layout == "raw" else 1]) == 0
                    )
                    dispatch_counts.append((tag, layout, counts[1], counts[2]))
                    print(tag, layout, counts.group(), file=log, flush=True)
        cmp = p / (tag + ".o")
        report = output("readelf", "-p", ".comment", cmp)
        print(tag, report.strip(), file=log)
        buildnext = lib.parent
        while not (buildnext / "CMakeCache.txt").is_file():
            if buildnext == buildnext.parent:
                parser.error(f"No CMakeCache.txt above {lib}")
            buildnext = buildnext.parent
        ninja = output(
            "ninja", "-C", buildnext, "-t", "commands", str(lib.relative_to(buildnext))
        )
        relevant = [
            ln
            for ln in ninja.splitlines()
            if (" -c " in ln and "slang_cpu_dot." in ln)
            or ("-target cpp" in ln and "w8a8_dot.slang" in ln)
        ]
        if tag == "llvm" and not any(
            "force-vector-width=64" in ln and "llvm-ir" in ln for ln in relevant
        ):
            raise ValueError("LLVM selection failed")
        if tag == "gcc" and not any(
            "g++" in ln and "slang_cpu_dot.cpp" in ln for ln in relevant
        ):
            raise ValueError("GCC selection failed")
        print("\n".join(relevant), file=log)
(p / "SHA256SUMS").write_text(
    "".join(f"{digest}  {key}\n" for key, digest in snapshots.items())
)
with (p / "dispatch.csv").open("w", newline="") as verification:
    csv_out = csv.writer(verification, lineterminator="\n")
    csv_out.writerow(["aot", "layout", "dot_raw_calls", "dot_prepared_calls"])
    csv_out.writerows(dispatch_counts)
with (p / "archive-members.csv").open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["member", "gcc_SHA256", "llvm_SHA256", "byte_identical"])
    for member in sorted(members["gcc"].keys() | members["llvm"].keys()):
        a, b = members["gcc"].get(member, ""), members["llvm"].get(member, "")
        w.writerow([member, a, b, str(a == b).lower()])
shapes = [
    (128, 2048, 2048, 1, 64),
    (64, 4096, 4096, 1, 32),
    (256, 4096, 4096, 1, 16),
    (1, 4096, 4096, 12, 48),
    (127, 2051, 2049, 1, 64),
    (65, 4099, 4093, 1, 32),
]
cases = [
    (r, m, k, ring, it, layout)
    for r, m, k, ring, it in shapes
    for layout in ("raw", "packed")
]
with (p / "process.csv").open("w", buffering=1, newline="") as stream:
    table = csv.writer(stream, lineterminator="\n")
    table.writerow(
        [
            "rep",
            "sequence",
            "pair",
            "in_pair",
            "aot",
            "rows",
            "m",
            "k",
            "workers",
            "ring",
            "iterations",
            "layout",
            "ms",
            "tops",
            "effective_W_GB_s",
            "bind_ms",
        ]
    )
    seq = 0
    for rep in range(3):
        schedule = list(enumerate(cases))
        random.Random(412371 + rep).shuffle(schedule)
        for pair, (r, m, k, ring, it, layout) in schedule:
            order = ["gcc", "llvm"] if (pair + rep) % 2 == 0 else ["llvm", "gcc"]
            results = {}
            for idx, tag in enumerate(order):
                a = runlocked(
                    p / (tag + ".graph"),
                    r,
                    m,
                    k,
                    16,
                    ring,
                    it,
                    *(["packed"] if layout == "packed" else []),
                )
                fields = a.stdout.strip().split(",")
                if len(fields) != 16 or fields[:2] != ["w8a8", "auto"]:
                    raise ValueError(a.stdout)
                row = [
                    rep,
                    seq,
                    pair,
                    idx,
                    tag,
                    r,
                    m,
                    k,
                    16,
                    ring,
                    it,
                    layout,
                    fields[8],
                    fields[11],
                    fields[12],
                    fields[15],
                ]
                table.writerow(row)
                seq += 1
                results[tag] = float(fields[12] if r == 1 else fields[11])
            print(
                rep,
                r,
                m,
                k,
                layout,
                f"gcc={results['gcc']:.4f} llvm={results['llvm']:.4f}",
                flush=True,
            )
if (
    hashfile(graph) != snapshots["graph.c"]
    or hashfile(source) != snapshots["w8a8_dot.slang"]
):
    raise ValueError("source changed during measurement")
for tag, lib in archives.items():
    if (
        hashfile(lib) != snapshots[f"{tag}/libvelri.a"]
        or hashfile(p / (tag + ".graph")) != snapshots[tag + ".graph"]
    ):
        raise ValueError(f"{tag} executable or archive changed")
