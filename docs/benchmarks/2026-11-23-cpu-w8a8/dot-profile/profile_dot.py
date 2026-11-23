"""Profile the current byte-dot AOT object with CPU-clock IPs on native Linux x86-64."""

import argparse
import csv
import fcntl
import hashlib
import os
import random
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

tools = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--graph", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--cpus", default="0-15")
parser.add_argument("--repeat", default=3, type=int)
parser.add_argument(
    "--all", action="store_true", help="Also include the 64/128-row GEMMs"
)
parser.add_argument("--prepare-only", action="store_true")
args = parser.parse_args()
build, graph_src, out = args.build.resolve(), args.graph.resolve(), args.out.resolve()
if args.repeat < 1:
    parser.error("--repeat must be at least 1")
if not (build / "build.ninja").is_file():
    parser.error(f"--build must point to a configured Ninja build: {build}")
sampler = graph_src.parent / "portable-profile/sample_cpu.c"
if not graph_src.is_file() or not sampler.is_file():
    parser.error(f"--graph must point to the benchmark graph next to {sampler}")
if out.exists() and (not out.is_dir() or any(out.iterdir())):
    parser.error(f"--out requires a new or empty directory: {out}")
out.mkdir(parents=True, exist_ok=True)


def run(*cmd, **kwargs):
    try:
        return subprocess.run(list(map(str, cmd)), check=True, text=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        parser.exit(
            1, f"Command failed ({error.returncode}): {shlex.join(error.cmd)}\n"
        )


run("ninja", "-C", build, "libvelri.a")
commands = run(
    "ninja", "-C", build, "-t", "commands", "libvelri.a", capture_output=True
).stdout.splitlines()
compile_commands = [
    shlex.split(stage)
    for line in commands
    for stage in line.split(" && ")
    if " -c " in stage and "slang_cpu_dot.cpp" in stage
]
if len(compile_commands) != 1:
    parser.error(
        "The build must contain one GCC compilation of slang_cpu_dot.cpp. Check the byte-dot AOT configuration."
    )
cc = compile_commands[0]
cpp = build / cc[cc.index("-c") + 1]
obj = build / cc[cc.index("-o") + 1]
lib = build / "libvelri.a"
src = Path(
    next(
        line.split("=", 1)[1]
        for line in (build / "CMakeCache.txt").read_text().splitlines()
        if line.startswith("CMAKE_HOME_DIRECTORY:")
    )
)
if not (src / "include/velri_lib").is_dir():
    src = src / "velri-lib"
dot = src / "src/kernels/slang/ops/w8a8_dot.slang"
inputs = {"dot.snapshot.slang": dot, "slang_cpu_dot.cpp": cpp, "graph.c": graph_src}
for filename, path in inputs.items():
    (out / filename).write_bytes(path.read_bytes())
with (out / "release.o").open("wb") as f:
    subprocess.run(["ar", "p", lib, obj.name], check=True, stdout=f)
run(
    "objcopy",
    "--dump-section",
    f".text={out / 'release.text'}",
    out / "release.o",
    os.devnull,
)
# Keep the real Release flags and compile the same generated source. Redirect
# both outputs, including the depfile Ninja uses for its production object.
debug_cc = cc.copy()
debug_cc[debug_cc.index("-o") + 1] = str(out / "dwarf.o")
if "-MF" in debug_cc:
    debug_cc[debug_cc.index("-MF") + 1] = str(out / "dwarf.d")
for flag in ["-g2", "-g1", "-g2 -fno-var-tracking-assignments"]:
    run(*debug_cc, *flag.split(), cwd=build)
    run(
        "objcopy",
        "--dump-section",
        f".text={out / 'dwarf.text'}",
        out / "dwarf.o",
        os.devnull,
    )
    if (out / "release.text").read_bytes() == (out / "dwarf.text").read_bytes():
        break
else:
    raise RuntimeError("Release and GCC line-table object code differ")
print("GCC debug flags:", flag, flush=True)
source = graph_src.read_text()
marker = "    double times[9];"
if source.count(marker) != 1:
    raise RuntimeError("Inspect the benchmark timing marker")
(out / "graph-profile.c").write_text(
    source.replace(
        marker,
        """    extern void profile_start(void), profile_stop(void);
    size_t calls = 0;
    double t = now_sec();
    profile_start();
    do {
        check_ok(velri_slot_run(slot, nullptr, nullptr), "sampled slot_run");
        ++calls;
    } while (now_sec() - t < 1.5);
    profile_stop();
    fprintf(stderr, "profiled %zu hot calls\\n", calls);
"""
        + marker,
    )
)
compiler = next(
    line.split("=", 1)[1]
    for line in (build / "CMakeCache.txt").read_text().splitlines()
    if line.startswith("CMAKE_C_COMPILER:")
)
flags = [
    "-std=c23",
    "-O3",
    "-no-pie",
    "-DVELRI_BENCH_PACKED_API",
    "-DVELRI_BENCH_FIXED_WARMUP",
    "-I",
    src / "include",
    "-I",
    src / "src/platform",
]
run(
    compiler,
    *flags,
    out / "graph-profile.c",
    sampler,
    lib,
    "-Wl,--wrap=pthread_create",
    "-lm",
    "-ldl",
    "-pthread",
    "-o",
    out / "graph",
)
run(compiler, *flags, graph_src, lib, "-lm", "-ldl", "-pthread", "-o", out / "timing")


outputs = [*inputs, "release.o", "release.text", "dwarf.text", "graph", "timing"]
before = {
    name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in outputs
}
(out / "SHA256SUMS").write_text(
    "".join(f"{digest}  {name}\n" for name, digest in before.items())
)
print(
    "Slang",
    before["dot.snapshot.slang"],
    "machine code",
    before["release.text"],
    flush=True,
)
if args.prepare_only:
    raise SystemExit
cases = [(256, 4096, 4096, 1, 16), (1, 4096, 4096, 12, 48)]
if args.all:
    cases.extend([(128, 2048, 2048, 1, 64), (64, 4096, 4096, 1, 32)])
labels = [(case, layout) for case in cases for layout in ("raw", "packed")]
with (out / "timings.csv").open("w", buffering=1, newline="") as stream:
    table = csv.writer(stream, lineterminator="\n")
    table.writerow(
        [
            "rep",
            "rows",
            "m",
            "k",
            "ring",
            "iterations",
            "layout",
            "tops",
            "ms",
            "effective_W_GB_s",
            "bind_ms",
        ]
    )
    for mode in ["sample", "time"]:
        for rep in range(args.repeat):
            random.Random(2331 + rep).shuffle(labels)
            for (r, m, k, ring, it), layout in labels:
                work = out / f"profile-{rep}-{r}-{layout}"
                env = os.environ.copy()
                exe = out / ("graph" if mode == "sample" else "timing")
                if mode == "sample":
                    env["VELRI_SAMPLE_FILE"] = str(work) + ".ip"
                command = [
                    "taskset",
                    "-c",
                    args.cpus,
                    exe,
                    r,
                    m,
                    k,
                    16,
                    ring,
                    ring if mode == "sample" else it,
                ]
                if layout == "packed":
                    command.append("packed")
                with open("/tmp/velri-w8-perf.lock", "a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    result = run(*command, env=env, capture_output=True)
                if mode == "sample":
                    work.with_suffix(".log").write_text(result.stderr)
                else:
                    rec = result.stdout.strip().split(",")
                    if len(rec) != 16 or rec[:2] != ["w8a8", "auto"]:
                        parser.error(
                            f"Unrecognized timing result: {result.stdout.strip()!r}"
                        )
                    table.writerow(
                        [
                            rep,
                            r,
                            m,
                            k,
                            ring,
                            it,
                            layout,
                            rec[11],
                            rec[8],
                            rec[12],
                            rec[15],
                        ]
                    )
with (out / "live-member.o").open("wb") as stream:
    subprocess.run(["ar", "p", lib, obj.name], stdout=stream, check=True)
if (out / "live-member.o").read_bytes() != (out / "release.o").read_bytes():
    raise RuntimeError("Production archive member changed during profiling")
for name, path in {**inputs, "release.o": obj}.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != before[name]:
        raise RuntimeError(f"Production file changed during profiling: {path}")
run("sha256sum", "-c", "SHA256SUMS", cwd=out)
run(sys.executable, tools / "attribute_dot.py", out)
with tarfile.open(out / "samples-and-symbols.tar.gz", "w:gz") as archive:
    for pattern in (
        "profile-*.ip",
        "profile-*.log",
        "*.nm",
        "*.asm",
        "dot.snapshot.slang",
        "sample-locations.txt",
    ):
        for path in sorted(out.glob(pattern)):
            archive.add(path, arcname=path.name)
