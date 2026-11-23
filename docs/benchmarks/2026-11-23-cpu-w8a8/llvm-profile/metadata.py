"""Record compiler provenance for the two archive paths supplied to paired.py."""

import argparse, csv, subprocess, shlex
from pathlib import Path

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("out", type=Path)
cli.add_argument("gcc_archive", type=Path)
cli.add_argument("llvm_archive", type=Path)
args = cli.parse_args()
dist = args.out
with (dist / "tools.csv").open("w", newline="") as file:
    out = csv.writer(file, lineterminator="\n")
    out.writerow(
        [
            "aot",
            "pipeline",
            "codegen_version",
            "common_graph_compiler",
            "slang_version",
            "dot_codegen_flags",
        ]
    )
    for tag, lib in [("gcc", args.gcc_archive), ("llvm", args.llvm_archive)]:
        lib = lib.resolve()
        build = next((p for p in lib.parents if (p / "CMakeCache.txt").is_file()), None)
        if build is None:
            cli.error(f"No CMake build configuration for {lib}")
        lines = subprocess.check_output(
            ["ninja", "-C", build, "-t", "commands", lib.relative_to(build)], text=True
        ).splitlines()
        stages = [
            shlex.split(stage)
            for ln in lines
            for stage in ln.split(" && ")
            if " -c " in stage and "slang_cpu_dot." in stage
        ]
        if len(stages) != 1:
            cli.error(f"Expected exactly one dot compilation in {build}")
        cc, *flags = stages[0]
        ccver = subprocess.check_output([cc, "--version"], text=True).splitlines()[0]
        cvers = subprocess.check_output(["clang", "--version"], text=True).splitlines()[
            0
        ]
        slang = subprocess.run(
            ["slangc", "-version"], text=True, capture_output=True, check=True
        )
        filter_flags = []
        i = 0
        while i < len(flags):
            if flags[i] in ("-MF", "-c", "-o"):
                i += 2
                continue
            if flags[i] != "-MD":
                filter_flags.append(flags[i])
            i += 1
        out.writerow(
            [
                tag,
                "Slang C++ -> GCC" if tag == "gcc" else "Slang LLVM IR -> Clang",
                ccver,
                cvers,
                (slang.stdout + slang.stderr).strip(),
                shlex.join(filter_flags),
            ]
        )
