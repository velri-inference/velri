"""Record the generated machine loops in each actually timed executable."""

from collections import Counter
import csv, re, subprocess, sys
from pathlib import Path

out = Path(sys.argv[1]).resolve()
with (out / "dot-loops.csv").open("w", newline="") as stream:
    w = csv.writer(stream, lineterminator="\n")
    w.writerow(
        [
            "aot",
            "layout",
            "symbol",
            "first",
            "backedge",
            "byte_dots_512",
            "word_dots_512",
            "byte_dots_256",
            "byte_dots_128",
            "body",
        ]
    )
    for aot in ("gcc", "llvm"):
        elf = out / (aot + ".graph")
        for layout in ("raw", "packed"):
            symbol = "velri_slang_dot_linear_w8a8_f32" + (
                "_packed" if layout == "packed" else ""
            )
            lines = subprocess.check_output(
                [
                    "objdump",
                    "-dw",
                    "--no-show-raw-insn",
                    f"--disassemble={symbol}",
                    elf,
                ],
                text=True,
            ).splitlines()
            first = next(
                int(m[1], 16)
                for row in lines
                if (m := re.fullmatch(r"([0-9a-f]+) <" + symbol + r">:", row))
            )
            ops = {}
            for line in lines:
                m = re.match(r"\s*([0-9a-f]+):\s+(\w+)\s*(.*)", line)
                if m:
                    ops[int(m[1], 16)] = (m[2], m[3])
            for end, (branch, operands) in ops.items():
                match = re.match(r"^([0-9a-f]+) <", operands)
                if not branch.startswith("j") or not match:
                    continue
                target = int(match[1], 16)
                if not first <= target < end:
                    continue
                body = Counter(
                    (
                        op,
                        "zmm"
                        if "%zmm" in args
                        else "ymm"
                        if "%ymm" in args
                        else "xmm"
                        if "%xmm" in args
                        else "",
                    )
                    for ip, (op, args) in ops.items()
                    if target <= ip < end
                )
                if any(op.startswith("j") for op, wide in body) or not any(
                    op.startswith("vpdp") for op, wide in body
                ):
                    continue
                w.writerow(
                    [
                        aot,
                        layout,
                        symbol,
                        hex(target - first),
                        hex(end - first),
                        body["vpdpbusd", "zmm"],
                        body["vpdpwssd", "zmm"],
                        body["vpdpbusd", "ymm"],
                        body["vpdpbusd", "xmm"],
                        " ".join(f"{op}/{wide}={n}" for (op, wide), n in body.items()),
                    ]
                )
