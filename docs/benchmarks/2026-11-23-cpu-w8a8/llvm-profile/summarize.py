"""Compute both medians and paired changes from timed graph processes."""

import argparse, csv
from collections import defaultdict
from statistics import median
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
args = parser.parse_args()
p = args.out
groups = defaultdict(lambda: defaultdict(dict))
with (p / "process.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        key = tuple(int(row[k]) for k in ("rows", "m", "k", "ring")) + (row["layout"],)
        group = groups[key][row["aot"]]
        rep = int(row["rep"])
        if rep in group:
            parser.error("duplicate process for shape/compiler/layout/repetition")
        group[rep] = row
with (
    (p / "medians.csv").open("w", newline="") as ms,
    (p / "paired.csv").open("w", newline="") as pairs,
):
    sums = csv.writer(ms, lineterminator="\n")
    out = csv.writer(pairs, lineterminator="\n")
    metrics = ["tops", "ms", "effective_W_GB_s", "bind_ms"]
    sums.writerow(
        [
            "rows",
            "m",
            "k",
            "ring",
            "layout",
            "aot",
            "processes",
            *[
                name + "_" + stat
                for name in metrics
                for stat in ("median", "min", "max")
            ],
        ]
    )
    out.writerow(
        [
            "rows",
            "m",
            "k",
            "ring",
            "layout",
            "unit",
            "gcc_median",
            "llvm_median",
            "ratio_of_medians",
            "paired_ratio_median",
            "paired_ratio_min",
            "paired_ratio_max",
            "faster_pairs",
        ]
    )
    for key, variants in sorted(
        groups.items(), key=lambda h: (h[0][0], h[0][1], h[0][4])
    ):
        if any(set(variants[tag]) != {0, 1, 2} for tag in ("gcc", "llvm")):
            parser.error(
                "expected three independent processes per implementation and shape"
            )
        for aot, runs in sorted(variants.items()):
            data = []
            for field in metrics:
                vals = [float(row[field]) for row in runs.values()]
                data.extend(f"{fn(vals):.9g}" for fn in (median, min, max))
            sums.writerow([*key, aot, len(runs), *data])
        field = "effective_W_GB_s" if key[0] == 1 else "tops"
        base, fast = (
            [float(variants[tag][i][field]) for i in range(3)]
            for tag in ("gcc", "llvm")
        )
        ratios = [b / a for a, b in zip(base, fast)]
        out.writerow(
            [
                *key,
                field,
                f"{median(base):.9g}",
                f"{median(fast):.9g}",
                f"{median(fast) / median(base):.6f}",
                *(f"{fn(ratios):.6f}" for fn in (median, min, max)),
                sum(v > 1 for v in ratios),
            ]
        )
