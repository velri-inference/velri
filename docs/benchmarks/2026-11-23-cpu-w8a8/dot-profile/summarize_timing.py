"""Summarize independent timing processes and compare to a measured arithmetic ceiling."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("timings", type=Path)
parser.add_argument("register_ceiling", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

if args.output.resolve() in (args.timings.resolve(), args.register_ceiling.resolve()):
    parser.error("Output CSV must differ from both inputs.")


def read_table(path, required):
    if not path.is_file():
        parser.error(f"Missing input CSV: {path}")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            parser.error(f"{path} is missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        parser.error(f"Empty CSV: {path}")
    return rows


fields = ["tops", "ms", "effective_W_GB_s", "bind_ms"]
ceiling_rows = read_table(args.register_ceiling, {"vnni_tops"})
timing_rows = read_table(args.timings, {"rows", "layout", "m", "k", "ring", *fields})
try:
    reference = median(float(row["vnni_tops"]) for row in ceiling_rows)
    if not math.isfinite(reference) or reference <= 0:
        parser.error("Expected a positive, finite register ceiling.")
except (TypeError, ValueError) as error:
    parser.error(f"Invalid ceiling in {args.register_ceiling}: {error}")
groups = defaultdict(list)
for lineno, row in enumerate(timing_rows, 2):
    try:
        key = (
            int(row["rows"]),
            row["layout"],
            int(row["m"]),
            int(row["k"]),
            int(row["ring"]),
        )
        groups[key].append({metric: float(row[metric]) for metric in fields})
    except (TypeError, ValueError) as error:
        parser.error(f"Invalid value on line {lineno} of {args.timings}: {error}")

with args.output.open("w", newline="") as dst:
    csv_out = csv.writer(dst, lineterminator="\n")
    header = [
        "rows",
        "m",
        "k",
        "layout",
        "ring",
        "runs",
        "register_ceiling_TOPS",
        "pct_register_TOPS",
    ]
    header.extend(
        f"{metric}_{p}" for metric in fields for p in ("median", "min", "max")
    )
    csv_out.writerow(header)
    for (r, layout, m, k, ring), runs in sorted(groups.items()):
        op_tops = median(row["tops"] for row in runs)
        data = [
            r,
            m,
            k,
            layout,
            ring,
            len(runs),
            reference,
            f"{100 * op_tops / reference:.2f}" if r > 1 else "",
        ]
        for metric in fields:
            values = [run[metric] for run in runs]
            data.extend(f"{aggregate(values):.9g}" for aggregate in (median, min, max))
        csv_out.writerow(data)
