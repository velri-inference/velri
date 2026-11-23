#!/usr/bin/env python3
"""Print all three process results and their median for one series."""
import csv
import statistics
import sys
from collections import defaultdict

samples = defaultdict(list)
for arg in sys.argv[1:]:
    with open(arg, newline="") as stream:
        for row in csv.DictReader(stream):
            shape = tuple(map(int, (row["rows"], row["m"], row["k"])))
            gemv = shape[0] == 1
            value = shape[1] * shape[2] / float(row["ms"]) / 1e6 if gemv else float(row["tops"])
            samples[(arg, shape, row["impl"], "GB/s" if gemv else "TOPS")].append(value)
print("series,rows,m,k,impl,unit,median,process1,process2,process3")
for (arg, shape, label, unit), values in sorted(samples.items()):
    if len(values) != 3:
        raise ValueError(f"{arg} {shape} {label} has {len(values)} processes, expected 3")
    print(arg, *shape, label, unit, f"{statistics.median(values):.4f}", *(f"{v:.4f}" for v in values), sep=",")
