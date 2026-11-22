"""Write per-process measurements and medians for the long-context benchmark."""
import argparse
import csv
import json
import math
from pathlib import Path
from signal import Signals
from statistics import median
from tempfile import NamedTemporaryFile

ENGINES = (
    "velri-cpu-f32", "llama-cpu-f32",
    "velri-vulkan-f16kv", "llama-cuda-f16kv", "llama-vulkan-f16kv",
    "velri-vulkan-f32", "llama-cuda-f32", "llama-vulkan-f32",
)
LENGTHS = (16384, 32768, 65536)
BLOCKS = (1, 2, 3)
METRICS = ("prefill_tps", "decode_tps", "rss_mib", "vram_mib")


def measurements(path, engine, prompt, block):
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    if (record["engine"], record["prompt"], record["block"], record["returncode"]) != (engine, prompt, block, 0):
        raise ValueError(f"{path}: invalid run identity or exit status")
    values = {
        "prefill_tps": record["prefill_tps"], "decode_tps": record["decode_tps"],
        "rss_mib": record["rss_bytes"] / 2**20, "vram_mib": record["vram_bytes"] / 2**20,
    }
    if not all(math.isfinite(x) and x > 0 for x in (values["prefill_tps"], values["decode_tps"], values["rss_mib"])):
        raise ValueError(f"{path}: invalid speed or RSS")
    if not math.isfinite(values["vram_mib"]) or values["vram_mib"] < 0:
        raise ValueError(f"{path}: invalid VRAM")
    return values


def write_csv(path, fields, rows):
    with NamedTemporaryFile(mode="w", dir=path.parent, newline="", delete=False) as stream:
        temp = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="output directory containing records/")
    args = parser.parse_args()
    records = args.root / "records"
    if not records.is_dir():
        parser.error(f"records directory does not exist: {records}")
    samples, summary = [], []
    for engine in ENGINES:
        for prompt in LENGTHS:
            key = {"engine": engine, "prompt": prompt}
            successful = []
            for block in BLOCKS:
                path = records / f"{engine}-{prompt}-{block}.json"
                values = measurements(path, engine, prompt, block)
                sample = {**key, "block": block, "status": "missing"}
                if values is not None:
                    successful.append(values)
                    sample.update(status="ok", **values)
                elif (failure_path := path.with_suffix(".failure.json")).is_file():
                    failure = json.loads(failure_path.read_text())
                    status = failure["returncode"]
                    if status == 0 or (failure["engine"], failure["prompt"], failure["block"]) != (engine, prompt, block):
                        raise ValueError(f"{failure_path}: invalid failure identity or exit status")
                    sample["status"] = Signals(-status).name if status < 0 else f"exit-{status}"
                samples.append(sample)
            row = {**key, "ok": len(successful)}
            if len(successful) == len(BLOCKS):
                row.update({k: median(run[k] for run in successful) for k in METRICS})
            summary.append(row)
    write_csv(args.root / "samples.csv", ("engine", "prompt", "block", "status", *METRICS), samples)
    write_csv(args.root / "summary.csv", ("engine", "prompt", "ok", *METRICS), summary)


if __name__ == "__main__":
    main()
