"""Compare two Release bench_session executables on the same quantized checkpoint (Linux)."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import statistics
import subprocess


def run(binary, weights, output, tag, device, cpu_list, workers, record_logits):
    command = ["taskset", "-c", cpu_list, str(binary)]
    if device == "vulkan":
        command += ["vulkan", "1"]
    else:
        command += ["cpu", str(workers), "--isa", device]
    command += ["--weights", str(weights), "--warmups", "2", "--runs", "1"]
    if record_logits:
        command += ["--logits", str(output / f"{tag}.f32")]
    result = subprocess.run(command, capture_output=True, text=True)
    (output / f"{tag}.csv").write_text(result.stdout)
    (output / f"{tag}.log").write_text(result.stderr)
    result.check_returncode()
    rows = [row for row in csv.DictReader(result.stdout.splitlines()) if row["run"] == "1"]
    trajectory = re.search(r"trajectory_fnv1a64=([0-9a-fA-F]+)", result.stderr)
    if len(rows) != 1 or int(rows[0]["tokens"]) != 64 or trajectory is None:
        raise ValueError(f"invalid 64-token run or missing trajectory in {tag}")
    return float(rows[0]["total_ms"]), trajectory[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path, help="new directory for raw timings, logits and summary")
    parser.add_argument("--cpus", default="0-15", help="taskset CPU list with one CPU per physical core")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--devices", nargs="+", choices=["avx2", "avx512", "vulkan"],
                        default=["avx2", "avx512", "vulkan"])
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    binaries = [args.before.resolve(), args.after.resolve()]
    with (args.output / "inputs.txt").open("w") as meta:
        for path in [*binaries, args.weights.resolve()]:
            with path.open("rb") as file:
                print(hashlib.file_digest(file, "sha256").hexdigest(), path, file=meta)
        print("VK_DRIVER_FILES", os.environ.get("VK_DRIVER_FILES", "(system default)"), file=meta)
        print("cpus", args.cpus, "workers", args.workers, file=meta)

    with (args.output / "summary.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["device", "version", "run1_ms", "run2_ms", "run3_ms", "mean_ms",
                         "tokens_per_second", "trajectory_fnv1a64", "logits_sha256"])
        for device in args.devices:
            timings = [[], []]
            expected_trajectory = None
            for trial, order in enumerate(((0, 1), (1, 0), (0, 1)), start=1):
                for version in order:
                    label = ("before", "after")[version]
                    tag = f"{device}-{label}-{trial}"
                    ms, trajectory = run(binaries[version], args.weights, args.output, tag, device,
                                         args.cpus, args.workers, trial == 1)
                    if expected_trajectory is None:
                        expected_trajectory = trajectory
                    if trajectory != expected_trajectory:
                        raise ValueError(f"greedy trajectory changed in {tag}")
                    timings[version].append(ms)
                    print(f"{tag} {ms:.3f} ms", flush=True)
            digests = []
            for version, times in enumerate(timings):
                label = ("before", "after")[version]
                with (args.output / f"{device}-{label}-1.f32").open("rb") as data:
                    digest = hashlib.file_digest(data, "sha256").hexdigest()
                digests.append(digest)
                mean = statistics.mean(times)
                writer.writerow([device, label, *times, mean, 64000 / mean, expected_trajectory, digest])
                file.flush()
            if digests[0] != digests[1]:
                raise ValueError(f"{device} logits changed; inspect both saved FP32 arrays")


if __name__ == "__main__":
    main()
