"""Sample the exact benchmark PID on one NVIDIA GPU and obtain its Linux RSS high-water mark."""
import csv
import ctypes as C
import gzip
import os
from contextlib import ExitStack
from pathlib import Path
import subprocess
import time


class ProcessInfo(C.Structure):
    _fields_ = [
        ("pid", C.c_uint),
        ("used_gpu_memory", C.c_ulonglong),
        ("gpu_instance_id", C.c_uint),
        ("compute_instance_id", C.c_uint),
    ]


class NvidiaProcesses:
    def __init__(self, index=0):
        self.lib = C.CDLL("libnvidia-ml.so.1")
        self.lib.nvmlErrorString.argtypes = [C.c_int]
        self.lib.nvmlErrorString.restype = C.c_char_p
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [C.c_uint, C.POINTER(C.c_void_p)]
        self.handle = C.c_void_p()
        self.buffer = (ProcessInfo * 32)()
        self.queries = [
            ("C", self.lib.nvmlDeviceGetComputeRunningProcesses_v3),
            ("G", self.lib.nvmlDeviceGetGraphicsRunningProcesses_v3),
        ]
        for _, query in self.queries:
            query.argtypes = [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(ProcessInfo)]
        self.check(self.lib.nvmlInit_v2())
        try:
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(index, C.byref(self.handle)))
        except BaseException:
            self.lib.nvmlShutdown()
            raise

    def check(self, rc):
        if rc:
            raise RuntimeError(self.lib.nvmlErrorString(rc).decode(errors="replace"))

    def sample(self, pid):
        amounts, classes = [], []
        for label, query in self.queries:
            while True:
                count = C.c_uint(len(self.buffer))
                rc = query(self.handle, C.byref(count), self.buffer)
                if rc == 7:  # NVML_ERROR_INSUFFICIENT_SIZE
                    self.buffer = (ProcessInfo * max(2 * len(self.buffer), count.value))()
                    continue
                self.check(rc)
                break
            for i in range(count.value):
                info = self.buffer[i]
                if info.pid == pid:
                    if info.used_gpu_memory == 2**64 - 1:
                        raise RuntimeError("GPU memory unavailable for sampled PID")
                    amounts.append(info.used_gpu_memory)
                    classes.append(label)
        # The two APIs both return the total for the process, even if it is in both classes.
        return max(amounts, default=0), "".join(classes)

    def close(self):
        self.lib.nvmlShutdown()


def run_monitored(command, env, stdout, stderr, trace=None, mode="gpu", interval_ms=5.0, device_index=0):
    """Run a benchmark directly (taskset is fine), monitoring from Popen through wait4.

    stdout/stderr must be open files. Optional trace is a path (.gz enables gzip).
    Return a JSON-serializable dictionary including returncode and rss_bytes/vram_bytes.
    vram_bytes is None with an explanatory nvml_status when the value could not be observed.
    """
    if interval_ms <= 0 or mode not in ("cpu", "gpu"):
        raise ValueError("mode must be cpu or gpu and interval_ms must be positive")
    errors = []
    observed_classes = set()
    calls = observed = peak_vram = 0
    peak_poll_gap = 0.0
    with ExitStack() as stack:
        try:
            nv = NvidiaProcesses(device_index)
            stack.callback(nv.close)
        except (OSError, RuntimeError, AttributeError) as exc:
            nv = None
            errors.append(str(exc))
        writer = None
        if trace is not None:
            trace = Path(trace)
            stream = stack.enter_context(gzip.open(trace, "wt", compresslevel=1) if trace.suffix == ".gz" else trace.open("w"))
            writer = csv.writer(stream)
            writer.writerow(("elapsed_s", "pid", "nvml_classes", "vram_bytes"))
        start = last = time.monotonic()
        child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        try:
            while True:
                pid, status, usage = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    break
                now = time.monotonic()
                peak_poll_gap = max(peak_poll_gap, now - last)
                last = now
                if nv is not None:
                    try:
                        amount, classes = nv.sample(child.pid)
                    except (OSError, RuntimeError) as exc:
                        errors.append(str(exc))
                        nv = None
                    else:
                        calls += 1
                        if classes:
                            observed += 1
                            observed_classes.update(classes)
                            peak_vram = max(peak_vram, amount)
                        if writer:
                            writer.writerow((round(now - start, 6), child.pid, classes, amount))
                time.sleep(interval_ms * .001)
        finally:
            if child.returncode is None:
                child.kill()
                child.wait()
        elapsed = time.monotonic() - start

    value = None
    if errors:
        nvml_status = "NVML unavailable: " + "; ".join(errors)
    elif observed and mode == "gpu" and peak_vram == 0:
        nvml_status = "GPU PID appeared but no positive allocation was observed"
    elif observed:
        value = peak_vram
        nvml_status = "PID observed in " + "".join(sorted(observed_classes))
    elif mode == "cpu" and calls:
        value = 0
        nvml_status = "no GPU context for CPU PID in either NVML process list"
    else:
        nvml_status = "PID did not appear in NVML process lists"
    return {
        "pid":child.pid,
        "returncode":child.returncode,
        "rss_bytes":usage.ru_maxrss * 1024,
        "vram_bytes":value,
        "nvml_status":nvml_status,
        "nvml_queries":calls,
        "nvml_pid_samples":observed,
        "nvml_classes":"".join(sorted(observed_classes)),
        "interval_ms":interval_ms,
        "max_observed_poll_gap_ms":round(peak_poll_gap * 1000, 3),
        "elapsed_seconds":round(elapsed, 4),
    }
