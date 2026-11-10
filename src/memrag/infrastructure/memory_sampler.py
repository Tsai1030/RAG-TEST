"""System memory sampling for measurements in a child process.

Memory on the Jetson is shared by CPU and GPU, so besides PyTorch's numbers we sample
system MemAvailable in a background thread and report the largest drop.
"""

from __future__ import annotations

import threading
import time


class MemSampler(threading.Thread):
    def __init__(self, interval_s: float = 0.02):
        super().__init__(daemon=True)
        import psutil

        self._psutil = psutil
        self._interval = interval_s
        self._done = threading.Event()  # not `_stop`: that name is a Thread method
        self.baseline = psutil.virtual_memory().available
        self.minimum = self.baseline

    def run(self):
        while not self._done.is_set():
            self.minimum = min(self.minimum, self._psutil.virtual_memory().available)
            time.sleep(self._interval)

    def used_now(self) -> int:
        """Bytes of MemAvailable gone since the baseline, right now."""
        return self.baseline - self._psutil.virtual_memory().available

    def stop(self) -> int:
        """Stop sampling; the largest drop of MemAvailable in bytes."""
        self._done.set()
        self.join()
        return self.baseline - self.minimum


def mib(n: float) -> float:
    return round(n / 2**20, 1)


def cuda_peak(reset: bool = False) -> int | None:
    """PyTorch's peak CUDA allocation in bytes since the last reset; None when CUDA is not in use."""
    try:
        import torch
    except ImportError:
        return None
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return None
    peak = torch.cuda.max_memory_allocated()
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return peak


def peak_rss() -> int:
    """Peak resident set size of this process in bytes (peak working set on Windows)."""
    try:
        import resource
    except ImportError:  # Windows
        import psutil

        return psutil.Process().memory_info().peak_wset
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
