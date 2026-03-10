"""CUDA-aware timing utilities using torch.cuda.Event for accurate GPU timing."""

import time

import torch


class CUDATimer:
    """Context manager for accurate GPU timing using CUDA events.

    Falls back to wall-clock time if CUDA is not available.
    """

    def __init__(self):
        self.elapsed_ms = 0.0
        self._use_cuda = torch.cuda.is_available()

    def __enter__(self):
        if self._use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._wall_start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self._start.elapsed_time(self._end)
        else:
            self.elapsed_ms = (time.perf_counter() - self._wall_start) * 1000.0
