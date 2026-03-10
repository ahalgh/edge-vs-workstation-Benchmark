"""PowerMonitor context manager with background sampling via multiprocessing."""

import multiprocessing
import time

from common.device_info import detect_platform
from common.result_schema import PowerResult


def _sample_loop(backend_cls, config, queue, stop_event):
    """Sampling loop that runs in a child process."""
    backend = backend_cls(config)
    interval = config.get("power", {}).get("sampling_interval_ms", 100) / 1000.0

    while not stop_event.is_set():
        try:
            watts = backend.read_power()
            if watts is not None and watts > 0:
                queue.put(watts)
        except Exception:
            pass
        time.sleep(interval)


class PowerMonitor:
    """Context manager that samples GPU power in a background process.

    Usage:
        with PowerMonitor(config) as pm:
            # run benchmark
            ...
        result = pm.get_results()
    """

    def __init__(self, config: dict):
        self.config = config
        self._platform = detect_platform()
        self._settling_time = config.get("power", {}).get("settling_time_s", 2)
        self._interval_ms = config.get("power", {}).get("sampling_interval_ms", 100)
        self._process = None
        self._queue = None
        self._stop_event = None
        self._start_time = 0.0
        self._samples = []

    def _get_backend_cls(self):
        if self._platform == "jetson":
            from power_monitor.backends.tegrastats import TegrastatsBackend
            return TegrastatsBackend
        else:
            from power_monitor.backends.nvidia_smi import NvidiaSmiBackend
            return NvidiaSmiBackend

    def __enter__(self):
        self._queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()
        backend_cls = self._get_backend_cls()

        self._process = multiprocessing.Process(
            target=_sample_loop,
            args=(backend_cls, self.config, self._queue, self._stop_event),
            daemon=True,
        )
        self._process.start()
        time.sleep(self._settling_time)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        duration = time.perf_counter() - self._start_time
        self._stop_event.set()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()

        # Drain the queue
        self._samples = []
        while not self._queue.empty():
            try:
                self._samples.append(self._queue.get_nowait())
            except Exception:
                break

        self._duration = duration

    def get_results(self) -> PowerResult:
        if not self._samples:
            return PowerResult(
                avg_watts=0.0,
                peak_watts=0.0,
                samples=[],
                duration_seconds=getattr(self, "_duration", 0.0),
                sampling_interval_ms=self._interval_ms,
            )
        return PowerResult(
            avg_watts=sum(self._samples) / len(self._samples),
            peak_watts=max(self._samples),
            samples=self._samples,
            duration_seconds=self._duration,
            sampling_interval_ms=self._interval_ms,
        )
