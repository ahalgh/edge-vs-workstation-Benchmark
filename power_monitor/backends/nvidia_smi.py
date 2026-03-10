"""Workstation power monitoring via pynvml (preferred) or nvidia-smi fallback."""

import subprocess
import sys


class NvidiaSmiBackend:
    """Read GPU power on workstation systems."""

    def __init__(self, config: dict):
        self._handle = None
        self._use_pynvml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
            self._use_pynvml = True
        except Exception:
            pass

    def read_power(self) -> float | None:
        """Read current GPU power draw in watts."""
        if self._use_pynvml:
            try:
                milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                return milliwatts / 1000.0
            except Exception:
                return None

        # Fallback: nvidia-smi subprocess
        try:
            extra = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,nounits,noheader"],
                text=True,
                timeout=2,
                **extra,
            ).strip()
            return float(output.split("\n")[0])
        except Exception:
            return None
