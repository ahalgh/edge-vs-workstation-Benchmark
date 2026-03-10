"""Jetson power monitoring via jtop, sysfs, or tegrastats fallback."""

import subprocess
from pathlib import Path


class TegrastatsBackend:
    """Read total system power on Jetson platforms."""

    def __init__(self, config: dict):
        self._method = self._detect_method()

    def _detect_method(self) -> str:
        try:
            from jtop import jtop  # noqa: F401
            return "jtop"
        except ImportError:
            pass

        # Check for INA3221 sysfs power rails
        rails = list(Path("/sys/bus/i2c/drivers/ina3221x/").glob("*/iio:device*/in_power*_input"))
        if rails:
            return "sysfs"

        return "tegrastats"

    def read_power(self) -> float | None:
        """Read current total power in watts."""
        if self._method == "jtop":
            return self._read_jtop()
        elif self._method == "sysfs":
            return self._read_sysfs()
        else:
            return self._read_tegrastats()

    def _read_jtop(self) -> float | None:
        try:
            from jtop import jtop
            with jtop() as jetson:
                power = jetson.power
                # Total power in milliwatts from all rails
                total_mw = sum(v.get("cur", 0) for v in power[1].values())
                return total_mw / 1000.0
        except Exception:
            return None

    def _read_sysfs(self) -> float | None:
        try:
            total_mw = 0
            rails = Path("/sys/bus/i2c/drivers/ina3221x/").glob("*/iio:device*/in_power*_input")
            for rail in rails:
                total_mw += int(rail.read_text().strip())
            return total_mw / 1000.0
        except Exception:
            return None

    def _read_tegrastats(self) -> float | None:
        try:
            output = subprocess.check_output(
                ["tegrastats", "--interval", "100", "--stop", "1"],
                text=True,
                timeout=2,
            ).strip()
            # Parse VDD_* power values from tegrastats output
            total_mw = 0
            for token in output.split():
                if token.startswith("VDD_") and "/" in token:
                    # Format: VDD_CPU_GPU_CV/1234 (milliwatts)
                    parts = token.split("/")
                    if len(parts) >= 2:
                        try:
                            total_mw += int(parts[1])
                        except ValueError:
                            pass
            return total_mw / 1000.0 if total_mw > 0 else None
        except Exception:
            return None
