"""System and GPU detection utilities."""

import platform
import subprocess
import sys
from pathlib import Path

import torch


def detect_platform() -> str:
    """Detect whether running on Jetson or a workstation."""
    if Path("/etc/nv_tegra_release").exists():
        return "jetson"
    try:
        import jtop  # noqa: F401
        return "jetson"
    except ImportError:
        pass
    return "workstation"


def get_os_type() -> str:
    """Return 'windows', 'linux', or 'darwin'."""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "darwin"
    return "linux"


def _subprocess_flags() -> dict:
    """Extra kwargs for subprocess calls to suppress console windows on Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def get_device_info() -> dict:
    """Gather GPU and system information."""
    info = {
        "platform": detect_platform(),
        "os_type": get_os_type(),
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_memory_total_mb": props.total_mem // (1024 * 1024),
            "gpu_multiprocessor_count": props.multi_processor_count,
            "cuda_version": torch.version.cuda or "unknown",
        })
        try:
            info["driver_version"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
                **_subprocess_flags(),
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            info["driver_version"] = "unknown"

    return info
