"""Platform-aware setup script that handles dependency installation."""

import platform
import subprocess
import sys
from pathlib import Path


def detect_platform() -> str:
    if Path("/etc/nv_tegra_release").exists():
        return "jetson"
    try:
        import jtop  # noqa: F401
        return "jetson"
    except ImportError:
        pass
    return "workstation"


def check_torch() -> bool:
    try:
        import torch
        print(f"  PyTorch {torch.__version__} found")
        if torch.cuda.is_available():
            print(f"  CUDA {torch.version.cuda} available — GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  WARNING: CUDA not available. Benchmarks require a CUDA-capable GPU.")
        return True
    except ImportError:
        return False


def install_torch_instructions():
    plat = detect_platform()
    os_type = sys.platform

    print("\n  PyTorch is NOT installed. Install it first:\n")

    if plat == "jetson":
        print("  # Jetson Thor (JetPack 6.x)")
        print("  # Option 1: Use the JetPack pre-installed PyTorch")
        print("  # Option 2: Install from NVIDIA's index:")
        print("  pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61")
    elif os_type == "win32":
        print("  # Windows with CUDA 12.4:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("  # Windows with CUDA 12.1:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("  # Linux with CUDA 12.4:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("  # Linux with CUDA 12.1:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    print()
    print("  Check https://pytorch.org/get-started/locally/ for the latest install command.")
    print("  Then re-run: python setup.py")


def pip_install(requirements_file: str, label: str) -> bool:
    print(f"\n  Installing {label}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            stdout=subprocess.DEVNULL,
        )
        print(f"  {label} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"  WARNING: Failed to install {label}")
        return False


def main():
    print("=" * 55)
    print("  Benchmark Suite Setup")
    print("=" * 55)

    plat = detect_platform()
    os_type = sys.platform
    print(f"\n  Platform: {plat}")
    print(f"  OS: {platform.platform()}")
    print(f"  Python: {sys.version.split()[0]}")

    # Step 1: Check PyTorch
    print("\n--- Step 1: PyTorch ---")
    if not check_torch():
        install_torch_instructions()
        sys.exit(1)

    # Step 2: Core dependencies
    print("\n--- Step 2: Core Dependencies ---")
    pip_install("requirements.txt", "core dependencies")

    # Step 3: Platform-specific
    print("\n--- Step 3: Platform-Specific ---")
    if plat == "jetson":
        pip_install("requirements-jetson.txt", "Jetson power monitoring (jtop)")
    else:
        print("  pynvml already included in core deps (workstation power monitoring)")

    # Step 4: Optional benchmarks
    print("\n--- Step 4: Optional Benchmark Dependencies ---")

    # Vision
    try:
        response = input("  Install vision benchmark deps (ultralytics, open-clip)? [Y/n] ").strip().lower()
    except EOFError:
        response = "y"
    if response != "n":
        pip_install("requirements-vision.txt", "vision benchmarks")

    # LLM
    if plat == "jetson" or os_type != "win32":
        try:
            response = input("  Install LLM benchmark deps (vLLM, transformers)? [Y/n] ").strip().lower()
        except EOFError:
            response = "y"
        if response != "n":
            pip_install("requirements-llm.txt", "LLM benchmarks")
    else:
        print("\n  NOTE: vLLM does not support native Windows.")
        print("  To run LLM benchmarks on this machine, use WSL2:")
        print("    wsl --install")
        print("    # Then run setup.py inside WSL2")
        try:
            response = input("  Try installing anyway? [y/N] ").strip().lower()
        except EOFError:
            response = "n"
        if response == "y":
            pip_install("requirements-llm.txt", "LLM benchmarks")

    # Done
    print("\n" + "=" * 55)
    print("  Setup complete!")
    print("=" * 55)
    print()
    print("  Run benchmarks with:")
    print(f"    python run_all.py configs/{'jetson_thor' if plat == 'jetson' else 'blackwell_workstation'}.yaml")
    print()


if __name__ == "__main__":
    main()
