# Jetson Thor vs Blackwell Workstation Benchmark Suite

A reproducible benchmarking suite comparing NVIDIA Jetson Thor edge systems (~$2,000) against Blackwell workstation GPUs (~$10,000) across AI inference and GPU compute workloads.

## Hardware Assumptions

| System | Jetson Thor | Blackwell Workstation |
|---|---|---|
| Cost | ~$2,000 | ~$10,000 |
| GPU | Jetson Thor (Blackwell arch) | RTX 5090 / RTX 6000 Ada |
| TDP | ~130W (system) | ~575W (GPU only) |
| Memory | 128GB unified (shared CPU/GPU) | 32GB GDDR7 (dedicated) |
| Interconnect | Shared memory | PCIe 5.0 x16 |
| OS | JetPack 6.x (Ubuntu-based) | Windows 10/11 or Linux |

## Benchmarks

| Benchmark | What it measures | Key metrics |
|---|---|---|
| **LLM Inference** | Llama-3.1-8B, Mistral-7B, Qwen-7B via vLLM | tokens/sec, latency/token, GPU memory |
| **Vision** | YOLOv8, CLIP, Segment Anything (optional) | frames/sec, latency/frame |
| **GEMM Compute** | Matrix multiply via PyTorch/cuBLAS | FP16/FP32 TFLOPS |
| **Bandwidth** | Device-to-device, host-to-device, device-to-host | GB/s |
| **E2E Pipeline** | image → CLIP → LLM → classification | e2e latency, inferences/sec |

All results include **power monitoring** (pynvml/nvidia-smi on workstation, tegrastats/jtop on Jetson) and are normalized to compute **performance per dollar** and **performance per watt**.

## Setup

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.0+
- PyTorch 2.1+ (with CUDA support) — **must be installed separately before other deps**

### Quick Setup (Interactive)

The setup script detects your platform and installs the right dependencies:

```bash
python setup.py
```

It will check for PyTorch, install core deps, and prompt for optional benchmark deps (vision, LLM).

### Manual Install (Windows Workstation via WSL2 — Recommended)

```bash
# Inside WSL2 (Ubuntu)
python -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA (check https://pytorch.org for latest command)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core + all benchmarks
pip install -r requirements.txt
pip install -r requirements-vision.txt
pip install -r requirements-llm.txt
```

### Manual Install (Windows Native)

```powershell
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core + vision (GEMM, bandwidth, vision work natively)
pip install -r requirements.txt
pip install -r requirements-vision.txt

# vLLM does NOT support native Windows — use WSL2 for LLM benchmarks
```

### Manual Install (Jetson Thor)

```bash
# PyTorch comes pre-installed with JetPack 6.x, or install from NVIDIA's index:
# pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61

# Core + Jetson power monitoring
pip install -r requirements.txt
pip install -r requirements-jetson.txt

# Vision + LLM
pip install -r requirements-vision.txt
pip install -r requirements-llm.txt
```

### Dependency Notes

| Dependency | Windows Native | WSL2 | Jetson Thor |
|---|---|---|---|
| PyTorch + CUDA | pip with `cu124` index | pip with `cu124` index | JetPack pre-built or NVIDIA aarch64 wheel |
| pynvml | Works | Works | May not work (use jtop instead) |
| vLLM | Not supported | Works | May need source build |
| ultralytics | Works | Works | Works |
| open-clip | Works | Works | Works |
| numpy | Pin `<2.0` (compat) | Pin `<2.0` (compat) | Pin `<2.0` (compat) |

The suite gracefully skips benchmarks whose dependencies aren't available, so partial runs always work.

## Usage

### Run All Benchmarks

```bash
# Cross-platform (works on Windows, Linux, Jetson)
python run_all.py configs/blackwell_workstation.yaml

# Linux/Jetson only (bash)
./run_all.sh configs/jetson_thor.yaml
```

You can also run a subset of benchmarks:

```bash
python run_all.py configs/blackwell_workstation.yaml --benchmarks gemm bandwidth vision
```

Results are saved to `results/<system_name>_<timestamp>/` with one JSON file per benchmark.

### Run Individual Benchmarks

```bash
# GEMM compute
python -m hpc_gemm.run --config configs/blackwell_workstation.yaml

# Memory bandwidth
python -m bandwidth_test.run --config configs/jetson_thor.yaml

# Vision models
python -m vision_benchmark.run --config configs/blackwell_workstation.yaml

# LLM inference (requires vLLM)
python -m llm_benchmark.run --config configs/jetson_thor.yaml

# End-to-end pipeline (requires both CLIP and vLLM)
python -m pipeline_benchmark.run --config configs/blackwell_workstation.yaml
```

### CLI Overrides

Override any config value via command line:

```bash
python -m hpc_gemm.run --config config.yaml --gemm.matrix_sizes 1024,2048 --benchmarks.benchmark_iterations 10
```

### Compare Results

After running benchmarks on both systems, copy the result directories to one machine and run:

```bash
python compare.py --system-a results/jetson_thor_20260310_120000/ --system-b results/blackwell_rtx5090_20260310_130000/ --plot
```

This produces:
- `results/comparison.json` with normalized metrics
- `plots/` directory with comparison charts

### Generate Plots Only

```bash
python -m plots.generate_plots --results-a results/jetson_thor_*/ --results-b results/blackwell_rtx5090_*/ --comparison results/comparison.json --output plots/
```

## Cross-Platform Workflow

The recommended workflow for comparing the two systems:

1. **On the Windows workstation**: Run benchmarks with `python run_all.py configs/blackwell_workstation.yaml`
2. **On the Jetson Thor**: Run benchmarks with `python run_all.py configs/jetson_thor.yaml`
3. **Copy results**: Transfer the `results/` directories to one machine (e.g. via `scp`, USB, or shared drive)
4. **Compare**: Run `python compare.py --system-a results/jetson_* --system-b results/blackwell_* --plot`

Each benchmark that cannot run on a given platform (e.g. vLLM on Windows) is automatically skipped with a warning. The comparison script handles partial results gracefully.

## Configuration

Edit `configs/jetson_thor.yaml` or `configs/blackwell_workstation.yaml` to customize:

- **System**: name, cost, TDP (for normalization)
- **LLM**: backend (vllm/tensorrt), models, batch sizes, quantization
- **Vision**: models, resolution, batch sizes
- **GEMM**: dtypes, matrix sizes
- **Bandwidth**: transfer sizes, iterations
- **Power**: sampling interval, settling time

## Output Format

Each benchmark writes a JSON file with this structure:

```json
{
  "system": "jetson_thor",
  "benchmark_name": "gemm_compute",
  "results": [ ... ],
  "power": { "avg_watts": 85.2, "peak_watts": 125.0 },
  "device_info": { "gpu_name": "...", "cuda_version": "...", "os_type": "windows" },
  "timestamp": "2026-03-10T12:00:00+00:00",
  "config": { ... }
}
```

## Project Structure

```
├── common/              # Shared utilities, base classes, config
├── power_monitor/       # Power sampling (pynvml, tegrastats)
├── llm_benchmark/       # LLM inference (vLLM, TensorRT-LLM stub)
├── vision_benchmark/    # Vision models (YOLOv8, CLIP, SAM)
├── hpc_gemm/            # GEMM compute benchmark
├── bandwidth_test/      # Memory bandwidth benchmark
├── pipeline_benchmark/  # End-to-end AI pipeline
├── plots/               # Plot generation
├── results/             # Benchmark output (JSON)
├── configs/             # System-specific YAML configs
├── compare.py           # Cross-system comparison
├── run_all.py           # Cross-platform orchestration (Windows + Linux)
├── run_all.sh           # Bash orchestration (Linux/Jetson only)
├── setup.py             # Interactive platform-aware dependency installer
└── requirements*.txt    # Layered dependencies
```

## License

MIT
