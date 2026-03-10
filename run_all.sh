#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Usage: ./run_all.sh [config.yaml]"
    echo "  e.g. ./run_all.sh configs/jetson_thor.yaml"
    exit 1
fi

SYSTEM_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['system']['name'])")
RESULTS_DIR=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['output']['results_dir'])")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${RESULTS_DIR}/${SYSTEM_NAME}_${TIMESTAMP}"

mkdir -p "$RUN_DIR"

echo "============================================="
echo "  Benchmark Suite: $SYSTEM_NAME"
echo "  Config: $CONFIG"
echo "  Results: $RUN_DIR"
echo "  Started: $(date)"
echo "============================================="

# Save config for reproducibility
cp "$CONFIG" "$RUN_DIR/config.yaml"

# Save system info
python3 -c "
from common.device_info import get_device_info
from common.utils import save_json
save_json(get_device_info(), '$RUN_DIR/system_info.json')
" 2>/dev/null || echo "WARN: Could not save system info"

echo ""
echo "[1/5] GEMM Compute Benchmark"
echo "---------------------------------------------"
python3 -m hpc_gemm.run --config "$CONFIG" --output "$RUN_DIR/gemm.json" || echo "WARN: GEMM benchmark failed"

echo ""
echo "[2/5] Memory Bandwidth Benchmark"
echo "---------------------------------------------"
python3 -m bandwidth_test.run --config "$CONFIG" --output "$RUN_DIR/bandwidth.json" || echo "WARN: Bandwidth benchmark failed"

echo ""
echo "[3/5] Vision Model Benchmark"
echo "---------------------------------------------"
python3 -m vision_benchmark.run --config "$CONFIG" --output "$RUN_DIR/vision.json" || echo "WARN: Vision benchmark failed/skipped"

echo ""
echo "[4/5] LLM Inference Benchmark"
echo "---------------------------------------------"
python3 -m llm_benchmark.run --config "$CONFIG" --output "$RUN_DIR/llm.json" || echo "WARN: LLM benchmark failed/skipped"

echo ""
echo "[5/5] End-to-End Pipeline Benchmark"
echo "---------------------------------------------"
python3 -m pipeline_benchmark.run --config "$CONFIG" --output "$RUN_DIR/pipeline.json" || echo "WARN: Pipeline benchmark failed/skipped"

echo ""
echo "============================================="
echo "  All benchmarks complete!"
echo "  Results saved to: $RUN_DIR"
echo "  Finished: $(date)"
echo "============================================="
echo ""
echo "To compare results between systems, run:"
echo "  python3 compare.py --system-a results/<system_a_dir> --system-b results/<system_b_dir> --plot"
