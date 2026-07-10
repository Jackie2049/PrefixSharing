#!/bin/bash
# PrefixSharing performance baseline benchmark runner
# Runs on 4090 server (zxw@219.223.198.62)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$HOME/Termius/proj_prefix-sharing"

# ---- Environment setup ----
source ~/miniconda3/bin/activate verl080_prefix-sharing

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0
export FLASHINFER_DISABLE_VERSION_CHECK=1

echo "[INFO] Conda env: $CONDA_DEFAULT_ENV"
echo "[INFO] CUDA: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "[INFO] Python: $(python3 --version)"
echo "[INFO] PrefixSharing: $(python3 -c 'import prefix_sharing; print(prefix_sharing.__version__ if hasattr(prefix_sharing, "__version__") else "installed")')"
echo "[INFO] Flash-attn: $(python3 -c 'import flash_attn; print(flash_attn.__version__)')"
echo ""

# ---- Verify key modules ----
python3 -c "
import torch
import prefix_sharing
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_detector import TriePrefixDetector
from prefix_sharing.core.prefix_store import PrefixAttentionStore
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.backends.flash_atten_gpu import GpuFlashAttentionBackend
from prefix_sharing.backends.packed_layout import PackedBatchLayout
print('[OK] All modules imported successfully')
print(f'[OK] GPU: {torch.cuda.get_device_name(0)}')
print(f'[OK] GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# ---- Run benchmarks ----
RESULTS_DIR="$BASE_DIR/perf_results"
mkdir -p "$RESULTS_DIR"

echo ""
echo "=== Running GPU FA backend benchmarks ==="
python3 "$SCRIPT_DIR/perf_baseline_benchmark.py" \
    --backend flash_atten_gpu \
    --sync 1 \
    --num-runs 50 \
    --output "$RESULTS_DIR/gpu_fa_baseline.jsonl"

echo ""
echo "=== Running TorchRef backend benchmarks (reference) ==="
python3 "$SCRIPT_DIR/perf_baseline_benchmark.py" \
    --backend torch_ref \
    --sync 1 \
    --num-runs 50 \
    --output "$RESULTS_DIR/torch_ref_baseline.jsonl"

echo ""
echo "=== All benchmarks complete ==="
echo "Results:"
echo "  GPU FA: $RESULTS_DIR/gpu_fa_baseline.jsonl"
echo "  TorchRef: $RESULTS_DIR/torch_ref_baseline.jsonl"
