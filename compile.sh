#!/bin/bash
# Compile Qwen3-0.6B with MLIR GPU-dialect GEMV kernels for Jetson AGX Orin (sm_87)
#
# Usage:
#   ./compile.sh                    # default: IREE codegen + MLIR GPU kernels
#   ./compile.sh --no-kernels       # IREE codegen only (no custom kernels)
#
# Prerequisites:
#   - iree-compile on PATH or at ~/iree/build/tools/iree-compile
#   - Weight files (see README.md for setup)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IREE_COMPILE="${IREE_COMPILE:-$(command -v iree-compile 2>/dev/null || echo ~/iree/build/tools/iree-compile)}"
MLIR="$SCRIPT_DIR/model/qwen3_full.mlir"
KERNELS="$SCRIPT_DIR/kernels"
OUTPUT="${OUTPUT:-qwen3.vmfb}"

CUDA_FLAGS="--iree-hal-target-device=cuda \
  --iree-cuda-target=sm_87 \
  --iree-cuda-target-features=+ptx74 \
  --iree-hal-indirect-command-buffers=false"

if [ "$1" = "--no-kernels" ]; then
  echo "=== Compiling (IREE codegen only, no custom kernels) ==="
  $IREE_COMPILE "$MLIR" $CUDA_FLAGS -o "$OUTPUT"
else
  echo "=== Compiling with MLIR GPU-dialect GEMV kernels ==="
  $IREE_COMPILE "$MLIR" $CUDA_FLAGS \
    --iree-hal-link-executables=false \
    "--iree-hal-substitute-executable-source=decode_dispatch_3=$KERNELS/dispatch_3_gpu.mlir" \
    "--iree-hal-substitute-executable-source=decode_dispatch_4=$KERNELS/dispatch_4_gpu.mlir" \
    "--iree-hal-substitute-executable-source=decode_dispatch_12=$KERNELS/dispatch_12_gpu.mlir" \
    "--iree-hal-substitute-executable-source=decode_dispatch_18=$KERNELS/dispatch_18_gpu.mlir" \
    "--iree-hal-substitute-executable-source=decode_dispatch_20=$KERNELS/dispatch_20_gpu.mlir" \
    "--iree-hal-substitute-executable-source=decode_dispatch_24=$KERNELS/dispatch_24_gpu.mlir" \
    -o "$OUTPUT"
fi

echo ""
ls -lh "$OUTPUT"
echo ""
echo "Run: python3 scripts/chat.py --vmfb $OUTPUT"
