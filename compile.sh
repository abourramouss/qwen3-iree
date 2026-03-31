#!/bin/bash
# Compile Qwen3-0.6B with custom GEMV kernels for Jetson AGX Orin (sm_87)
#
# Usage:
#   ./compile.sh                    # MLIR GPU-dialect kernels (no nvcc needed)
#   ./compile.sh --cuda             # CUDA kernels (requires nvcc)
#   ./compile.sh --no-kernels       # IREE codegen only (no custom kernels)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IREE_COMPILE="${IREE_COMPILE:-$(command -v iree-compile 2>/dev/null || echo ~/iree/build/tools/iree-compile)}"
MLIR="$SCRIPT_DIR/model/qwen3_full.mlir"
OUTPUT="${OUTPUT:-qwen3.vmfb}"

CUDA_FLAGS="--iree-hal-target-device=cuda \
  --iree-cuda-target=sm_87 \
  --iree-cuda-target-features=+ptx74 \
  --iree-hal-indirect-command-buffers=false"

MODE="${1:---mlir}"

case "$MODE" in
  --mlir|"")
    echo "=== Compiling with MLIR GPU-dialect kernels (no nvcc needed) ==="
    K="$SCRIPT_DIR/kernels/mlir"
    $IREE_COMPILE "$MLIR" $CUDA_FLAGS \
      --iree-hal-link-executables=false \
      "--iree-hal-substitute-executable-source=decode_dispatch_3=$K/dispatch_3_gpu.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_4=$K/dispatch_4_gpu.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_12=$K/dispatch_12_gpu.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_18=$K/dispatch_18_gpu.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_20=$K/dispatch_20_gpu.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_24=$K/dispatch_24_gpu.mlir" \
      -o "$OUTPUT"
    ;;

  --cuda)
    echo "=== Compiling with CUDA kernels (requires nvcc) ==="
    K="$SCRIPT_DIR/kernels/cuda"
    # Recompile PTX if needed
    for cu in "$K"/dispatch_*.cu; do
      ptx="${cu%.cu}.ptx"
      if [ "$cu" -nt "$ptx" ]; then
        echo "  nvcc: $(basename $cu)"
        nvcc -arch=sm_87 -ptx -O2 --use_fast_math "$cu" -o "$ptx"
      fi
    done
    $IREE_COMPILE "$MLIR" $CUDA_FLAGS \
      --iree-hal-executable-object-search-path="$K" \
      "--iree-hal-substitute-executable-source=decode_dispatch_3=$K/dispatch_3.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_4=$K/dispatch_4.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_12=$K/dispatch_12.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_18=$K/dispatch_18.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_20=$K/dispatch_20.mlir" \
      "--iree-hal-substitute-executable-source=decode_dispatch_24=$K/dispatch_24.mlir" \
      -o "$OUTPUT"
    ;;

  --no-kernels)
    echo "=== Compiling (IREE codegen only, no custom kernels) ==="
    $IREE_COMPILE "$MLIR" $CUDA_FLAGS -o "$OUTPUT"
    ;;

  *)
    echo "Usage: $0 [--mlir|--cuda|--no-kernels]"
    exit 1
    ;;
esac

echo ""
ls -lh "$OUTPUT"
echo ""
echo "Run: python3.11 scripts/chat.py --vmfb $OUTPUT"
