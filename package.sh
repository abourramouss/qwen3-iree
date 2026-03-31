#!/bin/bash
# Package qwen3-iree for deployment on another Jetson AGX Orin
#
# Creates a self-contained tarball with:
#   - Pre-compiled VMFB
#   - IREE Python runtime (our custom build)
#   - Weight files (.irpa)
#   - Tokenizer
#   - Chat script
#
# Usage: ./package.sh [output.tar.gz]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${1:-qwen3-jetson-orin.tar.gz}"
STAGING="/tmp/qwen3-package"

echo "=== Packaging qwen3-iree for Jetson AGX Orin ==="

rm -rf "$STAGING"
mkdir -p "$STAGING/qwen3-iree/runtime" "$STAGING/qwen3-iree/weights" "$STAGING/qwen3-iree/scripts"

# 1. VMFB (use codegen version — most portable, no custom kernel deps)
VMFB="$SCRIPT_DIR/qwen3_codegen.vmfb"
if [ ! -f "$VMFB" ]; then
    echo "Compiling VMFB..."
    "$SCRIPT_DIR/compile.sh" --no-kernels
    VMFB="$SCRIPT_DIR/qwen3_codegen.vmfb"
fi
cp "$VMFB" "$STAGING/qwen3-iree/"
echo "  VMFB: $(ls -lh "$VMFB" | awk '{print $5}')"

# 2. Chat script
cp "$SCRIPT_DIR/scripts/chat.py" "$STAGING/qwen3-iree/scripts/"

# 3. IREE Python runtime
RUNTIME_SRC="$HOME/iree-python-runtime"
if [ -d "$RUNTIME_SRC" ]; then
    cp -r "$RUNTIME_SRC/iree" "$STAGING/qwen3-iree/runtime/"
    # Ensure __init__.py exists in _runtime_libs
    touch "$STAGING/qwen3-iree/runtime/iree/_runtime_libs/__init__.py"
    echo "  Runtime: $(du -sh "$STAGING/qwen3-iree/runtime" | awk '{print $1}')"
else
    echo "WARNING: IREE runtime not found at $RUNTIME_SRC"
fi

# 4. Weight files
MODELS="$HOME/models"
for f in qwen3-0.6b-f16-stacked.irpa qwen3-output-T.irpa qwen3-0.6b-q8-stacked.irpa; do
    if [ -f "$MODELS/$f" ]; then
        cp "$MODELS/$f" "$STAGING/qwen3-iree/weights/"
        echo "  Weight: $f ($(ls -lh "$MODELS/$f" | awk '{print $5}'))"
    else
        echo "  WARNING: $f not found"
    fi
done

# 5. Tokenizer
HF_CACHE="$HOME/models/qwen3-0.6b/models--Qwen--Qwen3-0.6B/snapshots"
if [ -d "$HF_CACHE" ]; then
    SNAP=$(ls -d "$HF_CACHE"/*/ | head -1)
    if [ -f "$SNAP/tokenizer.json" ]; then
        cp "$SNAP/tokenizer.json" "$STAGING/qwen3-iree/weights/"
        echo "  Tokenizer: $(ls -lh "$SNAP/tokenizer.json" | awk '{print $5}')"
    fi
fi

# 6. Run script
cat > "$STAGING/qwen3-iree/run.sh" << 'RUNEOF'
#!/bin/bash
# Run Qwen3-0.6B chat on Jetson AGX Orin
DIR="$(cd "$(dirname "$0")" && pwd)"

# Set max GPU clocks (requires sudo on first run)
if command -v jetson_clocks &>/dev/null; then
    sudo jetson_clocks 2>/dev/null && echo "GPU clocks set to max" || echo "Run 'sudo jetson_clocks' for max performance"
fi

export PYTHONPATH="$DIR/runtime:$PYTHONPATH"
exec python3.11 "$DIR/scripts/chat.py" \
    --vmfb "$DIR/qwen3_codegen.vmfb" \
    --weights "$DIR/weights" \
    --tokenizer "$DIR/weights/tokenizer.json" \
    "$@"
RUNEOF
chmod +x "$STAGING/qwen3-iree/run.sh"

# 7. README
cat > "$STAGING/qwen3-iree/DEPLOY.md" << 'EOF'
# Qwen3-0.6B on Jetson AGX Orin

## Prerequisites
- JetPack 5.x
- Python 3.11: `sudo apt install python3.11`
- `pip3.11 install tokenizers numpy`

## Run
```bash
./run.sh
```

## First run
Set GPU clocks to max for best performance:
```bash
sudo jetson_clocks
```

## Performance
~26 tok/s Q8_0 decode on AGX Orin at full GPU clocks (1.3 GHz).
EOF

# 8. Create tarball
echo ""
echo "Creating tarball..."
cd "$STAGING"
tar czf "$OUTPUT" qwen3-iree/
mv "$OUTPUT" "$SCRIPT_DIR/"

echo ""
ls -lh "$SCRIPT_DIR/$OUTPUT"
echo ""
echo "Deploy to another Jetson:"
echo "  scp $OUTPUT jetson2:~/"
echo "  ssh jetson2 'tar xzf $OUTPUT && cd qwen3-iree && ./run.sh'"
