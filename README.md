# Qwen3-0.6B on IREE with hand-authored MLIR

Qwen3-0.6B inference at **25+ tok/s** on Jetson AGX Orin, compiled entirely
through [IREE](https://iree.dev) from hand-authored MLIR. No PyTorch, no
ONNX, no external CUDA kernels — the entire model and all performance-critical
GPU kernels are expressed in MLIR.

## What's here

```
model/qwen3_full.mlir     # Full Qwen3-0.6B in MLIR (decode + prefill + generate)
kernels/                  # GPU-dialect MLIR kernels for Q8_0 fused GEMV
  dispatch_3_gpu.mlir     #   attn_q  [2048,1024] — shared mem + warp shuffle
  dispatch_4_gpu.mlir     #   attn_kv [1024,1024]
  dispatch_12_gpu.mlir    #   attn_o  [1024,2048]
  dispatch_18_gpu.mlir    #   gate_up [6144,1024]
  dispatch_20_gpu.mlir    #   ffn_dn  [1024,3072]
  dispatch_24_gpu.mlir    #   vocab   [151936,1024] f16
compile.sh                # One-command compilation
scripts/chat.py           # Interactive chat interface
tools/                    # Weight conversion utilities
```

## Performance

On Jetson AGX Orin (sm_87, 64GB unified memory):

| Config | Decode tok/s |
|--------|-------------|
| IREE default codegen | 5 |
| + Q8_0 fusion (flow.dispatch.region) | 24 |
| + MLIR GPU kernels | 25 |
| llama.cpp Q8_0 CUDA | 49 |

The remaining gap to llama.cpp is IREE runtime overhead (sync + memory
allocation), not GPU kernel quality.

## Quick start

### 1. Prerequisites

- Jetson AGX Orin with JetPack 5.x
- IREE compiler and runtime ([build instructions](https://iree.dev/building-from-source/getting-started/))
- Python 3.11+ with `tokenizers` package

### 2. Prepare weights

Download the Qwen3-0.6B GGUF from Hugging Face, then convert to IREE
parameter archives:

```bash
# Convert GGUF f32 weights to stacked f16 IRPA
python3 tools/convert_gguf_to_irpa.py \
  path/to/qwen3-0.6b.gguf \
  --output ~/models/qwen3-0.6b-f16-stacked.irpa

# Convert to Q8_0 IRPA (for quantized decode)
python3 tools/create_q8_irpa.py \
  path/to/qwen3-0.6b-q8_0.gguf \
  --output ~/models/qwen3-0.6b-q8-stacked.irpa
```

### 3. Compile

```bash
./compile.sh
```

This compiles the MLIR model with all 6 GPU-dialect GEMV kernels substituted,
producing `qwen3.vmfb`.

### 4. Chat

```bash
python3 scripts/chat.py --vmfb qwen3.vmfb
```

## Architecture

### Hand-authored MLIR model

The entire Qwen3-0.6B transformer is written directly in MLIR using IREE's
`util`, `flow`, `linalg`, and `tensor` dialects. Key optimizations:

- **Static shapes** for all decode dimensions (only seq_len is dynamic)
- **Transposed GEMV**: `[N,K] @ [K,1]` layout for better GPU tiling
- **Direct KV cache**: flat contiguous tensors, no paging overhead
- **Q8_0 quantization**: fused dequant+GEMV via `flow.dispatch.region`
- **Tiled argmax**: 2-phase parallel reduction for vocab projection
- **Batch speculative decode**: 4 tokens between EOS checks

### MLIR GPU-dialect kernels

All 6 decode GEMV dispatches are hand-authored in MLIR using GPU dialect ops,
compiled through IREE's LLVM backend to PTX — no nvcc or external CUDA:

- `gpu.thread_id` / `gpu.block_id` for thread mapping
- `memref.alloc` with `#gpu.address_space<workgroup>` for shared memory
- `gpu.shuffle xor` for warp-level reduction
- `gpu.barrier` for synchronization
- `#gpu.address_space<global>` to avoid address space conversion overhead

Each kernel: 256 threads (8 warps), 8 rows per block, shared memory for
input vector, warp shuffle reduction, f32 accumulation with f16 output.

### Compilation flags

Two critical flags for correct kernel substitution:

```bash
--iree-hal-link-executables=false     # Prevent linker from overwriting kernels
--iree-hal-substitute-executable-source=NAME=FILE.mlir  # Per-dispatch substitution
```

## License

MIT
