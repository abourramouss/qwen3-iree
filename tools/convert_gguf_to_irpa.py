#!/usr/bin/env python3
"""Convert Qwen3-0.6B GGUF (FP32 or Q8_0) to stacked f16 IRPA for frank-models.

Usage:
  python3 scripts/convert_gguf_to_irpa.py /path/to/Qwen3-0.6B-FP32.gguf output.irpa
  python3 scripts/convert_gguf_to_irpa.py /path/to/Qwen3-0.6B-Q8_0.gguf output.irpa
"""

import sys
import struct
import numpy as np
from pathlib import Path

# Qwen3-0.6B config
N_LAYERS = 28
DIM = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
FFN_DIM = 3072
VOCAB = 151936
N_EMBD_Q = N_HEADS * HEAD_DIM  # 2048
N_EMBD_KV = N_KV_HEADS * HEAD_DIM  # 1024

# Q8_0 format: 32 values per block, 34 bytes/block (2 byte f16 scale + 32 i8 values)
Q8_BLOCK_SIZE = 32
Q8_BYTES_PER_BLOCK = 34


def dequant_q8_0(raw_bytes, shape):
    """Dequantize Q8_0 raw bytes to f32 array."""
    n_elements = 1
    for d in shape:
        n_elements *= d
    n_blocks = n_elements // Q8_BLOCK_SIZE
    expected_bytes = n_blocks * Q8_BYTES_PER_BLOCK
    assert len(raw_bytes) == expected_bytes, f"Expected {expected_bytes} bytes, got {len(raw_bytes)}"

    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, Q8_BYTES_PER_BLOCK)
    # Scale: first 2 bytes as f16
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    # Values: bytes 2..33 as i8
    values = raw[:, 2:].view(np.int8).astype(np.float32)
    # Dequantize
    result = (scales * values).reshape(-1)
    return result.reshape(shape)


def read_gguf(path):
    """Read GGUF file and return dict of name -> numpy array (f32)."""
    with open(path, "rb") as f:
        data = f.read()

    # Parse header
    magic = struct.unpack_from("<I", data, 0)[0]
    assert magic == 0x46554747, f"Not a GGUF file: {hex(magic)}"
    version = struct.unpack_from("<I", data, 4)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    metadata_kv_count = struct.unpack_from("<Q", data, 16)[0]

    print(f"GGUF v{version}: {tensor_count} tensors, {metadata_kv_count} metadata entries")

    # Skip metadata (parse sequentially due to variable-length strings)
    offset = 24
    alignment = 32

    def read_string(off):
        length = struct.unpack_from("<Q", data, off)[0]
        s = data[off + 8 : off + 8 + length].decode("utf-8")
        return s, off + 8 + length

    def skip_value(off, vtype):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype == 8:  # string
            _, off = read_string(off)
            return off
        elif vtype == 9:  # array
            arr_type = struct.unpack_from("<I", data, off)[0]
            arr_len = struct.unpack_from("<Q", data, off + 4)[0]
            off += 12
            for _ in range(arr_len):
                off = skip_value(off, arr_type)
            return off
        else:
            return off + sizes[vtype]

    for _ in range(metadata_kv_count):
        key, offset = read_string(offset)
        vtype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if key == "general.alignment":
            alignment = struct.unpack_from("<I", data, offset)[0]
        offset = skip_value(offset, vtype)

    # Parse tensor info
    ggml_type_info = {
        0: ("f32", 1, 4),
        1: ("f16", 1, 2),
        8: ("q8_0", 32, 34),
    }

    tensors_info = []
    for _ in range(tensor_count):
        name, offset = read_string(offset)
        n_dims = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack_from("<Q", data, offset)[0])
            offset += 8
        ttype = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        toffset = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        tensors_info.append((name, dims, ttype, toffset))

    # Calculate tensor data start
    tensor_data_offset = (offset + alignment - 1) // alignment * alignment

    # Extract tensors
    tensors = {}
    for name, dims, ttype, toffset in tensors_info:
        type_name, block_size, type_size = ggml_type_info.get(ttype, ("unknown", 1, 0))
        n_elements = 1
        for d in dims:
            n_elements *= d
        byte_size = (n_elements // block_size) * type_size
        abs_offset = tensor_data_offset + toffset
        raw = data[abs_offset : abs_offset + byte_size]

        shape = list(reversed(dims))  # GGUF stores dims in reverse order

        if type_name == "f32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        elif type_name == "f16":
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32)
        elif type_name == "q8_0":
            arr = dequant_q8_0(raw, shape)
        else:
            print(f"  Skipping {name}: unsupported type {type_name}")
            continue

        tensors[name] = arr
        if len(tensors) <= 3 or name.startswith("blk.0"):
            print(f"  {name}: {shape} {type_name} -> f32")

    return tensors


def to_stacked_irpa(tensors, output_path):
    """Convert per-layer GGUF tensors to stacked f16 IRPA."""
    try:
        import iree.runtime as rt
    except ImportError:
        sys.path.insert(0, "/home/bourram/iree-python-runtime")
        import iree.runtime as rt

    pi = rt.ParameterIndex()

    # Non-stacked params
    embd = tensors["token_embd.weight"].astype(np.float16)  # [vocab, dim]
    pi.add_buffer("token_embd.weight", embd)
    print(f"  token_embd.weight: {embd.shape}")

    norm = tensors["output_norm.weight"].astype(np.float16)  # [dim]
    pi.add_buffer("output_norm.weight", norm)

    # Output weight: GGUF stores [vocab, dim], model needs [dim, vocab]
    out_w = tensors["output.weight"].astype(np.float16).T.copy()  # [dim, vocab]
    pi.add_buffer("output.weight", out_w)
    print(f"  output.weight: {out_w.shape}")

    # Stack per-layer weights
    def stack_layers(gguf_key, expected_shape):
        layers = []
        for l in range(N_LAYERS):
            key = f"blk.{l}.{gguf_key}"
            t = tensors[key].astype(np.float16)
            assert t.shape == expected_shape, f"{key}: {t.shape} != {expected_shape}"
            layers.append(t.reshape(-1))
        return np.stack(layers)  # [28, flat_size]

    stacked = {
        "stacked.attn_norm.weight": stack_layers("attn_norm.weight", (DIM,)),
        "stacked.ffn_norm.weight": stack_layers("ffn_norm.weight", (DIM,)),
        "stacked.attn_q_norm.weight": stack_layers("attn_q_norm.weight", (HEAD_DIM,)),
        "stacked.attn_k_norm.weight": stack_layers("attn_k_norm.weight", (HEAD_DIM,)),
        "stacked.attn_q.weight": stack_layers("attn_q.weight", (N_EMBD_Q, DIM)),
        "stacked.attn_k.weight": stack_layers("attn_k.weight", (N_EMBD_KV, DIM)),
        "stacked.attn_v.weight": stack_layers("attn_v.weight", (N_EMBD_KV, DIM)),
        "stacked.attn_output.weight": stack_layers("attn_output.weight", (DIM, N_EMBD_Q)),
        "stacked.ffn_down.weight": stack_layers("ffn_down.weight", (DIM, FFN_DIM)),
    }

    # FFN gate+up: GGUF has separate gate and up, we fuse them
    for l in range(N_LAYERS):
        gate = tensors[f"blk.{l}.ffn_gate.weight"].astype(np.float16)  # [ffn, dim]
        up = tensors[f"blk.{l}.ffn_up.weight"].astype(np.float16)  # [ffn, dim]
        if l == 0:
            gate_up_layers = []
        # Fuse: [dim, 2*ffn] = concat([dim, ffn_gate], [dim, ffn_up]) along last dim
        # But GGUF stores as [ffn, dim], and model expects [dim, 2*ffn]
        # gate.T = [dim, ffn], up.T = [dim, ffn]
        fused = np.concatenate([gate, up], axis=0)  # [2*ffn, dim]
        # Model expects [dim, 2*ffn] = fused.T
        gate_up_layers.append(fused.T.reshape(-1).copy())
    stacked["stacked.ffn_gate_up.weight"] = np.stack(gate_up_layers)

    for name, arr in stacked.items():
        pi.add_buffer(name, arr)
        print(f"  {name}: {arr.shape}")

    pi.create_archive_file(str(output_path))
    print(f"\nSaved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.0f} MB)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <gguf_file> <output_irpa>")
        sys.exit(1)

    gguf_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    print(f"Reading {gguf_path}...")
    tensors = read_gguf(gguf_path)
    print(f"\nConverting to stacked f16 IRPA...")
    to_stacked_irpa(tensors, output_path)


if __name__ == "__main__":
    main()
