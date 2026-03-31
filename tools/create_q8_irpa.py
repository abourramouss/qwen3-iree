#!/usr/bin/env python3
"""Create Q8_0 IRPA for Qwen3-0.6B: large weights as raw Q8_0 bytes, small weights as f16.

Stores raw Q8_0 blocks as i8 tensors. The MLIR dequants at runtime.
Stacked layout: [28, n_blocks_per_layer * 34] for per-layer weights.

Usage:
  python3.11 scripts/create_q8_irpa.py /path/to/Qwen3-0.6B-Q8_0.gguf output.irpa
"""

import sys
import struct
import numpy as np
from pathlib import Path

N_LAYERS = 28
DIM = 1024
FFN_DIM = 3072
HEAD_DIM = 128
N_HEADS = 16
N_KV_HEADS = 8
N_EMBD_Q = 2048
N_EMBD_KV = 1024
Q8_BLOCK = 32
Q8_BYTES = 34


def read_gguf_raw(path):
    """Read GGUF, return tensors as raw bytes (Q8_0 stays as bytes, f32 converted)."""
    with open(path, "rb") as f:
        data = f.read()

    magic = struct.unpack_from("<I", data, 0)[0]
    assert magic == 0x46554747
    version = struct.unpack_from("<I", data, 4)[0]
    tensor_count = struct.unpack_from("<Q", data, 8)[0]
    metadata_kv_count = struct.unpack_from("<Q", data, 16)[0]

    offset = 24
    alignment = 32

    def read_string(off):
        length = struct.unpack_from("<Q", data, off)[0]
        s = data[off + 8 : off + 8 + length].decode("utf-8")
        return s, off + 8 + length

    def skip_value(off, vtype):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if vtype == 8:
            _, off = read_string(off)
            return off
        elif vtype == 9:
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

    ggml_type_info = {0: ("f32", 1, 4), 1: ("f16", 1, 2), 8: ("q8_0", 32, 34)}

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
        tensors_info.append((name, list(reversed(dims)), ttype, toffset))

    tensor_data_offset = (offset + alignment - 1) // alignment * alignment

    result = {}
    for name, shape, ttype, toffset in tensors_info:
        type_name, block_size, type_size = ggml_type_info.get(ttype, ("?", 1, 0))
        n_elements = 1
        for d in shape:
            n_elements *= d
        byte_size = (n_elements // block_size) * type_size
        abs_offset = tensor_data_offset + toffset
        raw = data[abs_offset : abs_offset + byte_size]
        result[name] = (raw, shape, type_name)

    return result


def quantize_q8_0(arr_f32):
    """Quantize f32 array to Q8_0 raw bytes as int8. Total elements must be divisible by 32."""
    flat = arr_f32.reshape(-1).astype(np.float32)
    n = len(flat)
    assert n % 32 == 0, f"Length {n} not divisible by 32"
    n_blocks = n // 32
    blocks = flat.reshape(n_blocks, 32)
    max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = (max_abs / 127.0).astype(np.float32)
    scales = np.where(scales == 0, 1.0, scales)
    qvals = np.round(blocks / scales).clip(-128, 127).astype(np.int8)
    scales_f16 = scales.astype(np.float16)
    scales_bytes = scales_f16.view(np.uint8)  # [n_blocks, 2]
    qvals_bytes = qvals.view(np.uint8)        # [n_blocks, 32]
    raw = np.concatenate([scales_bytes, qvals_bytes], axis=1)  # [n_blocks, 34]
    return raw.reshape(-1).view(np.int8)


def dequant_q8_0(raw_bytes, shape):
    n_elements = 1
    for d in shape:
        n_elements *= d
    n_blocks = n_elements // Q8_BLOCK
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_blocks, Q8_BYTES)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    values = raw[:, 2:].view(np.int8).astype(np.float32)
    return (scales * values).reshape(shape)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <q8_gguf> <output_irpa>")
        sys.exit(1)

    gguf_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    sys.path.insert(0, "/home/bourram/iree-python-runtime")
    import iree.runtime as rt

    print(f"Reading {gguf_path}...")
    tensors = read_gguf_raw(gguf_path)

    pi = rt.ParameterIndex()

    # Non-quantized params: store as f16
    for name in ["token_embd.weight", "output_norm.weight"]:
        raw, shape, tname = tensors[name]
        if tname == "q8_0":
            arr = dequant_q8_0(raw, shape).astype(np.float16)
        elif tname == "f32":
            arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
        else:
            arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        pi.add_buffer(name, arr)
        print(f"  {name}: {arr.shape} f16")

    # Output weight: needs transpose [vocab, dim] -> [dim, vocab]
    raw, shape, tname = tensors["output.weight"]
    if tname == "q8_0":
        arr = dequant_q8_0(raw, shape).astype(np.float16)
    else:
        arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
    arr = arr.T.copy()
    pi.add_buffer("output.weight", arr)
    print(f"  output.weight: {arr.shape} f16 (transposed)")

    # Per-layer norm weights: f16 (these are tiny, no point quantizing)
    for key_suffix in ["attn_norm.weight", "ffn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight"]:
        layers = []
        for l in range(N_LAYERS):
            raw, shape, tname = tensors[f"blk.{l}.{key_suffix}"]
            if tname == "f32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape).astype(np.float16)
            else:
                arr = np.frombuffer(raw, dtype=np.float16).reshape(shape)
            layers.append(arr.reshape(-1))
        stacked = np.stack(layers)
        irpa_key = f"stacked.{key_suffix}"
        pi.add_buffer(irpa_key, stacked)
        print(f"  {irpa_key}: {stacked.shape} f16")

    # Per-layer matmul weights: store in GGUF native [N, K] layout — NO transpose.
    # Model now uses transposed GEMV: [N, K] @ [K, 1] -> [N, 1]
    q8_weights = {
        "attn_q.weight": (N_EMBD_Q, DIM),       # [2048, 1024]
        "attn_k.weight": (N_EMBD_KV, DIM),      # [1024, 1024]
        "attn_v.weight": (N_EMBD_KV, DIM),      # [1024, 1024]
        "attn_output.weight": (DIM, N_EMBD_Q),   # [1024, 2048]
        "ffn_down.weight": (DIM, FFN_DIM),        # [1024, 3072]
    }

    for key_suffix, gguf_shape in q8_weights.items():
        n_elems = gguf_shape[0] * gguf_shape[1]
        n_blocks = n_elems // Q8_BLOCK
        bytes_per_layer = n_blocks * Q8_BYTES
        layers = []
        for l in range(N_LAYERS):
            raw, _, tname = tensors[f"blk.{l}.{key_suffix}"]
            assert tname == "q8_0" and len(raw) == bytes_per_layer
            layers.append(np.frombuffer(raw, dtype=np.int8))
        stacked = np.stack(layers)
        irpa_key = f"stacked.q8.{key_suffix}"
        pi.add_buffer(irpa_key, stacked)
        print(f"  {irpa_key}: {stacked.shape} i8 (Q8_0, GGUF native [N,K])")

    # FFN gate+up: concat raw Q8_0 bytes of gate [3072,1024] + up [3072,1024] = [6144,1024]
    n_blocks_per = (FFN_DIM * DIM) // Q8_BLOCK
    bytes_per = n_blocks_per * Q8_BYTES
    layers = []
    for l in range(N_LAYERS):
        raw_gate, _, _ = tensors[f"blk.{l}.ffn_gate.weight"]
        raw_up, _, _ = tensors[f"blk.{l}.ffn_up.weight"]
        assert len(raw_gate) == bytes_per and len(raw_up) == bytes_per
        combined = np.concatenate([
            np.frombuffer(raw_gate, dtype=np.int8),
            np.frombuffer(raw_up, dtype=np.int8),
        ])
        layers.append(combined)
    stacked = np.stack(layers)
    pi.add_buffer("stacked.q8.ffn_gate_up.weight", stacked)
    print(f"  stacked.q8.ffn_gate_up.weight: {stacked.shape} i8 (Q8_0, gate+up [6144,1024])")

    pi.create_archive_file(str(output_path))
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {output_path} ({size_mb:.0f} MB)")

    # Compare with f16 size
    f16_size = (
        151936 * 1024 * 2  # token_embd
        + 1024 * 2  # output_norm
        + 1024 * 151936 * 2  # output
        + N_LAYERS * (1024 + 1024 + 128 + 128) * 2  # norms
        + N_LAYERS * (2048*1024 + 1024*1024 + 1024*1024 + 2048*1024 + 3072*1024 + 1024*3072*2) * 2  # matmul weights f16
    )
    print(f"  f16 equivalent: {f16_size / 1024 / 1024:.0f} MB")
    print(f"  compression: {f16_size / 1024 / 1024 / size_mb:.1f}x")


if __name__ == "__main__":
    main()
