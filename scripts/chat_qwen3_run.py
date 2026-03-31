#!/usr/bin/env python3
"""Qwen3-0.6B single ctx.invoke inference."""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iree.runtime import *
from tokenizers import Tokenizer
import os

SNAP = "/home/bourram/models/qwen3-0.6b/models--Qwen--Qwen3-0.6B/snapshots"
snap_dir = next(os.path.join(SNAP, d) for d in os.listdir(SNAP) if os.path.isdir(os.path.join(SNAP, d)))
tokenizer = Tokenizer.from_file(os.path.join(snap_dir, "tokenizer.json"))

inst = VmInstance()
device = get_device("cuda")
hal = create_hal_module(inst, device)
pi = ParameterIndex()
pi.load("/home/bourram/models/qwen3-0.6b-f16-stacked.irpa")
pi.load("/home/bourram/models/qwen3-output-T.irpa")
# Also load Q8_0 params if available (for quantized decode path)
import os
q8_path = "/home/bourram/models/qwen3-0.6b-q8-stacked.irpa"
if os.path.exists(q8_path):
    pi.load(q8_path)
params = create_io_parameters_module(inst, pi.create_provider(scope="model"))
with open("/tmp/qwen3_run_SINGLE.vmfb", "rb") as f:
    mod = VmModule.copy_buffer(inst, f.read())
ctx = VmContext(inst, modules=[params, hal, mod])
run = mod.lookup_function("run")

def to_bv(arr):
    return device.allocator.allocate_buffer_copy(
        memory_type=MemoryType.DEVICE_LOCAL,
        allowed_usage=BufferUsage.DEFAULT | BufferUsage.MAPPING,
        device=device, buffer=np.ascontiguousarray(arr),
        element_type=HalElementType.SINT_64)

while True:
    try:
        text = input("\n> ")
    except (EOFError, KeyboardInterrupt):
        break
    if not text.strip():
        break
    prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(prompt).ids
    a = VmVariantList(4)
    a.push_ref(to_bv(np.array(tokens, dtype=np.int64)))
    a.push_int(len(tokens))
    a.push_int(256)
    a.push_int(151645)
    r = VmVariantList(2)
    t0 = time.time()
    ctx.invoke(run, a, r)
    dt = time.time() - t0
    out = DeviceArray(device, r.get_as_object(0, HalBufferView), implicit_host_transfer=True).to_host()
    n = int(r.get_variant(1))
    # Trim at EOS (batch decode may overshoot past end-of-turn)
    tokens_out = out[:n].tolist()
    eos_id = 151645
    if eos_id in tokens_out:
        tokens_out = tokens_out[:tokens_out.index(eos_id)]
        n = len(tokens_out)
    print(tokenizer.decode(tokens_out))
    print(f"[{n} tok, {dt:.1f}s, {n/dt:.1f} tok/s]")
