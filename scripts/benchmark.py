#!/usr/bin/env python3
"""Benchmark all three VMFB variants across multiple prompts."""
import sys, time, os, numpy as np, argparse
from iree.runtime import *
from tokenizers import Tokenizer

PROMPTS = [
    ("short", "Hi"),
    ("medium", "What is the speed of light?"),
    ("long", "Explain the theory of relativity in simple terms and give three real-world examples of how it affects everyday life"),
]

EOS_TOKEN = 151645
MAX_TOKENS = 128


def find_tokenizer():
    hf = os.path.expanduser("~/models/qwen3-0.6b/models--Qwen--Qwen3-0.6B/snapshots")
    if os.path.isdir(hf):
        snap = next(os.path.join(hf, d) for d in os.listdir(hf) if os.path.isdir(os.path.join(hf, d)))
        return os.path.join(snap, "tokenizer.json")
    return None


def load_model(vmfb_path):
    inst = VmInstance()
    device = get_device("cuda")
    hal = create_hal_module(inst, device)
    pi = ParameterIndex()
    models = os.path.expanduser("~/models")
    pi.load(f"{models}/qwen3-0.6b-f16-stacked.irpa")
    pi.load(f"{models}/qwen3-output-T.irpa")
    q8 = f"{models}/qwen3-0.6b-q8-stacked.irpa"
    if os.path.exists(q8):
        pi.load(q8)
    params = create_io_parameters_module(inst, pi.create_provider(scope="model"))
    with open(vmfb_path, "rb") as f:
        mod = VmModule.copy_buffer(inst, f.read())
    ctx = VmContext(inst, modules=[params, hal, mod])
    return inst, device, ctx, mod


def bench_one(device, ctx, mod, tokenizer, prompt_text):
    run = mod.lookup_function("run")
    prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    tokens = tokenizer.encode(prompt).ids

    def to_bv(arr):
        return device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=BufferUsage.DEFAULT | BufferUsage.MAPPING,
            device=device, buffer=np.ascontiguousarray(arr),
            element_type=HalElementType.SINT_64)

    a = VmVariantList(4)
    a.push_ref(to_bv(np.array(tokens, dtype=np.int64)))
    a.push_int(len(tokens))
    a.push_int(MAX_TOKENS)
    a.push_int(EOS_TOKEN)
    r = VmVariantList(2)

    t0 = time.perf_counter()
    ctx.invoke(run, a, r)
    dt = time.perf_counter() - t0

    out = DeviceArray(device, r.get_as_object(0, HalBufferView),
                      implicit_host_transfer=True).to_host()
    n = int(r.get_variant(1))
    tokens_out = out[:n].tolist()
    if EOS_TOKEN in tokens_out:
        tokens_out = tokens_out[:tokens_out.index(EOS_TOKEN)]
    return len(tokens), len(tokens_out), dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vmfbs", nargs="*",
                        default=["qwen3_codegen.vmfb", "qwen3_cuda.vmfb", "qwen3_mlir.vmfb"])
    args = parser.parse_args()

    tok_path = find_tokenizer()
    if not tok_path:
        print("Error: tokenizer not found", file=sys.stderr)
        sys.exit(1)
    tokenizer = Tokenizer.from_file(tok_path)

    # Header
    print(f"{'VMFB':<25} {'Prompt':<10} {'In':>4} {'Out':>4} {'Time':>6} {'tok/s':>6}")
    print("-" * 62)

    for vmfb in args.vmfbs:
        if not os.path.exists(vmfb):
            print(f"{vmfb:<25} NOT FOUND — skipping")
            continue

        label = os.path.basename(vmfb).replace("qwen3_", "").replace(".vmfb", "")
        inst, device, ctx, mod = load_model(vmfb)

        for name, prompt in PROMPTS:
            n_in, n_out, dt = bench_one(device, ctx, mod, tokenizer, prompt)
            tps = n_out / dt if dt > 0 else 0
            print(f"{label:<25} {name:<10} {n_in:>4} {n_out:>4} {dt:>5.1f}s {tps:>5.1f}")

        del ctx, mod, inst, device
        print()


if __name__ == "__main__":
    main()
