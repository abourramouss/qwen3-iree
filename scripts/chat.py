#!/usr/bin/env python3
"""Qwen3-0.6B interactive chat on IREE CUDA runtime."""
import sys, time, argparse, numpy as np, os
from iree.runtime import *
from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Chat with Qwen3-0.6B via IREE")
    parser.add_argument("--vmfb", default="qwen3.vmfb", help="Compiled VMFB path")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json path (auto-detected from HF cache)")
    parser.add_argument("--f16-irpa", default=None, help="f16 stacked IRPA path")
    parser.add_argument("--q8-irpa", default=None, help="Q8_0 stacked IRPA path")
    parser.add_argument("--output-irpa", default=None, help="Output weight (transposed) IRPA path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--device", default="cuda", help="IREE device (cuda, llvm-cpu)")
    args = parser.parse_args()

    # Auto-detect tokenizer from HF cache
    if args.tokenizer is None:
        hf_cache = os.path.expanduser("~/models/qwen3-0.6b/models--Qwen--Qwen3-0.6B/snapshots")
        if os.path.isdir(hf_cache):
            snap = next(os.path.join(hf_cache, d) for d in os.listdir(hf_cache)
                        if os.path.isdir(os.path.join(hf_cache, d)))
            args.tokenizer = os.path.join(snap, "tokenizer.json")
    if not args.tokenizer or not os.path.exists(args.tokenizer):
        print("Error: tokenizer.json not found. Use --tokenizer PATH", file=sys.stderr)
        sys.exit(1)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Auto-detect IRPAs
    models_dir = os.path.expanduser("~/models")
    if args.f16_irpa is None:
        args.f16_irpa = os.path.join(models_dir, "qwen3-0.6b-f16-stacked.irpa")
    if args.output_irpa is None:
        args.output_irpa = os.path.join(models_dir, "qwen3-output-T.irpa")
    if args.q8_irpa is None:
        args.q8_irpa = os.path.join(models_dir, "qwen3-0.6b-q8-stacked.irpa")

    # Load IREE runtime
    inst = VmInstance()
    device = get_device(args.device)
    hal = create_hal_module(inst, device)
    pi = ParameterIndex()
    pi.load(args.f16_irpa)
    pi.load(args.output_irpa)
    if os.path.exists(args.q8_irpa):
        pi.load(args.q8_irpa)
    params = create_io_parameters_module(inst, pi.create_provider(scope="model"))
    with open(args.vmfb, "rb") as f:
        mod = VmModule.copy_buffer(inst, f.read())
    ctx = VmContext(inst, modules=[params, hal, mod])
    run = mod.lookup_function("run")

    EOS_TOKEN = 151645

    def to_bv(arr):
        return device.allocator.allocate_buffer_copy(
            memory_type=MemoryType.DEVICE_LOCAL,
            allowed_usage=BufferUsage.DEFAULT | BufferUsage.MAPPING,
            device=device, buffer=np.ascontiguousarray(arr),
            element_type=HalElementType.SINT_64)

    print("Qwen3-0.6B on IREE (type 'quit' to exit)\n")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in ("quit", "exit", "q"):
            break
        if not text.strip():
            continue

        prompt = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
        tokens = tokenizer.encode(prompt).ids

        a = VmVariantList(4)
        a.push_ref(to_bv(np.array(tokens, dtype=np.int64)))
        a.push_int(len(tokens))
        a.push_int(args.max_tokens)
        a.push_int(EOS_TOKEN)
        r = VmVariantList(2)

        t0 = time.perf_counter()
        ctx.invoke(run, a, r)
        dt = time.perf_counter() - t0

        out = DeviceArray(device, r.get_as_object(0, HalBufferView),
                          implicit_host_transfer=True).to_host()
        n = int(r.get_variant(1))

        # Trim at EOS (batch decode may overshoot)
        tokens_out = out[:n].tolist()
        if EOS_TOKEN in tokens_out:
            tokens_out = tokens_out[:tokens_out.index(EOS_TOKEN)]
        n = len(tokens_out)

        print(tokenizer.decode(tokens_out))
        print(f"[{n} tok, {dt:.1f}s, {n/dt:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
