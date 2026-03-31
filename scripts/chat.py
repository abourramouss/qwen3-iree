#!/usr/bin/env python3
"""Qwen3-0.6B interactive chat on IREE CUDA runtime."""
import sys, time, argparse, numpy as np, os
from pathlib import Path

# Auto-detect IREE runtime
_script_dir = Path(__file__).resolve().parent
_repo_dir = _script_dir.parent
_runtime_dir = _repo_dir / "runtime"
if _runtime_dir.exists():
    sys.path.insert(0, str(_runtime_dir))

from iree.runtime import *
from tokenizers import Tokenizer


def find_file(name, search_dirs):
    """Search for a file in multiple directories."""
    for d in search_dirs:
        p = Path(d).expanduser() / name
        if p.exists():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Chat with Qwen3-0.6B via IREE")
    parser.add_argument("--vmfb", default=None, help="Compiled VMFB path")
    parser.add_argument("--weights", default=None, help="Directory containing .irpa weight files")
    parser.add_argument("--tokenizer", default=None, help="tokenizer.json path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--device", default="cuda", help="IREE device (cuda, llvm-cpu)")
    args = parser.parse_args()

    # Search paths for files
    search = [
        _repo_dir,
        _repo_dir / "weights",
        Path("~").expanduser() / "models",
        Path("."),
    ]
    if args.weights:
        search.insert(0, Path(args.weights))

    # Find VMFB
    if args.vmfb is None:
        for name in ["qwen3_codegen.vmfb", "qwen3_cuda.vmfb", "qwen3_mlir.vmfb", "qwen3.vmfb"]:
            args.vmfb = find_file(name, search)
            if args.vmfb:
                break
    if not args.vmfb or not os.path.exists(args.vmfb):
        print("Error: VMFB not found. Use --vmfb PATH or place in repo root.", file=sys.stderr)
        sys.exit(1)

    # Find tokenizer
    if args.tokenizer is None:
        args.tokenizer = find_file("tokenizer.json", search)
        if not args.tokenizer:
            # Check HF cache
            hf = Path("~").expanduser() / "models/qwen3-0.6b/models--Qwen--Qwen3-0.6B/snapshots"
            if hf.is_dir():
                snap = next(hf.iterdir())
                args.tokenizer = str(snap / "tokenizer.json")
    if not args.tokenizer or not os.path.exists(args.tokenizer):
        print("Error: tokenizer.json not found. Use --tokenizer PATH", file=sys.stderr)
        sys.exit(1)
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Find weight files
    f16_irpa = find_file("qwen3-0.6b-f16-stacked.irpa", search)
    out_irpa = find_file("qwen3-output-T.irpa", search)
    q8_irpa = find_file("qwen3-0.6b-q8-stacked.irpa", search)

    if not f16_irpa or not out_irpa:
        print("Error: weight files not found. Use --weights DIR", file=sys.stderr)
        print("  Need: qwen3-0.6b-f16-stacked.irpa, qwen3-output-T.irpa", file=sys.stderr)
        sys.exit(1)

    # Load IREE runtime
    inst = VmInstance()
    device = get_device(args.device)
    hal = create_hal_module(inst, device)
    pi = ParameterIndex()
    pi.load(f16_irpa)
    pi.load(out_irpa)
    if q8_irpa and os.path.exists(q8_irpa):
        pi.load(q8_irpa)
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

    print(f"Qwen3-0.6B on IREE ({os.path.basename(args.vmfb)}, type 'quit' to exit)\n")
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

        tokens_out = out[:n].tolist()
        if EOS_TOKEN in tokens_out:
            tokens_out = tokens_out[:tokens_out.index(EOS_TOKEN)]
        n = len(tokens_out)

        print(tokenizer.decode(tokens_out))
        print(f"[{n} tok, {dt:.1f}s, {n/dt:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
