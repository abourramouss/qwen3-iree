#!/usr/bin/env python3
"""Benchmark VMFB variants with violin plots of tok/s distribution."""
import sys, time, os, json, argparse
import numpy as np
from iree.runtime import *
from tokenizers import Tokenizer

PROMPTS = [
    "Hi",
    "What is 2+2?",
    "Define entropy",
    "What is the speed of light?",
    "Name the planets in our solar system",
    "Write a haiku about the ocean",
    "Explain what a transformer neural network is",
    "What are the differences between Python and C++?",
    "Explain the theory of relativity in simple terms and give examples",
    "Write a short story about a robot learning to paint, include dialogue",
]

EOS_TOKEN = 151645
MAX_TOKENS = 128
RUNS_PER_PROMPT = 1  # each prompt is unique enough


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


def make_plot(results, output_path):
    """Generate violin + strip plot of tok/s for each VMFB variant."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot", file=sys.stderr)
        return

    labels = list(results.keys())
    data = [results[l]["tps"] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Violin plot
    parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True,
                          showmedians=True, widths=0.7)

    # Color the violins
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.7)
    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("black")

    # Overlay individual points
    for i, d in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(d))
        ax.scatter([i] * len(d) + jitter, d, color=colors[i % len(colors)],
                   alpha=0.6, s=40, zorder=3, edgecolors="white", linewidth=0.5)

    # Stats annotations
    for i, (label, d) in enumerate(zip(labels, data)):
        mean = np.mean(d)
        std = np.std(d)
        ax.annotate(f"{mean:.1f} ± {std:.1f}",
                    xy=(i, max(d) + 0.5), ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Tokens / second", fontsize=13)
    ax.set_title("Qwen3-0.6B Decode Performance on Jetson AGX Orin", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to {output_path}")


def bench_single_vmfb(vmfb_path):
    """Benchmark one VMFB (run in subprocess to avoid semaphore issues)."""
    tok_path = find_tokenizer()
    tokenizer = Tokenizer.from_file(tok_path)
    inst, device, ctx, mod = load_model(vmfb_path)

    tps_list = []
    for i, prompt in enumerate(PROMPTS):
        n_in, n_out, dt = bench_one(device, ctx, mod, tokenizer, prompt)
        tps = n_out / dt if dt > 0 else 0
        tps_list.append(tps)
        print(f"  [{i+1:2d}/{len(PROMPTS)}] {prompt[:40]:<40s} "
              f"in={n_in:3d} out={n_out:3d} {dt:5.1f}s {tps:5.1f} tok/s",
              flush=True)

    mean = np.mean(tps_list)
    std = np.std(tps_list)
    print(f"  → mean={mean:.1f} ± {std:.1f} tok/s", flush=True)
    return tps_list


def main():
    import subprocess

    parser = argparse.ArgumentParser(description="Benchmark Qwen3 VMFB variants")
    parser.add_argument("vmfbs", nargs="*",
                        default=["qwen3_codegen.vmfb", "qwen3_cuda.vmfb", "qwen3_mlir.vmfb"])
    parser.add_argument("--plot", default="benchmark.png", help="Output plot path")
    parser.add_argument("--json", default="benchmark.json", help="Output JSON path")
    parser.add_argument("--_single", help=argparse.SUPPRESS)  # internal: run one vmfb
    args = parser.parse_args()

    # Internal mode: benchmark a single VMFB and dump JSON to stdout
    if args._single:
        tps = bench_single_vmfb(args._single)
        # Write results to a temp file
        out_file = args._single + ".bench.json"
        with open(out_file, "w") as f:
            json.dump(tps, f)
        return

    # Main mode: fork a subprocess per VMFB to avoid CUDA semaphore issues
    results = {}
    for vmfb in args.vmfbs:
        if not os.path.exists(vmfb):
            print(f"  {vmfb} — NOT FOUND, skipping\n")
            continue

        label = os.path.basename(vmfb).replace("qwen3_", "").replace(".vmfb", "")
        print(f"=== {label} ({vmfb}) ===", flush=True)

        # Run in subprocess
        ret = subprocess.run(
            [sys.executable, __file__, "--_single", vmfb],
            env={**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
        )
        if ret.returncode != 0:
            print(f"  FAILED (exit {ret.returncode})\n")
            continue

        # Read results
        bench_file = vmfb + ".bench.json"
        if os.path.exists(bench_file):
            with open(bench_file) as f:
                tps_list = json.load(f)
            os.remove(bench_file)
            results[label] = {
                "tps": tps_list,
                "mean": float(np.mean(tps_list)),
                "std": float(np.std(tps_list)),
                "prompts": [p[:60] for p in PROMPTS],
            }
        print()

    # Save JSON
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.json}")

    # Generate plot
    if len(results) > 0:
        make_plot(results, args.plot)


if __name__ == "__main__":
    main()
