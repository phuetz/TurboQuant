"""Benchmark TurboQuant on Devstral Small 2 (24B) with 2x RTX 3090.

Usage:
    python scripts/benchmark_devstral.py

Requirements:
    - 2x RTX 3090 (48 GB total VRAM)
    - pip install -e ".[eval]"
    - pip install accelerate bitsandbytes
    - huggingface-cli login  (for gated model access)
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquant_harness import TurboQuantCache


def get_gpu_memory():
    """Return (used_gb, total_gb) per GPU."""
    info = []
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        info.append((used, total))
    return info


def print_gpu_status(label=""):
    if not torch.cuda.is_available():
        return
    for i, (used, total) in enumerate(get_gpu_memory()):
        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {used:.1f} / {total:.1f} GB {label}")


def generate_and_measure(model, tokenizer, prompt, cache=None, max_new_tokens=200):
    """Generate with optional TurboQuant cache, return text + timing + memory."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_tokens = ids.shape[-1]

    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    if cache is not None:
        kwargs["past_key_values"] = cache

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    dt = time.time() - t0

    generated_tokens = out.shape[-1] - prompt_tokens
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    mem = get_gpu_memory() if torch.cuda.is_available() else []

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "time_s": dt,
        "tok_per_s": generated_tokens / dt if dt > 0 else 0,
        "gpu_memory": mem,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TurboQuant on Devstral 24B")
    parser.add_argument("--model", default="mistralai/Devstral-Small-2-24B-Instruct-2512")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit (saves ~35 GB)")
    parser.add_argument("--nbits", type=int, default=4, choices=[2, 4], help="TurboQuant bits")
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--mode", default="mse", choices=["mse", "prod"])
    parser.add_argument("--skip-baseline", action="store_true", help="Skip FP16 baseline")
    parser.add_argument("--output", default="artifacts/benchmark_devstral.json")
    args = parser.parse_args()

    print(f"=== TurboQuant Benchmark: {args.model} ===")
    print()

    # --- Hardware check ---
    if torch.cuda.is_available():
        print(f"GPUs: {torch.cuda.device_count()}")
        print_gpu_status("(before load)")
    else:
        print("WARNING: No CUDA GPU detected. Running on CPU (will be very slow).")
    print()

    # --- Load model ---
    print(f"Loading {args.model}...")
    load_kwargs = dict(device_map="auto", torch_dtype=torch.float16)
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("  Using 4-bit quantization (bitsandbytes)")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded.")
    print_gpu_status("(after load)")
    print()

    # --- Prompts ---
    prompts = {
        "short": "Write a Python function that checks if a number is prime.",
        "medium": (
            "You are a senior software engineer. Review the following code and suggest improvements:\n\n"
            "```python\n"
            "def process(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if item['status'] == 'active':\n"
            "            result.append(item['name'].upper())\n"
            "    return sorted(result)\n"
            "```\n\n"
            "Provide a detailed review covering performance, readability, and edge cases."
        ),
        "long_context": (
            "The following is a technical document about distributed systems.\n\n"
            + "Distributed consensus algorithms like Raft and Paxos ensure that multiple nodes "
            * 200  # ~4K tokens of repeated filler
            + "\n\nBased on the above, explain the key differences between Raft and Paxos."
        ),
    }

    results = {"model": args.model, "config": vars(args), "runs": []}

    # --- Baseline (FP16 cache) ---
    if not args.skip_baseline:
        print("--- Baseline (FP16 KV cache) ---")
        for name, prompt in prompts.items():
            print(f"  Prompt: {name}")
            r = generate_and_measure(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
            r["cache"] = "fp16"
            r["prompt_name"] = name
            results["runs"].append(r)
            print(f"    {r['generated_tokens']} tokens in {r['time_s']:.1f}s ({r['tok_per_s']:.1f} tok/s)")
            print(f"    Output: {r['text'][:100]}...")
            print_gpu_status("(after generate)")
            print()

    # --- TurboQuant ---
    print(f"--- TurboQuant {args.nbits}-bit (mode={args.mode}) ---")

    # Auto-calibrate skip layers
    print("  Calibrating skip layers...")
    skip = TurboQuantCache.calibrate_skip_layers(model, tokenizer)
    print(f"  Skip layers: {skip}")

    for name, prompt in prompts.items():
        print(f"  Prompt: {name}")
        cache = TurboQuantCache(
            model.config,
            nbits=args.nbits,
            residual_length=args.residual_length,
            skip_layers=skip,
            mode=args.mode,
        )
        r = generate_and_measure(model, tokenizer, prompt, cache=cache, max_new_tokens=args.max_new_tokens)
        r["cache"] = f"turboquant_{args.nbits}bit_{args.mode}"
        r["prompt_name"] = name
        r["skip_layers"] = list(skip)
        r["seq_length"] = cache.get_seq_length()

        # Measure quantized storage
        total_quantized = 0
        total_residual = 0
        for layer in cache.layers:
            if hasattr(layer, "_quantized_keys") and layer._quantized_keys is not None:
                packed = layer._quantized_keys[0]
                total_quantized += packed.numel() * packed.element_size()
            if hasattr(layer, "keys") and layer.keys is not None and layer.keys.numel() > 0:
                total_residual += layer.keys.numel() * layer.keys.element_size()

        r["quantized_bytes"] = total_quantized
        r["residual_bytes"] = total_residual
        r["compression_vs_fp16"] = (
            (total_quantized + total_residual) / max(1, r["seq_length"] * 2 * 28 * 2 * 128 * 2)
            if r["seq_length"] > 0 else 0
        )

        results["runs"].append(r)
        print(f"    {r['generated_tokens']} tokens in {r['time_s']:.1f}s ({r['tok_per_s']:.1f} tok/s)")
        print(f"    Cache: {r['seq_length']} tokens, {total_quantized/1024:.0f} KB quantized + {total_residual/1024:.0f} KB residual")
        print(f"    Output: {r['text'][:100]}...")
        print_gpu_status("(after generate)")
        print()

    # --- Comparison ---
    print("=== SUMMARY ===")
    for r in results["runs"]:
        print(f"  [{r['cache']:>25s}] {r['prompt_name']:>12s}: {r['tok_per_s']:5.1f} tok/s, {r['generated_tokens']:3d} tokens")

    # --- Save ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serializable gpu memory tuples
    for r in results["runs"]:
        r["gpu_memory"] = [(round(u, 2), round(t, 2)) for u, t in r.get("gpu_memory", [])]
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
