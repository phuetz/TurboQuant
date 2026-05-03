"""Benchmark TurboQuantCache disk persistence vs re-prefill.

Measures, for a real model + prompt:
- Time to prefill a long context.
- Time + disk size to save the cache.
- Time to load the cache from disk.
- Speedup of load vs re-prefill (the headline number for cross-host
  routing or session resume).

Usage::

    python scripts/bench_cache_persistence.py
    python scripts/bench_cache_persistence.py --model Qwen/Qwen2.5-7B-Instruct --ctx 8192
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant_harness import TurboQuantCache


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def make_prompt(tokenizer, ctx: int) -> torch.Tensor:
    """Build a prompt of approximately `ctx` tokens by repeating filler text."""
    base = (
        "The history of the world is a long and intricate tale of humans, "
        "machines, ideas, and accidents that compound across generations. "
    )
    text = ""
    while True:
        text += base
        ids = tokenizer(text, return_tensors="pt").input_ids
        if ids.shape[1] >= ctx:
            return ids[:, :ctx]


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_block(label: str, fn):
    cuda_sync()
    t0 = time.perf_counter()
    out = fn()
    cuda_sync()
    dt = time.perf_counter() - t0
    print(f"  {label:30s} {dt*1000:8.1f} ms")
    return out, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--nbits", type=int, default=4)
    ap.add_argument("--residual-length", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--save-load-cycles", type=int, default=3,
                    help="Run save+load N times and report median.")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    print(f"Model      : {args.model}")
    print(f"Context    : {args.ctx} tokens")
    print(f"nbits      : {args.nbits}")
    print(f"residual   : {args.residual_length}")
    print(f"Device     : {args.device}  dtype={args.dtype}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device, local_files_only=False,
    )
    model.eval()

    ids = make_prompt(tokenizer, args.ctx).to(model.device)
    print(f"Prompt shape: {tuple(ids.shape)}")
    print()

    # ----- Prefill -----
    print("Prefill (compute KV cache from scratch):")
    cache = TurboQuantCache(model.config, nbits=args.nbits,
                            residual_length=args.residual_length)
    def _prefill():
        with torch.no_grad():
            return model(ids, past_key_values=cache, use_cache=True)
    _, prefill_dt = time_block("model forward (prefill)", _prefill)
    print()

    # ----- Save -----
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.pt")

        save_dts = []
        load_dts = []
        for i in range(args.save_load_cycles):
            print(f"Cycle {i+1}/{args.save_load_cycles}:")
            _, dt_save = time_block("save_to_disk", lambda: cache.save_to_disk(path))
            save_dts.append(dt_save)
            disk_size = os.path.getsize(path)

            _, dt_load = time_block(
                "load_from_disk (to device)",
                lambda: TurboQuantCache.load_from_disk(
                    path, model_config=model.config, map_location=args.device,
                ),
            )
            load_dts.append(dt_load)
            print()

        save_dts.sort()
        load_dts.sort()
        med_save = save_dts[len(save_dts) // 2]
        med_load = load_dts[len(load_dts) // 2]

    # ----- FP16 raw payload, for ratio context -----
    text_cfg = (
        model.config.get_text_config(decoder=True)
        if hasattr(model.config, "get_text_config") else model.config
    )
    nlay = text_cfg.num_hidden_layers
    nkv = getattr(text_cfg, "num_key_value_heads", text_cfg.num_attention_heads)
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // text_cfg.num_attention_heads
    )
    fp16_raw_bytes = 2 * nlay * nkv * args.ctx * head_dim * 2  # K+V * fp16

    speedup = prefill_dt / med_load if med_load > 0 else float("inf")
    compression = fp16_raw_bytes / disk_size if disk_size > 0 else float("inf")

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Prefill (fresh)          : {prefill_dt*1000:8.1f} ms")
    print(f"  Save  (median, n={args.save_load_cycles}): {med_save*1000:8.1f} ms")
    print(f"  Load  (median, n={args.save_load_cycles}): {med_load*1000:8.1f} ms")
    print(f"  Disk size                : {fmt_bytes(disk_size)}")
    print(f"  Equivalent FP16 raw      : {fmt_bytes(fp16_raw_bytes)}")
    print(f"  Compression ratio        : {compression:.2f}x")
    print(f"  Load speedup vs prefill  : {speedup:.2f}x")
    print()
    if speedup > 2:
        print(f"  >> Cache reload is {speedup:.1f}x faster than re-prefilling. <<")
    else:
        print(f"  >> Cache reload speedup modest at this context length.")


if __name__ == "__main__":
    main()
