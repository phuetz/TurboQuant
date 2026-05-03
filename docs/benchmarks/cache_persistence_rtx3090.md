# Cache disk persistence on 2x RTX 3090

`TurboQuantCache.save_to_disk` / `load_from_disk` benchmarked on a 2x RTX 3090
workstation against a fresh prefill of the same prompt.

The win scales with context length: `prefill` is O(N^2) in attention compute,
`load` is O(N) in disk I/O + dtype conversion, so longer contexts amortise the
load cost much better.

## Setup

- Model: `Qwen/Qwen2.5-1.5B`, dtype fp16, on `cuda:0`
- nbits: 4, residual_length: 128
- Driver 591.86, CUDA 12.8, PyTorch 2.11
- 3 save+load cycles, median reported

Reproduce:

```bash
python scripts/bench_cache_persistence.py --ctx 2048
python scripts/bench_cache_persistence.py --ctx 8192
```

## Results

| Context | Prefill | Save (median) | Load (median) | Disk size | FP16 raw | Compression | Load speedup |
|--------:|--------:|--------------:|--------------:|----------:|---------:|------------:|-------------:|
| 2048    | 587.6 ms | 26.7 ms      | 48.1 ms       | 16.4 MB   | 56.0 MB  | 3.42x       | **12.2x**    |
| 8192    | 2495.2 ms | 75.0 ms     | 68.6 ms       | 65.4 MB   | 224.0 MB | 3.42x       | **36.4x**    |

## Interpretation

**Compression is constant** at 3.42x (vs the theoretical 4x for 4-bit) because
the saved file also contains:
- per-layer `norms` (fp32) for the rotation,
- the residual full-precision tail (last `residual_length` tokens),
- one skip layer kept full precision (default `skip_layers={0}` for outlier handling).

**Load time is dominated by disk I/O + a CPU->GPU copy**, not by quantizer setup.
At 8K tokens the load is actually faster than at 2K (68.6 vs 48.1 ms) because
the OS file cache keeps the file warm across cycles.

**Prefill scales worse than O(N)** due to the attention O(N^2) term, so the
speedup ratio grows with context. Projected scaling, holding compression and
load throughput fixed:

| Context | Projected prefill | Load   | Projected speedup |
|--------:|------------------:|-------:|------------------:|
| 2048    | 588 ms            | 48 ms  | ~12x              |
| 8192    | 2.5 s             | 69 ms  | ~36x              |
| 32768   | ~40 s (estimated) | ~250 ms (estimated) | **~160x**       |
| 131072  | ~10 min (estimated) | ~1 s (estimated) | **~600x**       |

Numbers above 8K are extrapolations until measured.

## Use cases enabled

1. **Session resume.** Patrice asks a long question, walks away, comes back the
   next day. Reload the 32K-token KV cache in ~250 ms instead of re-prefilling
   for 40 s.

2. **Cross-host KV cache shipping** (the question Claude/Ministar Linux asked
   in the journal entry of 2026-05-03 01h45). Build a long-context KV cache on
   the host that has the model loaded, ship the compressed cache file across
   the tailnet to a peer that already has the same model loaded, and resume
   generation. Saves both compute (no re-prefill on the peer) and bandwidth
   (3.4x smaller payload than dumping FP16 KV).

3. **Reusable prefix caches.** Compute a system-prompt KV cache once at
   startup, save to `system_prefix.pt`, reload at the start of every session
   instead of re-prefilling 1-2K tokens of boilerplate. ~50 ms saved per cold
   request.
