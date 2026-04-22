# Changelog

## Unreleased — RTX 3090 / CUDA optimisations (2026-04-22)

### Performance

- **`bed2010`** — *Keep KV cache on GPU.* Removed the `_quantize`/`_dequantize` CPU↔GPU roundtrip in `turboquant_harness/cache.py` and `quantization.py`. Lazy device migration of rotation matrix, centroids, boundaries and Walsh-Hadamard sign vectors so they follow the input tensor's device. `MseTensorCode`/`ProdTensorCode` now keep `indices`/`norms` on the input device; the cache itself stores them as packed `torch.uint8` on `cuda:0` (verified via `quicktest_cuda.py`). All 24 Python tests still pass.

  **Impact on Qwen2.5-7B-Instruct, 2× RTX 3090, 100 new tokens:**

  | Cache | short (12 tok prompt) | medium (71 tok) | long_context (2628 tok) |
  |---|---:|---:|---:|
  | FP16 | 13.0 tok/s | 15.0 tok/s | 13.0 tok/s |
  | TQ-4bit before patch | 11.5 | 9.4 | **3.1** |
  | TQ-4bit after patch | 11.6 | 11.6 | **10.3** (×3.3) |
  | TQ-2bit before patch | 10.8 | 9.3 | 3.2 |
  | TQ-2bit after patch | 9.7 | 10.1 | **9.3** (×2.9) |

  Output identical to the CPU-roundtrip path on the smoke test (FP16 and TQ-4bit produce the same first ~50 tokens of Qwen's response). Pure data-movement optimisation, no numerical change. The remaining gap to FP16 (~21% on long context) is dominated by the still-materialised dequant tensor in attention; fused dequant during `Q @ K.T` is the natural next step.

### Bug fixes

- **`77d68c7`** — *Fix `total_mem` → `total_memory`.* PyTorch renamed the attribute on `CudaDeviceProperties` between releases; the previous name no longer exists on torch 2.11+. Patched 4 scripts that crashed at startup: `quicktest_cuda.py`, `benchmark_devstral.py`, `setup_rtx3090.{ps1,sh}`.

### Documentation

- **`3d80b02`** — *First CUDA RTX 3090 benchmark in the repo.* Added [`docs/benchmarks/rtx3090_qwen7b.md`](docs/benchmarks/rtx3090_qwen7b.md) with raw JSON traces in `docs/benchmarks/qwen7b_rtx3090*.json`. The 2-bit run identified data movement as the speed bottleneck (4-bit and 2-bit had nearly identical throughput before the GPU patch), which led to `bed2010`.
- **`f525118`** — Re-bench at 2-bit with the GPU patch, included in the doc and JSON trace.

### How to reproduce

```bash
pip install -e .
pip install accelerate bitsandbytes
python scripts/quicktest_cuda.py
python scripts/benchmark_devstral.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 100 \
  --output artifacts/benchmark_qwen7b_3090.json
```

### Scaling test — context length sweep

Pushed the same harness to 8K and 16K. With bnb 4-bit weights (frees ~10 GB), TurboQuant matches FP16 throughput at 8K (5.5 vs 5.6 tok/s). At 16K, **both** caches OOM in SDPA's `Q @ K.T` materialisation (25 GiB alloc), not in the KV storage — TurboQuant can't help when the bottleneck is the attention matrix itself. Flash-Attention is the missing piece for 32K+ on RTX 3090.

### Open follow-ups

1. Fused dequant inside attention (`Q @ K.T`) to close the remaining 21% vs FP16 at medium context.
2. Flash-Attention path so context can scale past SDPA's FP16 logits ceiling.
3. NIAH retrieval validation on RTX 3090 (currently Apple Silicon only in the README). The legacy `hf_runner.generate_incremental` path needs a transformers v5 compat shim — separate work.
4. Custom CUDA kernel for the iterative FWHT (Python overhead is now visible after the GPU patch).
5. Bench with bigger model (Devstral 24B, Llama-70B Q4_K_M) to confirm the trend at scale.
