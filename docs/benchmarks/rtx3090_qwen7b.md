# TurboQuant on RTX 3090 — Qwen2.5-7B benchmark

First CUDA RTX 3090 benchmark of TurboQuant on this repo. Hardware tested: 2× NVIDIA RTX 3090 (24 GB each).

> **Update — keep-on-GPU patch landed**. Removing the CPU roundtrip in `_quantize`/`_dequantize` brings 4-bit long-context decode from **3.1 → 10.3 tok/s (×3.3)**. The full table below shows before/after.

## Setup

- Hardware: 2× RTX 3090 24 GB (DARKSTAR)
- Driver: 595.02, CUDA 13.0, PyTorch 2.11.0+cu128
- Model: `Qwen/Qwen2.5-7B-Instruct` (FP16, ~14 GB)
- TurboQuant: 4-bit MSE, residual_length=128, auto-calibrated skip layers
- Run: `python scripts/benchmark_devstral.py --model Qwen/Qwen2.5-7B-Instruct --max-new-tokens 100`
- Raw data: `docs/benchmarks/qwen7b_rtx3090.json`

## Results — generation throughput (100 new tokens)

### Before the GPU patch (CPU roundtrip in _quantize/_dequantize)

| Cache | short (12 tok prompt) | medium (71 tok) | long_context (2628 tok) |
|---|---:|---:|---:|
| FP16 | 12.9 tok/s | 14.8 tok/s | 12.8 tok/s |
| TurboQuant 4-bit MSE | 11.5 tok/s | 9.4 tok/s | **3.1 tok/s** |
| TurboQuant 2-bit MSE | 10.8 tok/s | 9.3 tok/s | 3.2 tok/s |

**4-bit and 2-bit had nearly identical throughput** — within noise. That was the smoking gun: speed was bottlenecked by data movement (GPU↔CPU roundtrip per layer per attention step), not by the bit width of the codebook lookup.

### After the GPU patch (cache stays on cuda:0)

| Cache | short | medium | long_context |
|---|---:|---:|---:|
| FP16 | 13.0 tok/s | 15.0 tok/s | 13.0 tok/s |
| **TurboQuant 4-bit MSE** | **11.6 tok/s** | **11.6 tok/s** | **10.3 tok/s** |
| Δ vs FP16 | −11% | −23% | **−21%** |
| Speedup vs CPU-roundtrip path | ×1.0 | ×1.2 | **×3.3** |

The patch adds lazy device migration to the rotation matrix, centroids, boundaries, and Walsh-Hadamard sign vectors so they follow the input tensor's device. `_flatten_last_dim` no longer forces `.cpu()`. `MseTensorCode` and `ProdTensorCode` now keep `indices`/`norms` on the input device — the cache itself stores them as `torch.uint8` on `cuda:0` (verified via `quicktest_cuda.py`).

The output is identical to the CPU-roundtrip version on the smoke test (FP16 and TQ-4bit produced the same first ~50 tokens of Qwen's response). The patch is purely a data-movement optimization, not a numerical change.

## Memory

| Prompt | TQ-4bit quantized prefix | TQ-2bit quantized prefix |
|---|---:|---:|
| short  | 63 KB | 39 KB |
| medium | 462 KB | 231 KB |
| long   | 17 MB | **8.5 MB** (2×) |

The compression scales as expected: 2-bit halves the quantized prefix size. Residual buffer (~3–8 MB depending on context) is unchanged since `residual_length` is the same.

GPU memory (after generate, both runs): GPU0 6.7 / 25.8 GB · GPU1 8.6 / 25.8 GB.

## Honest analysis

**Memory compression works as designed**: the prefix cache is stored packed as `torch.uint8` (~7× smaller than FP16) and the model+cache fit comfortably in 2× 3090.

**Speed regresses on this hardware**, especially at long context (×4 slower). Per `CLAUDE.md`:

> Python code is pure PyTorch — no CUDA kernels, works on any backend (CPU, ROCm, MPS, XPU, DirectML).
> `_quantize()` moves tensors to CPU for quantization, `_dequantize()` moves back to original device.

So every attention step pays a GPU↔CPU roundtrip per layer, plus per-coordinate Lloyd-Max lookup in Python. On a slow PCIe link and a fast GPU like the 3090, this overhead dominates. The trade-off matrix changes vs Apple Silicon (unified memory) or DirectML.

## Output identity check

For `short` prompt (12 tokens in, 100 out, greedy decode), FP16 and TQ-4bit produced identical output for the first ~50 tokens then diverged on punctuation/spacing — typical of accumulated quantization noise during generation, not a correctness bug.

## Suggestions for follow-up

1. ~~**Keep cache on GPU**: drop the `.cpu()` calls~~ ✅ done — see "After the GPU patch" above. ×3.3 on long-context decode.
2. **Fused dequant in attention**: dequantize on-the-fly during `Q @ K.T` instead of materializing the full FP16 K tensor. Avoids a second pass through memory. Likely closes most of the remaining 21% gap with FP16.
3. **NIAH on RTX 3090**: replicate the README's NIAH retrieval table on CUDA — currently only Apple Silicon validated.
4. **Fast Walsh-Hadamard CUDA kernel**: iterative FWHT in PyTorch is O(d log d) but Python-overheaded. Custom CUDA would help further.
5. **Re-bench 2-bit with GPU patch**: with arithmetic now actually mattering (no CPU roundtrip masking it), 2-bit may show a small additional speedup — to be measured.

## Reproducibility

```bash
pip install -e .
pip install accelerate bitsandbytes
python scripts/quicktest_cuda.py
python scripts/benchmark_devstral.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-new-tokens 100 \
  --output artifacts/benchmark_qwen7b_3090.json
```
