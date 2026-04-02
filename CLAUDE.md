# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Rust + Python implementation of TurboQuant from "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (arXiv:2504.19874). Compresses LLM KV caches 4-8x via random rotation + optimal scalar quantization. Drop-in replacement for HuggingFace DynamicCache.

## Build & Test Commands

```bash
cargo build                      # Debug build
cargo build --release            # Release build (needed for benchmarks)
cargo test                       # All Rust tests (75 tests)
cargo test technical             # Just tests/technical.rs
cargo test kv_cache              # Just tests/kv_cache.rs
cargo test stress                # Just tests/stress.rs (35 stress tests)

# Python tests (23 tests)
python -m unittest discover -s python_tests -v

# Smoke test with real model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant_harness import TurboQuantCache
model = AutoModelForCausalLM.from_pretrained('gpt2')
tok = AutoTokenizer.from_pretrained('gpt2'); tok.pad_token = tok.eos_token
cache = TurboQuantCache(model.config, nbits=4, residual_length=8)
ids = tok('Hello world', return_tensors='pt').input_ids
out = model.generate(ids, past_key_values=cache, max_new_tokens=20, do_sample=False)
print(tok.decode(out[0]))
"

# Install as package
pip install -e .
```

## Key Binaries

```bash
cargo run --release --bin paper_validation -- mse --dimension 256 --samples 128 --bits 1,2,3,4 --seed 7
cargo run --release --bin kv_cache_benchmark -- --mode prod_mse --layers 32 --batch 1 --heads 8 --head-dim 128 --prefill 8192 --decode-tokens 128 --residual-length 128 --bits 4 --skip-layers 0,31 --seed 7
cargo run --release --bin ann_benchmark -- --method turboquant --data embeddings.txt --train-size 100000 --query-size 1000 --bits 2,4 --ks 1,2,4,8,16,32,64 --seed 7
```

## Architecture

### Python HF Integration (turboquant_harness/)

The primary user-facing API:

- **`cache.py`**: `TurboQuantCache(Cache)` — drop-in for `model.generate()`. Uses HF `QuantizedLayer` API for real compressed storage. `TurboQuantLayer` (MSE) and `TurboQuantProdLayer` (MSE + QJL) implement `_quantize()`/`_dequantize()`. Cache is stored as packed uint4/uint2 between generation steps; decompressed on-the-fly during attention.
- **`packing.py`**: `pack_uint4`/`unpack_uint4`, `pack_uint2`/`unpack_uint2` — bit packing for 4x storage reduction.
- **`quantization.py`**: `TorchTurboQuantMse`, `TorchTurboQuantProd`, `TorchMixedQuantizer` — core PyTorch quantizers. `_WalshHadamardRotation` for O(d log d) fast rotation via iterative FWHT. `_gaussian_orthogonal_matrix` for paper-exact O(d²) rotation.
- **`hf_runner.py`**: `generate_incremental()` — manual generation loop with quantize/materialize cycle. Used by evaluation scripts. Separate from `cache.py` (legacy path).

### Rust Core (src/)

- **`mse.rs`** / **`prod.rs`**: Core quantizers (paper-exact algorithms).
- **`rotation.rs`**: `DenseGaussian` (O(d²)) and `WalshHadamard` (O(d log d)) backends. Batch GEMM for DenseGaussian.
- **`lloyd_max.rs`**: Optimal codebook solver with binary search lookup and LRU cache.
- **`packed.rs`**: `PackedBits` with bulk `unpack_values_into()` for hot-path performance.
- **`kv.rs`**: `TurboQuantKvCache` with `QuantizedKvCacheLayer` (quantized prefix + residual window) and `FullPrecisionKvCacheLayer` (skip layers). `calibrate_skip_layers()` for automatic outlier detection.
- **`mixed.rs`**: `OutlierSplitPlan` for non-integer bit rates.
- **`experiment.rs`**: Validation metrics. Note: upper bound formula is `sqrt(3) * PI / 2 * 4^(-b)` (not `sqrt(3*PI)/2`).

### Tests

- `tests/technical.rs`: Core quantizer correctness (8 tests)
- `tests/kv_cache.rs`: KV cache operations (5 tests)
- `tests/mixed_precision.rs`: Outlier handling (5 tests)
- `tests/stress.rs`: Edge cases, high dims, large batches (35 tests)
- `tests/pq.rs`, `tests/rabitq.rs`: Baseline quantizers (7 tests)
- `python_tests/test_harness.py`: Original harness tests (7 tests)
- `python_tests/test_cache.py`: Cache integration, packing, WHT (16 tests)

## Quantizer Spec Format

- `mse:<bits>` / `prod:<bits>` — fixed bit width, dense Gaussian rotation
- `fast_mse:<bits>` / `fast_prod:<bits>` — Walsh-Hadamard rotation
- `mixed_mse:<outliers>:<outlier_bits>:<regular_bits>` — non-integer effective bit rates
- `none` — full precision passthrough

## Key Invariants

- All quantizers require dimension >= 2 and are seeded deterministically.
- `TurboQuantProd` requires bit width >= 1 (allocates b-1 to MSE, 1 to QJL).
- Python code is pure PyTorch — no CUDA kernels, works on any backend (CPU, ROCm, MPS, XPU, DirectML).
- `TurboQuantCache` requires transformers >= 5.0 (QuantizedLayer API).
- GQA is transparent: tensors already have shape `[batch, num_kv_heads, seq, head_dim]`. Tested with Qwen2.5 (GQA 6:1).
- Device handling: `_quantize()` moves tensors to CPU for quantization, `_dequantize()` moves back to original device. This ensures compatibility with DirectML, MPS, and other backends.
- Rust edition 2024; uses `nalgebra` for linear algebra and `rayon` for parallelism.

## Benchmark Scripts

- `scripts/quicktest_cuda.py` — 30-second CUDA sanity check (Qwen2.5-1.5B)
- `scripts/benchmark_devstral.py` — Full benchmark of Devstral 24B on multi-GPU (supports `--load-in-4bit`, `--nbits`, `--mode`)
- `scripts/setup_rtx3090.bat` / `.ps1` — Windows setup scripts for 2x RTX 3090
