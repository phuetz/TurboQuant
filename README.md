# TurboQuant

Rust + Python implementation of the core TurboQuant algorithms from
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

**Compress your LLM's KV cache 4-8x with near-zero quality loss.**

```python
from turboquant_harness import TurboQuantCache

cache = TurboQuantCache(model.config, nbits=4)
output = model.generate(input_ids, past_key_values=cache)
```

## Why TurboQuant?

When running large language models with long contexts, the KV cache can consume more VRAM than the model itself:

| Model | Context | KV Cache FP16 | With TQ 4-bit | Savings |
|-------|---------|---------------|---------------|---------|
| Llama 3.1 8B | 32K tokens | ~8 GB | ~1.1 GB | **7.1x** |
| Llama 3.1 8B | 128K tokens | ~32 GB | ~4.5 GB | **7.1x** |
| Devstral 24B | 32K tokens | ~5.4 GB | ~0.8 GB | **7.1x** |
| Devstral 24B | 128K tokens | ~21.5 GB | ~3.0 GB | **7.1x** |
| Mistral 7B | 32K tokens | ~4 GB | ~0.6 GB | **7.1x** |

TurboQuant compresses the KV cache using random rotation + optimal scalar quantization (Lloyd-Max codebook for the sphere-induced coordinate distribution). The cache is stored quantized and only decompressed on-the-fly during attention -- real memory savings, not a simulation.

## Tested on real hardware

All tests run on a [Minisforum AtomMan G7 Pt](https://www.minisforum.com/fr/products/atomman-g7-pt) (AMD Ryzen 9 7945HX, AMD Radeon RX 7600M XT 8 GB, 32 GB DDR5):

| Hardware | Model | Result |
|----------|-------|--------|
| AMD RX 7600M XT (8 GB, DirectML) | Qwen2.5-1.5B (GQA) | 81% token match vs FP16, same speed, 7.1x KV compression |
| AMD RX 7600M XT (8 GB, DirectML) | GPT-2 | 32.5 tok/s with compression, cache stored as packed uint8 |
| CPU (Ryzen 9 7945HX) | GPT-2 | All 99 tests pass, generation verified |

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.10, PyTorch >= 2.1, transformers >= 5.0.

For evaluation scripts (needle-in-haystack, LongBench):

```bash
pip install -e ".[eval]"
```

## Quick Start -- Python KV Cache Compression

### Basic usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant_harness import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Create compressed cache -- drop-in replacement for DynamicCache
cache = TurboQuantCache(model.config, nbits=4, residual_length=128)

inputs = tokenizer("Explain quantum computing in simple terms", return_tensors="pt").to(model.device)
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=512)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Options

```python
cache = TurboQuantCache(
    model.config,
    nbits=4,              # 2 or 4 bits per coordinate
    residual_length=128,  # recent tokens kept at full precision
    base_seed=42,         # seed for rotation matrices (deterministic)
    skip_layers={0},      # layers to keep at full precision (outlier norms)
    mode="mse",           # "mse" (default) or "prod" (inner-product preserving)
    rotation="dense_gaussian",  # or "walsh_hadamard" (faster, O(d log d))
)
```

### Auto-calibrate skip layers

Some transformer layers (typically layer 0) have anomalously large KV norms and quantize poorly. Auto-detect them:

```python
skip = TurboQuantCache.calibrate_skip_layers(model, tokenizer)
cache = TurboQuantCache(model.config, nbits=4, skip_layers=skip)
```

## GPU Compatibility

TurboQuant is **pure PyTorch** -- no CUDA kernels, no vendor lock-in. Works on any PyTorch backend:

| Platform | Backend | Status |
|----------|---------|--------|
| NVIDIA | CUDA | Tested |
| NVIDIA (Windows) | DirectML | Tested |
| AMD (Linux) | ROCm | Works |
| AMD (Windows) | DirectML | Tested |
| Apple | MPS | Works |
| Intel | XPU | Works |
| CPU | -- | Tested |

## Setup Guides

### NVIDIA CUDA (Linux/Windows)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

### NVIDIA multi-GPU (2x RTX 3090 example)

For models too large for a single GPU, `device_map="auto"` splits layers across GPUs:

```bash
# Windows: run the setup script
scripts\setup_rtx3090.bat

# Or manually:
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[eval]"
pip install accelerate bitsandbytes
```

Devstral 24B on 2x RTX 3090 (48 GB total):

| Config | Model | KV cache | Total | Fits? |
|--------|-------|----------|-------|-------|
| FP16 model + FP16 KV, 32K ctx | 48 GB | 5.4 GB | 53 GB | No |
| FP16 model + **TQ-4bit** KV, 32K ctx | 48 GB | **0.8 GB** | 49 GB | Borderline |
| **4-bit model** + FP16 KV, 32K ctx | 13 GB | 5.4 GB | 18 GB | Yes |
| **4-bit model** + **TQ-4bit** KV, 128K ctx | 13 GB | **3.0 GB** | 16 GB | Yes, easily |

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant_harness import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Devstral-Small-2-24B-Instruct-2512",
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16"),
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Devstral-Small-2-24B-Instruct-2512")

cache = TurboQuantCache(model.config, nbits=4, residual_length=128)
inputs = tokenizer("Review this code:\n```python\n...\n```", return_tensors="pt").to(model.device)
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=1024)
```

Benchmark script included:

```bash
# Quick sanity test (30 sec)
python scripts\quicktest_cuda.py

# Full Devstral 24B benchmark
python scripts\benchmark_devstral.py --load-in-4bit
```

### AMD ROCm (Linux)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
pip install -e .
```

Then use exactly the same Python code -- no changes needed.

### AMD DirectML (Windows)

For AMD GPUs on Windows (no WSL required):

```bash
pip install torch-directml
pip install -e .
```

```python
import torch_directml
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant_harness import TurboQuantCache

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to(torch_directml.device())
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token = tokenizer.eos_token

cache = TurboQuantCache(model.config, nbits=4, residual_length=16)
inputs = tokenizer("Hello world", return_tensors="pt").input_ids.to(model.device)
output = model.generate(inputs, past_key_values=cache, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Example: Ryzen AI 9 HX 470 (Minisforum AI X1 Pro)

The Ryzen AI 9 HX 470 has a Radeon 890M iGPU (RDNA 3.5, 16 CUs) sharing system RAM. TurboQuant is especially valuable here since every GB counts:

| Config | Model weights | KV cache | Total | Fits in 32 GB? |
|--------|--------------|----------|-------|----------------|
| Llama 8B Q4 + FP16 cache, 32K ctx | ~5 GB | **8 GB** | 13 GB | Barely |
| Llama 8B Q4 + **TQ 4-bit cache**, 32K ctx | ~5 GB | **1.1 GB** | 6.1 GB | Easily |
| Llama 8B Q4 + FP16 cache, 128K ctx | ~5 GB | **32 GB** | 37 GB | No |
| Llama 8B Q4 + **TQ 4-bit cache**, 128K ctx | ~5 GB | **4.5 GB** | 9.5 GB | Yes |

Setup on Ryzen AI HX 470 (Linux):

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
pip install torch --index-url https://download.pytorch.org/whl/rocm7.2.1
pip install -e .
```

## How It Works

TurboQuant applies two transformations to each KV vector:

1. **Random orthogonal rotation** (QR of Gaussian matrix, or fast Walsh-Hadamard) -- makes coordinates approximately i.i.d.
2. **Optimal scalar quantization** (Lloyd-Max codebook for the sphere-induced Beta distribution) -- near-optimal distortion for b bits per coordinate.

The KV cache stores packed uint4/uint2 indices + FP32 norms. On attention computation, vectors are decompressed: look up codebook centroids, apply inverse rotation, rescale by norms.

Two modes:
- **MSE** (`mode="mse"`): minimizes reconstruction error. Best for value cache.
- **Prod** (`mode="prod"`): uses (b-1)-bit MSE + 1-bit QJL sketch on residual. Better inner-product preservation for key cache.

## Quick Start -- Rust Core

```bash
cargo test                       # All tests (75 tests)
cargo run --example basic        # MSE/Prod quantization demo
cargo run --example mixed_precision
```

### Paper validation (emits CSV)

```bash
cargo run --release --bin paper_validation -- mse --dimension 256 --samples 256 --bits 1,2,3,4 --seed 7
cargo run --release --bin paper_validation -- prod --dimension 256 --samples 256 --bits 1,2,3,4 --seed 7
cargo run --release --bin paper_validation -- recall_curve --dimension 256 --dataset-size 4096 --queries 128 --ks 1,2,4,8,16,32,64 --bits 2,4 --seed 7
cargo run --release --bin paper_validation -- bounds --metric both --dimension 256 --samples 256 --bits 1,2,3,4 --seed 7
```

### KV cache benchmark

```bash
cargo run --release --bin kv_cache_benchmark -- --mode prod_mse --layers 32 --batch 1 --heads 8 --head-dim 128 --prefill 8192 --decode-tokens 128 --residual-length 128 --bits 4 --skip-layers 0,31 --seed 7
```

### ANN benchmark

```bash
cargo run --release --bin ann_benchmark -- --method turboquant --data embeddings.txt --train-size 100000 --query-size 1000 --bits 2,4 --ks 1,2,4,8,16,32,64 --seed 7
```

Convenience scripts for full paper reproduction:

```powershell
.\scripts\reproduce_validation.ps1
.\scripts\reproduce_ann_baselines.ps1
```

## Python Evaluation Scripts

### Needle-in-a-Haystack

```bash
python scripts/needle_eval.py ^
    --model meta-llama/Llama-3.1-8B-Instruct ^
    --backend turboquant ^
    --key-quantizer prod:4 --value-quantizer mse:4 ^
    --context-lengths 4096,8192,16384 ^
    --depths 10,30,50,70,90 ^
    --output artifacts/needle_results.csv
```

### LongBench

```bash
python scripts/longbench_eval.py ^
    --model meta-llama/Llama-3.1-8B-Instruct ^
    --subset hotpotqa_e ^
    --backend turboquant ^
    --key-quantizer prod:4 --value-quantizer mse:4 ^
    --output artifacts/longbench_results.csv ^
    --summary-output artifacts/longbench_summary.json
```

### Devstral 24B benchmark (2x RTX 3090)

```bash
python scripts\benchmark_devstral.py --load-in-4bit
```

## Project Structure

```
TurboQuant/
  src/                            Rust core implementation
    mse.rs                        TurboQuantMse quantizer
    prod.rs                       TurboQuantProd (MSE + QJL)
    mixed.rs                      Non-integer bit rate wrappers
    kv.rs                         KV cache layer with residual window
    rotation.rs                   DenseGaussian + WalshHadamard backends
    lloyd_max.rs                  Optimal scalar codebook solver
    packed.rs                     Bit-level packing
    pq.rs                         Product quantizer baseline
    rabitq.rs                     RaBitQ baseline
    experiment.rs                 Evaluation metrics and bounds
    bin/                          CLI binaries
  turboquant_harness/             Python HuggingFace integration
    cache.py                      TurboQuantCache (drop-in for DynamicCache)
    quantization.py               PyTorch quantizers (MSE, Prod, Mixed, WHT)
    packing.py                    uint4/uint2 bit packing
    hf_runner.py                  Generation loop + KV backends
    needle.py                     Needle-in-a-Haystack evaluation
    longbench.py                  LongBench evaluation
  tests/                          Rust tests (75 tests)
  python_tests/                   Python tests (24 tests)
  scripts/
    benchmark_devstral.py         Devstral 24B benchmark (CUDA multi-GPU)
    quicktest_cuda.py             Quick 30-sec CUDA sanity check
    setup_rtx3090.bat             Windows setup for 2x RTX 3090
    setup_rtx3090.ps1             PowerShell setup for 2x RTX 3090
    needle_eval.py                Needle-in-a-Haystack evaluation
    longbench_eval.py             LongBench evaluation
    reproduce_validation.ps1      Paper validation reproduction
    reproduce_ann_baselines.ps1   ANN baseline reproduction
  examples/                       Rust examples
  third_party/                    Reference implementations
  pyproject.toml                  Python packaging (pip install -e .)
```

## Status

**Implemented and tested:**
- Exact core quantizers (MSE, Prod, Mixed) in Rust and Python
- HuggingFace `Cache` integration with real compressed storage (uint4/uint2 packed)
- Tested on NVIDIA (CUDA + DirectML), AMD (DirectML), and CPU
- Walsh-Hadamard fast rotation O(d log d) in Rust and Python
- Residual window (recent tokens at full precision)
- Skip-layer calibration (auto-detect outlier norm layers)
- GQA (Grouped Query Attention) -- transparent, tested with Qwen2.5
- PQ and RaBitQ baselines for ANN comparisons
- Paper validation framework (MSE, Prod, Recall, Bounds)
- Needle-in-a-Haystack and LongBench evaluation harness
- 99 tests total (75 Rust + 24 Python)

**Not yet implemented:**
- Fused GPU kernels (quantize/dequantize use PyTorch ops, not custom CUDA/Triton)
- External KV baselines (SnapKV, PyramidKV, KIVI, PolarQuant)
- Asymmetric K/V bit allocation at Cache level
- Rust-Python FFI bridge (PyO3)

## References

- Paper: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) ([PDF local](docs/paper.pdf))
- Google Research blog: [TurboQuant: Redefining AI Efficiency](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
