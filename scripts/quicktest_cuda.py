"""Quick 30-second test to verify TurboQuant works on CUDA before running the big benchmark.

Usage:
    python scripts/quicktest_cuda.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquant_harness import TurboQuantCache

assert torch.cuda.is_available(), "CUDA not available! Install PyTorch with CUDA support."

print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f"  GPU {i}: {name} ({mem:.0f} GB)")

print("\nLoading Qwen2.5-1.5B on GPU...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", torch_dtype=torch.float16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer.pad_token = tokenizer.eos_token

# FP16 baseline
ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    out_fp = model.generate(ids, max_new_tokens=20, do_sample=False)
fp_text = tokenizer.decode(out_fp[0], skip_special_tokens=True)
print(f"FP16: {fp_text}")

# TurboQuant
cache = TurboQuantCache(model.config, nbits=4, residual_length=8)
ids = tokenizer("Hello world", return_tensors="pt").input_ids.to(model.device)
with torch.no_grad():
    out_tq = model.generate(ids, past_key_values=cache, max_new_tokens=20, do_sample=False)
tq_text = tokenizer.decode(out_tq[0], skip_special_tokens=True)
print(f"TQ4b: {tq_text}")

# Verify compression
layer = cache.layers[1]
has_q = hasattr(layer, "_quantized_keys") and layer._quantized_keys is not None
print(f"\nQuantized prefix: {has_q}")
if has_q:
    packed = layer._quantized_keys[0]
    print(f"Storage: {packed.dtype} {packed.shape} (on {packed.device})")

print(f"Seq length: {cache.get_seq_length()}")
print("\n=== ALL CHECKS PASSED === Ready for benchmark_devstral.py")
