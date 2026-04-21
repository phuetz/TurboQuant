# Setup script for TurboQuant on 2x RTX 3090 (48 GB VRAM) — Windows
# Run in PowerShell: .\scripts\setup_rtx3090.ps1

Write-Host "=== TurboQuant Setup for 2x RTX 3090 (Windows) ===" -ForegroundColor Cyan

# 1. GPU Check
Write-Host "`n--- GPU Check ---" -ForegroundColor Yellow
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Install PyTorch with CUDA
Write-Host "`n--- Installing PyTorch (CUDA 12.4) ---" -ForegroundColor Yellow
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install TurboQuant + dependencies
Write-Host "`n--- Installing TurboQuant ---" -ForegroundColor Yellow
pip install -e ".[eval]"
pip install accelerate bitsandbytes

# 4. HuggingFace login reminder
Write-Host "`n--- HuggingFace Login ---" -ForegroundColor Yellow
Write-Host "Run:  huggingface-cli login"
Write-Host "Paste your token from https://huggingface.co/settings/tokens"

# 5. Sanity check
Write-Host "`n--- Sanity Check ---" -ForegroundColor Yellow
python -c @"
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)')
from turboquant_harness import TurboQuantCache
print('TurboQuantCache: OK')
"@

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host @"

Run the benchmark:

  # Quick test (30 sec):
  python scripts\quicktest_cuda.py

  # Devstral 24B FP16 (uses both GPUs):
  python scripts\benchmark_devstral.py

  # Devstral 24B 4-bit (fits on 1 GPU):
  python scripts\benchmark_devstral.py --load-in-4bit

  # Maximum compression (4-bit model + 2-bit KV cache):
  python scripts\benchmark_devstral.py --load-in-4bit --nbits 2
"@
