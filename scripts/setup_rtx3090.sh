#!/bin/bash
# Setup script for TurboQuant on 2x RTX 3090 (48 GB VRAM)
# Run this on the target machine.

set -e

echo "=== TurboQuant Setup for 2x RTX 3090 ==="

# 1. Check GPU
echo ""
echo "--- GPU Check ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Install PyTorch with CUDA
echo ""
echo "--- Installing PyTorch (CUDA) ---"
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3. Install TurboQuant + dependencies
echo ""
echo "--- Installing TurboQuant ---"
pip install -e ".[eval]"
pip install accelerate bitsandbytes

# 4. Login to HuggingFace (needed for gated models like Devstral)
echo ""
echo "--- HuggingFace Login ---"
echo "Run: huggingface-cli login"
echo "Then paste your token from https://huggingface.co/settings/tokens"

# 5. Quick sanity check
echo ""
echo "--- Sanity Check ---"
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.0f} GB)')
from turboquant_harness import TurboQuantCache
print('TurboQuantCache: OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Run the benchmark:"
echo "  # Full FP16 model (needs ~48 GB, uses both GPUs):"
echo "  python scripts/benchmark_devstral.py"
echo ""
echo "  # 4-bit model (needs ~13 GB, fits on 1 GPU):"
echo "  python scripts/benchmark_devstral.py --load-in-4bit"
echo ""
echo "  # 4-bit model + 2-bit KV cache (maximum compression):"
echo "  python scripts/benchmark_devstral.py --load-in-4bit --nbits 2"
