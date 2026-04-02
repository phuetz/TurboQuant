@echo off
REM Setup TurboQuant on 2x RTX 3090 (Windows)
REM Run: scripts\setup_rtx3090.bat

echo === TurboQuant Setup for 2x RTX 3090 (Windows) ===

echo.
echo --- GPU Check ---
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo.
echo --- Installing PyTorch (CUDA 12.4) ---
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo.
echo --- Installing TurboQuant ---
pip install -e ".[eval]"
pip install accelerate bitsandbytes

echo.
echo --- HuggingFace Login ---
echo Run:  huggingface-cli login
echo Paste your token from https://huggingface.co/settings/tokens

echo.
echo --- Sanity Check ---
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]; from turboquant_harness import TurboQuantCache; print('TurboQuantCache: OK')"

echo.
echo === Setup complete ===
echo.
echo Quick test:       python scripts\quicktest_cuda.py
echo Devstral FP16:    python scripts\benchmark_devstral.py
echo Devstral 4-bit:   python scripts\benchmark_devstral.py --load-in-4bit
echo Max compression:  python scripts\benchmark_devstral.py --load-in-4bit --nbits 2
