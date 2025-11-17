#!/bin/bash
#SBATCH -J vnir_latent
#SBATCH -o /scratch/pkadukar/soil_proj/logs/%x_%j.out
#SBATCH -e /scratch/pkadukar/soil_proj/logs/%x_%j.err

# ---- GPU/CPU/Memory/Time ----
#SBATCH -x sc099
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00

set -euo pipefail
echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

# ---- Conda env activation ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate furi310

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# ---- Go to code dir ----
cd /scratch/pkadukar/soil_proj/code

# ---- Quick CUDA sanity check ----
python - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
print(f"[CHECK] torch.cuda.is_available(): {ok}")
if ok:
    i = torch.cuda.current_device()
    print(f"[CHECK] Device: {torch.cuda.get_device_name(i)}")
    print(f"[CHECK] Capability: {torch.cuda.get_device_capability(i)}")
else:
    sys.exit("[ERROR] CUDA not available in this job.")
PY

echo "[INFO] Extracting Conv-VAE latents..."
srun -u /home/pkadukar/venvs/furi310/bin/python -u extract_latents.py \
  --model conv \
  --ckpt /scratch/pkadukar/soil_proj/runs/vnir2_conv_main_metrics/ep025.pt \
  --pattern "/scratch/mkhorram/Soil/VNIR2/*.bip" \
  --batch 8 \
  --num-workers 4 \
  --latent 1024 \
  --max-samples 110 \
  --out /scratch/pkadukar/soil_proj/runs/vnir2_conv_main_metrics/latents_conv_ep025.npz

echo "[INFO] Extracting spatial ViT-VAE latents..."
srun -u /home/pkadukar/venvs/furi310/bin/python -u extract_latents.py \
  --model vit_spatial \
  --ckpt /scratch/pkadukar/soil_proj/runs/vnir2_vit_spatial_main_metrics/ep025.pt \
  --pattern "/scratch/mkhorram/Soil/VNIR2/*.bip" \
  --batch 8 \
  --num-workers 4 \
  --latent 8 \
  --max-samples 110 \
  --out /scratch/pkadukar/soil_proj/runs/vnir2_vit_spatial_main_metrics/latents_vit_spatial_ep025.npz

echo "[INFO] Latent extraction finished."
date

