#!/bin/bash
#SBATCH -J vnir_vit_spatial
#SBATCH -o /scratch/pkadukar/soil_proj/logs/%x_%j.out
#SBATCH -e /scratch/pkadukar/soil_proj/logs/%x_%j.err

# ---- GPU/CPU/Memory/Time ----
#SBATCH -x sc099
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

# ---- Conda env activation ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate furi310

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

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

# ---- Spatial ViT-VAE training run ----
srun -u /home/pkadukar/venvs/furi310/bin/python -u train_vnir_vae.py \
  --model vit_spatial \
  --pattern "/scratch/mkhorram/Soil/VNIR2/*.bip" \
  --epochs 25 \
  --batch 8 \
  --num-workers 4 \
  --lr 3e-4 \
  --beta 1e-3 \
  --latent 8 \
  --amp \
  --outdir "/scratch/pkadukar/soil_proj/runs/vnir2_vit_spatial_main_metrics"

echo "[INFO] Spatial ViT-VAE run finished."
date

#SBATCH -J vnir_vit_spatial
#SBATCH -o /scratch/pkadukar/soil_proj/logs/%x_%j.out
#SBATCH -e /scratch/pkadukar/soil_proj/logs/%x_%j.err

# ---- GPU/CPU/Memory/Time ----
#SBATCH -x sc099
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

# ---- Conda env activation ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate furi310

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

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

# ---- Spatial ViT-VAE training run ----
srun -u /home/pkadukar/venvs/furi310/bin/python -u train_vnir_vae.py \
  --model vit_spatial \
  --pattern "/scratch/mkhorram/Soil/VNIR2/*.bip" \
  --epochs 25 \
  --batch 8 \
  --num-workers 4 \
  --lr 3e-4 \
  --beta 1e-3 \
  --latent 8 \
  --amp \
  --outdir "/scratch/pkadukar/soil_proj/runs/vnir2_vit_spatial_main_metrics"

echo "[INFO] Spatial ViT-VAE run finished."
date
#!/bin/bash
#SBATCH --job-name=vnir2_vit_spatial
#SBATCH --output=/scratch/pkadukar/soil_proj/logs/vnir2_vit_spatial_%j.out
#SBATCH --error=/scratch/pkadukar/soil_proj/logs/vnir2_vit_spatial_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=32G

echo "[INFO] Node: $(hostname)"
echo "[INFO
#!/bin/bash
#SBATCH -J vnir_vit_spatial
#SBATCH -o /scratch/pkadukar/soil_proj/logs/%x_%j.out
#SBATCH -e /scratch/pkadukar/soil_proj/logs/%x_%j.err

# ---- GPU/CPU/Memory/Time ----
#SBATCH -x sc099
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00

set -euo pipefail
echo "[INFO] Node: $(hostname)"
echo "[INFO] CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
date

# ---- Conda env activation ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate furi310

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

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

# ---- Spatial ViT-VAE training run ----
srun -u /home/pkadukar/venvs/furi310/bin/python -u train_vnir_vae.py \
  --model vit_spatial \
  --pattern "/scratch/mkhorram/Soil/VNIR2/*.bip" \
  --epochs 25 \
  --batch 8 \
  --num-workers 4 \
  --lr 3e-4 \
  --beta 1e-3 \
  --latent 8 \
  --amp \
  --outdir "/scratch/pkadukar/soil_proj/runs/vnir2_vit_spatial_main_metrics"

echo "[INFO] Spatial ViT-VAE run finished."
date
