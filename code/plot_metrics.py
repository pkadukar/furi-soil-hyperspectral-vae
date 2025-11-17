import os, sys
import numpy as np
import matplotlib.pyplot as plt

run = "runs/vnir2_vit_main_metrics"
tsv = os.path.join(run, "metrics.tsv")
data = np.loadtxt(tsv)  # cols: epoch, PSNR, SAM
ep   = data[:,0]
psnr = data[:,1]
sam  = data[:,2]

plt.figure()
plt.plot(ep, psnr, marker='o')
plt.xlabel("Epoch"); plt.ylabel("PSNR (dB)"); plt.title("VNIR ViT-VAE: PSNR vs Epoch")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(run, "psnr_curve.png"), bbox_inches="tight")

plt.figure()
plt.plot(ep, sam, marker='o')
plt.xlabel("Epoch"); plt.ylabel("SAM (rad)"); plt.title("VNIR ViT-VAE: SAM vs Epoch (lower is better)")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(run, "sam_curve.png"), bbox_inches="tight")
