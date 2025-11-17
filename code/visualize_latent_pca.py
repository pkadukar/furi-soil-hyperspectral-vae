import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="npz file from extract_latents.py")
    ap.add_argument("--out", default="latent_pca.png")
    args = ap.parse_args()

    data = np.load(args.npz)
    z = data["z"]  # (N, D)
    print("[INFO] Loaded latents:", z.shape)

    z_centered = z - z.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z2d = z_centered @ Vt[:2].T  # (N,2)

    plt.figure(figsize=(6, 5))
    plt.scatter(z2d[:, 0], z2d[:, 1], s=5, alpha=0.5)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Latent space PCA projection")
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, bbox_inches="tight")
    print("[INFO] Saved plot ->", args.out)

if __name__ == "__main__":
    main()

