import os, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from hsi_dataset import HSIDataset

def build_model(name, latent_dim, device):
    if name == "conv":
        from models.vae_conv import VAEConv
        model = VAEConv(in_ch=200, latent_dim=latent_dim)
    elif name == "vit":
        from models.vit_vae import ViTVAE
        model = ViTVAE(in_ch=200, latent_dim=latent_dim)
    elif name == "vit_spatial":
        from models.vit_vae import ViTVAESpatial
        model = ViTVAESpatial(in_ch=200, latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["conv", "vit", "vit_spatial"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pattern", default="/scratch/mkhorram/Soil/VNIR2/*.bip")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--latent", type=int, default=1024,
                    help="latent_dim (1024 for conv/vit, 8 for vit_spatial)")
    ap.add_argument("--out", default="latents.npz")
    ap.add_argument("--max-samples", type=int, default=1000,
                    help="use only the first N samples (0 = all)")   
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    ds = HSIDataset(args.pattern)
    if args.max_samples > 0 and args.max_samples < len(ds):
        ds = Subset(ds, list(range(args.max_samples)))
        print(f"[INFO] Using subset of {len(ds)} samples for latent extraction")
    else:
        print(f"[INFO] Using full dataset of {len(ds)} samples")
    num_workers = args.num_workers if device.type == "cuda" else 0
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, args.latent, device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    zs = []

    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            xrec, mu, logvar = model(xb)

            if args.model == "vit_spatial":
                # mu: (B, N, latent_dim) -> (B, N*latent_dim)
                z = mu.reshape(mu.size(0), -1)
            else:
                z = mu    # (B, latent_dim)

            zs.append(z.cpu())

    z_all = torch.cat(zs, dim=0).numpy()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, z=z_all)
    print(f"[INFO] Saved latents {z_all.shape} -> {args.out}")

if __name__ == "__main__":
    main()
