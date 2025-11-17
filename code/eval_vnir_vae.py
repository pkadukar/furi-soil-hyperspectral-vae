import os, argparse, math
import torch, torchvision
from torch.utils.data import DataLoader
from hsi_dataset import HSIDataset

@torch.no_grad()
def psnr(a, b, max_val=1.0):
    mse = torch.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(float(mse))

@torch.no_grad()
def sam(a, b, eps=1e-8):
    # a,b: (B,C,H,W) in [0,1]
    A = a.permute(0,2,3,1).contiguous().view(-1, a.size(1))
    B = b.permute(0,2,3,1).contiguous().view(-1, b.size(1))
    dot = (A * B).sum(dim=1)
    na = A.norm(dim=1).clamp_min(eps)
    nb = B.norm(dim=1).clamp_min(eps)
    cosang = (dot / (na * nb)).clamp(-1+1e-7, 1-1e-7)
    ang = torch.arccos(cosang)
    return float(ang.mean().item())

@torch.no_grad()
def save_preview(model, dl, device, out_png, bands=(10,50,120)):
    model.eval()
    xb = next(iter(dl)).to(device)
    xrec,_,_ = model(xb)
    xrec = xrec.clamp(0,1)
    def fc(x):
        x3 = torch.stack([x[b] for b in bands], dim=0)
        x3 = (x3 - x3.amin()) / (x3.amax() - x3.amin() + 1e-8)
        return x3
    k = min(4, xb.size(0))
    tops = torch.stack([fc(xb[i].cpu()) for i in range(k)], dim=0)
    bots = torch.stack([fc(xrec[i].cpu()) for i in range(k)], dim=0)
    grid = torchvision.utils.make_grid(torch.cat([tops,bots], dim=0), nrow=k)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    torchvision.utils.save_image(grid, out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt to evaluate")
    ap.add_argument("--pattern", default="/scratch/mkhorram/Soil/VNIR2/*.bip")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    # ✨ NEW: include vit_spatial here too
    ap.add_argument("--model", choices=["conv","vit","vit_spatial"], default="vit")
    ap.add_argument("--latent", type=int, default=1024)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & loader
    ds = HSIDataset(args.pattern)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type=="cuda"),
    )

    # model
    if args.model == "conv":
        from models.vae_conv import VAEConv
        model = VAEConv(in_ch=200, latent_dim=args.latent).to(device)
    elif args.model == "vit":
        from models.vit_vae import ViTVAE
        model = ViTVAE(in_ch=200, latent_dim=args.latent).to(device)
    else:  # vit_spatial
        from models.vit_vae import ViTVAESpatial
        model = ViTVAESpatial(in_ch=200, latent_dim=args.latent).to(device)

    # load weights
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    # eval loop
    model.eval()
    n_batches, psnr_sum, sam_sum = 0, 0.0, 0.0
    for xb in dl:
        xb = xb.to(device, non_blocking=True)
        xrec, _, _ = model(xb)
        xrec = xrec.clamp(0,1)

        ps = psnr(xb, xrec); sa = sam(xb, xrec)
        psnr_sum += ps; sam_sum += sa
        n_batches += 1

    avg_psnr = psnr_sum / max(n_batches,1)
    avg_sam  = sam_sum  / max(n_batches,1)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "eval.txt"), "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Model: {args.model}  Latent: {args.latent}\n")
        f.write(f"Avg PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Avg SAM : {avg_sam:.4f} rad\n")

    # qualitative grid
    save_preview(model, dl, device, os.path.join(args.outdir, "preview_eval.png"))
    print(f"Avg PSNR: {avg_psnr:.2f} dB | Avg SAM: {avg_sam:.4f} rad")
    print("Wrote:", os.path.join(args.outdir, "eval.txt"))

if __name__ == "__main__":
    main()
