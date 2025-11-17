import os, argparse, time
import torch, torchvision
import math
from torch.utils.data import DataLoader
from hsi_dataset import HSIDataset


@torch.no_grad()
def psnr(a, b, max_val=1.0):
    # a,b in [0,1], shape (B,C,H,W)
    mse = torch.mean((a - b) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val) - 10.0 * math.log10(float(mse))

@torch.no_grad()
def sam(a, b, eps=1e-8):
    # Spectral Angle Mapper (radians), averaged over pixels
    # a,b: (B,C,H,W) in [0,1]
    A = a.permute(0,2,3,1).contiguous().view(-1, a.size(1))  # (Npix, C)
    B = b.permute(0,2,3,1).contiguous().view(-1, b.size(1))
    dot = (A * B).sum(dim=1)
    na = A.norm(dim=1).clamp_min(eps)
    nb = B.norm(dim=1).clamp_min(eps)
    cosang = (dot / (na * nb)).clamp(-1+1e-7, 1-1e-7)
    ang = torch.arccos(cosang)  # radians
    return float(ang.mean().item())

# ----- loss pieces -----
def kl_divergence(mu, logvar):
    # D_KL(q(z|x) || N(0,1)) = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_one_epoch(model, dl, opt, device, scaler=None, beta=1.0):
    """
    Single clean implementation (no duplicate defs).
    Uses AMP iff `scaler` is provided.
    """
    model.train()
    total = 0.0
    n = 0

    for xb in dl:
        xb = xb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        if scaler is None:
            # FP32 path
            xrec, mu, logvar = model(xb)
            xrec = xrec.clamp(0, 1)
            recon = torch.nn.functional.mse_loss(xrec, xb, reduction="mean")
            kld = kl_divergence(mu, logvar)  # already mean
            loss = recon + beta * kld

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            # AMP path
            with torch.amp.autocast('cuda'):
                xrec, mu, logvar = model(xb)
                xrec = xrec.clamp(0, 1)
                recon = torch.nn.functional.mse_loss(xrec, xb, reduction="mean")
                kld = kl_divergence(mu, logvar)
                loss = recon + beta * kld

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        total += float(loss.detach().item())
        n += 1

    return total / max(n, 1)


@torch.no_grad()
def save_preview(model, dl, device, out_png, bands=(10, 50, 120)):
    """
    Saves a 2xK grid (K originals on top, reconstructions on bottom)
    using 3-band false color from the specified band indices.
    """
    model.eval()
    xb = next(iter(dl)).to(device)
    xrec, _, _ = model(xb)
    xrec = xrec.clamp(0, 1)

    def fc(x):
        # x: (C,H,W); select three bands and min-max normalize per-sample
        x3 = torch.stack([x[b] for b in bands], dim=0)
        x3 = (x3 - x3.amin()) / (x3.amax() - x3.amin() + 1e-8)
        return x3

    k = min(4, xb.size(0))
    tops = torch.stack([fc(xb[i].cpu()) for i in range(k)], dim=0)
    bots = torch.stack([fc(xrec[i].cpu()) for i in range(k)], dim=0)
    grid = torchvision.utils.make_grid(torch.cat([tops, bots], dim=0), nrow=k)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    torchvision.utils.save_image(grid, out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="/scratch/mkhorram/Soil/VNIR2/*.bip")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=1024)
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight")
    ap.add_argument("--outdir", default="/scratch/pkadukar/soil_proj/runs/vnir2_vae")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num-workers", type=int, default=2)
    # ✨ NEW: add vit_spatial as an option
    ap.add_argument("--model", choices=["conv", "vit", "vit_spatial"], default="conv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # data
    ds = HSIDataset(args.pattern)
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
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
        # here latent_dim is "per-token" latent size → total latent ≈ 256 * latent_dim
        model = ViTVAESpatial(in_ch=200, latent_dim=args.latent).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    os.makedirs(args.outdir, exist_ok=True)
    log_path = os.path.join(args.outdir, "train_log.txt")
    with open(log_path, "a") as f:
        f.write(
            f"Start: {time.asctime()} | epochs={args.epochs} "
            f"batch={args.batch} lr={args.lr} beta={args.beta} latent={args.latent}\n"
        )

    with open(log_path, "a") as f:
        f.write(f"model={args.model}\n")

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model, dl, opt, device,
            scaler if args.amp else None,
            beta=args.beta,
        )
        msg = f"Epoch {ep:03d} | loss={loss:.6f}"
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

        model.eval()
        with torch.no_grad():
            xb_val = next(iter(dl)).to(device)
            xrec_val, _, _ = model(xb_val)
            xrec_val = xrec_val.clamp(0, 1)
            m_psnr = psnr(xb_val, xrec_val)
            m_sam  = sam(xb_val, xrec_val)
        met = f"PSNR={m_psnr:.2f}dB SAM={m_sam:.4f}rad"
        print("Metrics:", met)
        with open(log_path, "a") as f:
            f.write("Metrics: " + met + "\n")

        # save ckpt + preview every epoch
        ckpt = os.path.join(args.outdir, f"ep{ep:03d}.pt")
        torch.save({"model": model.state_dict(), "ep": ep}, ckpt)

        prev = os.path.join(args.outdir, f"preview_ep{ep:03d}.png")
        save_preview(model, dl, device, prev)

    print("Done. Logs/ckpts at:", args.outdir)


if __name__ == "__main__":
    main()
