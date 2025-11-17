import argparse, torch, torchvision
from torch.utils.data import DataLoader
from hsi_dataset import HSITiles

def false_color(x):
    # x: (C,H,W) in [0,1]; pick 3 bands for a false-color preview
    bands = [10, 50, 120]          # arbitrary but usually distinct
    rgb = torch.stack([x[b] for b in bands], dim=0)  # (3,H,W)
    rgb = (rgb - rgb.amin()) / (rgb.amax() - rgb.amin() + 1e-8)
    return rgb

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="/scratch/mkhorram/Soil/VNIR2/*.bip")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    ds = HSITiles(args.pattern)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    xb = next(iter(dl))  # (B,200,128,128)
    print("Batch shape:", tuple(xb.shape), "min/max:", float(xb.min()), float(xb.max()))

    # save a 2x2 grid of false-color previews
    imgs = torch.stack([false_color(xb[i]) for i in range(min(4, xb.size(0)))], dim=0)
    grid = torchvision.utils.make_grid(imgs, nrow=2)  # (3,H*,W*)
    out = "/scratch/pkadukar/soil_proj/runs/vnir2_quicklook.png"
    torchvision.utils.save_image(grid, out)
    print("Saved preview ->", out)
