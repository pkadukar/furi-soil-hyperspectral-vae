import os, glob, numpy as np, torch, torch.nn.functional as F
import rasterio
from torch.utils.data import Dataset

def _is_dir(p):
    try: return os.path.isdir(p)
    except: return False

class HSIDataset(Dataset):
    def __init__(self, root_or_glob, target_hw=(128,128)):
        if _is_dir(root_or_glob):
            pattern = os.path.join(root_or_glob, "*.bip")
        else:
            pattern = root_or_glob
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No files match: {pattern}")
        self.target_h, self.target_w = target_hw

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with rasterio.open(p) as src:
            cube = src.read(out_dtype='float32')  # (C,H,W)
        x = torch.from_numpy(cube)  # (C,H,W)

        # Robust sanitize + normalize (no nanmin/nanmax)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mn = x.min()
        mx = x.max()
        rng = mx - mn
        x = (x - mn) / rng if rng > 0 else torch.zeros_like(x)

        # Force spatial size (128,128)
        C, H, W = x.shape
        th, tw = self.target_h, self.target_w
        pad_h = max(0, th - H)
        pad_w = max(0, tw - W)
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # left,right,top,bottom
        _, H2, W2 = x.shape
        if H2 > th or W2 > tw:
            sh = (H2 - th) // 2
            sw = (W2 - tw) // 2
            x = x[:, sh:sh+th, sw:sw+tw]

        return x.contiguous()
