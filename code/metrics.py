import os
import re
import numpy as np

# Change this to point to whichever run you want to summarize
run = "runs/vnir2_vit_spatial_main_metrics"

log_path = os.path.join(run, "train_log.txt")
tsv_path = os.path.join(run, "metrics.tsv")

print(f"[INFO] Reading log from: {log_path}")

if not os.path.exists(log_path):
    raise FileNotFoundError(f"train_log.txt not found at {log_path}")

epochs = []
psnrs = []
sams = []

current_epoch = None

with open(log_path, "r") as f:
    for line in f:
        line = line.strip()

        # Look for "Epoch 005 | loss=..."
        m_ep = re.match(r"Epoch\s+(\d+)\s*\|", line)
        if m_ep:
            current_epoch = int(m_ep.group(1))
            continue

        # Look for "Metrics: PSNR=24.80dB SAM=0.0490rad"
        if line.startswith("Metrics:"):
            m_psnr = re.search(r"PSNR=([\d\.]+)dB", line)
            m_sam = re.search(r"SAM=([\d\.]+)rad", line)
            if m_psnr and m_sam and current_epoch is not None:
                psnr_val = float(m_psnr.group(1))
                sam_val = float(m_sam.group(1))

                epochs.append(current_epoch)
                psnrs.append(psnr_val)
                sams.append(sam_val)

                current_epoch = None  # reset until next "Epoch ..." line

if not epochs:
    raise RuntimeError("No epoch/metric lines were parsed from train_log.txt.")

data = np.column_stack([epochs, psnrs, sams])
np.savetxt(tsv_path, data, fmt=["%d", "%.6f", "%.6f"])

print(f"[INFO] Wrote {tsv_path} with {len(epochs)} rows.")

