import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEConv(nn.Module):
    def __init__(self, in_ch=200, latent_dim=1024):
        super().__init__()
        # ---- Encoder: 128 -> 64 -> 32 -> 16 ----
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.GroupNorm(16, 128), nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),    # 64x64  -> 32x32
            nn.GroupNorm(32, 256), nn.SiLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),    # 32x32  -> 16x16
            nn.GroupNorm(32, 512), nn.SiLU(),
        )
        self.spatial = 16  # after 3 downsamples from 128
        enc_flat = 512 * self.spatial * self.spatial

        # latent heads
        self.fc_mu = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)

        # ---- Decoder: 16 -> 32 -> 64 -> 128 ----
        self.fc_dec = nn.Linear(latent_dim, enc_flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 16->32
            nn.GroupNorm(32, 256), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 32->64
            nn.GroupNorm(16, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, in_ch, kernel_size=4, stride=2, padding=1),# 64->128
            nn.Sigmoid(),  # inputs scaled to [0,1] -> reconstruct in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, self.spatial, self.spatial)
        xh = self.dec(h)
        return xh

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xrec = self.decode(z)
        return xrec, mu, logvar
