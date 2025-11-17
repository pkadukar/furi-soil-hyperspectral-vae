import torch
import torch.nn as nn

# ---- Patchify / Unpatchify ----
def patchify(x, patch=8):
    # x: (B, C=200, H=128, W=128) -> (B, N=256, P*C)
    B, C, H, W = x.shape
    assert H % patch == 0 and W % patch == 0
    ph = pw = patch
    n_h, n_w = H // ph, W // pw
    x = x.unfold(2, ph, ph).unfold(3, pw, pw)           # (B,C,n_h,n_w,ph,pw)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()        # (B,n_h,n_w,C,ph,pw)
    x = x.view(B, n_h * n_w, C * ph * pw)               # (B,256, 200*8*8)
    return x

def unpatchify(tokens, patch=8, C=200, H=128, W=128):
    # tokens: (B, N=256, P*C) -> (B, C, H, W)
    B, N, D = tokens.shape
    n_h, n_w = H // patch, W // patch
    ph = pw = patch
    x = tokens.view(B, n_h, n_w, C, ph, pw)             # (B,n_h,n_w,C,ph,pw)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()        # (B,C,n_h,ph,n_w,pw)
    x = x.view(B, C, H, W)
    return x

# ---- Minimal ViT encoder (we add VAE/decoder next) ----
class ViTEncoder(nn.Module):
    def __init__(self, patch=8, in_ch=200, d_model=256, nhead=8, depth=4):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(in_ch * patch * patch, d_model)   # 200*8*8 -> 256
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))   # 16*16 tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x):
        # x: (B,200,128,128) -> tokens (B,256,256)
        tok = patchify(x, self.patch)
        tok = self.proj(tok) + self.pos
        h = self.encoder(tok)                                   # (B,256,256)
        z_tokens = h.mean(dim=1)                                # (B,256) compact summary
        return h, z_tokens

# ---- Transformer Decoder + VAE wrapper ----
class ViTDecoder(nn.Module):
    """
    We decode by expanding the latent vector z into a sequence of
    256 token embeddings (matching 16x16 patches), process with a
    small Transformer, then project tokens back to patch pixels and
    unpatchify to (B, C, H, W).
    """
    def __init__(
        self,
        patch=8,
        out_ch=200,
        d_model=256,
        nhead=8,
        depth=4,
        H=128,
        W=128,
    ):
        super().__init__()
        self.patch = patch
        self.H, self.W = H, W
        self.pos = nn.Parameter(torch.zeros(1, 256, d_model))       # 16x16 tokens
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=depth)
        self.proj_out = nn.Linear(d_model, out_ch * patch * patch)  # -> pixels per patch
        self.act = nn.Sigmoid()

    def forward(self, token_seq):
        # token_seq: (B, 256, d_model)
        h = self.decoder(token_seq + self.pos)                      # (B,256,d_model)
        pix = self.proj_out(h)                                      # (B,256, C*P*P)
        x = unpatchify(pix, patch=self.patch, C=200, H=self.H, W=self.W)
        return self.act(x)


class ViTVAE(nn.Module):
    """
    Original ViT-VAE with a *single* latent vector of size `latent_dim`.
    """
    def __init__(
        self,
        in_ch=200,
        patch=8,
        d_model=256,
        nhead=8,
        depth_enc=4,
        depth_dec=4,
        latent_dim=1024,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            patch=patch,
            in_ch=in_ch,
            d_model=d_model,
            nhead=nhead,
            depth=depth_enc,
        )

        # VAE heads from pooled token representation (d_model dims)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Map latent -> 256 tokens for the decoder
        self.fc_tokens = nn.Linear(latent_dim, 256 * d_model)
        self.decoder = ViTDecoder(
            patch=patch,
            out_ch=in_ch,
            d_model=d_model,
            nhead=nhead,
            depth=depth_dec,
            H=128,
            W=128,
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode to tokens and pooled summary
        tokens, pooled = self.encoder(x)               # tokens: (B,256,d_model), pooled: (B,d_model)
        mu = self.fc_mu(pooled)
        logvar = self.fc_logvar(pooled)
        z = self.reparam(mu, logvar)                   # (B, latent_dim)

        # Decode
        dec_tokens = self.fc_tokens(z).view(z.size(0), 256, -1)   # (B,256,d_model)
        xrec = self.decoder(dec_tokens)                           # (B,200,128,128)
        return xrec, mu, logvar


class ViTVAESpatial(nn.Module):
    """
    ViT-based VAE with a *spatial* latent bottleneck.

    - Encoder:
        Uses ViTEncoder to get patch tokens h: (B, N=256, d_model).

    - Latent:
        Per-token mu/logvar -> z_tok: (B, N, latent_dim), where
        N = 16 * 16 is the patch grid. This is your spatial latent map.

    - Decoder:
        Project z_tok back to d_model and decode with the existing
        ViTDecoder (tokens -> patches -> full image).

    - API:
        forward(x) returns (xrec, mu_flat, logvar_flat) so your existing
        KL function in train_vnir_vae.py can be reused unchanged.
    """
    def __init__(
        self,
        in_ch: int = 200,
        patch: int = 8,
        d_model: int = 256,
        nhead: int = 8,
        depth_enc: int = 4,
        depth_dec: int = 4,
        latent_dim: int = 64,   # per-token latent size (can be overridden)
        H: int = 128,
        W: int = 128,
    ):
        super().__init__()

        # ViT encoder gives patch tokens
        self.encoder = ViTEncoder(
            patch=patch,
            in_ch=in_ch,
            d_model=d_model,
            nhead=nhead,
            depth=depth_enc,
        )

        # Per-token latent heads: (B, N, d_model) -> (B, N, latent_dim)
        self.fc_mu_tok = nn.Linear(d_model, latent_dim)
        self.fc_logvar_tok = nn.Linear(d_model, latent_dim)

        # Map latent tokens back to d_model for the decoder
        self.fc_dec_tok = nn.Linear(latent_dim, d_model)

        # Reuse transformer decoder + unpatchify
        self.decoder = ViTDecoder(
            patch=patch,
            out_ch=in_ch,
            d_model=d_model,
            nhead=nhead,
            depth=depth_dec,
            H=H,
            W=W,
        )

    def reparam(self, mu, logvar):
        """
        Reparameterization per token.

        mu, logvar: (B, N, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # tokens: (B, 256, d_model), pooled: (B, d_model) [pooled unused]
        tokens, _ = self.encoder(x)

        # Per-token mu/logvar -> spatial latent
        mu_tok = self.fc_mu_tok(tokens)          # (B, N, latent_dim)
        logvar_tok = self.fc_logvar_tok(tokens)  # (B, N, latent_dim)

        z_tok = self.reparam(mu_tok, logvar_tok) # (B, N, latent_dim)

        # Decode: latent tokens -> d_model tokens -> image
        dec_tokens = self.fc_dec_tok(z_tok)      # (B, N, d_model)
        xrec = self.decoder(dec_tokens)          # (B, 200, 128, 128)

        # Flatten mu/logvar so KL loss code still works
        B, N, D = mu_tok.shape
        mu_flat = mu_tok.view(B, N * D)
        logvar_flat = logvar_tok.view(B, N * D)

        return xrec, mu_flat, logvar_flat
