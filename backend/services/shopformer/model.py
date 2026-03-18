"""
Netra AI — ShopFormer inference wrapper.

GCAE tokenizer + Transformer encoder-decoder for pose sequence anomaly detection.
Ported from Shopformer-main/models/ and adapted for real-time selective inference.

Two-stage design (mirrors Shopformer training):
  Stage 1: GCAE encodes skeleton sequence → tokens
  Stage 2: Transformer reconstructs tokens → reconstruction error = anomaly score

This module wraps both stages for inference-only use.
Trained checkpoints required at:
  settings.shopformer_gcae_path
  settings.shopformer_model_path
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from backend.core.settings import get_settings
from backend.core.logging import get_logger

log      = get_logger("shopformer.model")
settings = get_settings()


# ---------------------------------------------------------------------------
# GCAE — Graph Convolutional Autoencoder (skeleton tokenizer)
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """Single graph convolution: aggregates features over skeleton topology."""

    def __init__(self, in_dim: int, out_dim: int, num_nodes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Learnable adjacency (dense, not sparse — acceptable for 17 nodes)
        self.adj = nn.Parameter(torch.eye(num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, in_dim)
        adj_norm = torch.softmax(self.adj, dim=-1)
        x = torch.bmm(adj_norm.unsqueeze(0).expand(x.size(0), -1, -1), x)
        return torch.relu(self.linear(x))


class GCAEEncoder(nn.Module):
    def __init__(self, num_kp: int, latent_dim: int) -> None:
        super().__init__()
        self.gcn1 = GraphConvLayer(2,          32, num_kp)   # xy coords
        self.gcn2 = GraphConvLayer(32,         latent_dim, num_kp)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, num_kp, 2)
        batch, seq, kp, c = x.shape
        x = x.view(batch * seq, kp, c)
        x = self.gcn1(x)
        x = self.gcn2(x)            # (batch*seq, kp, latent_dim)
        x = x.permute(0, 2, 1)     # (batch*seq, latent_dim, kp)
        x = self.pool(x).squeeze(-1)  # (batch*seq, latent_dim)
        x = x.view(batch, seq, -1)  # (batch, seq, latent_dim)
        return x


class GCAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_kp: int) -> None:
        super().__init__()
        self.expand = nn.Linear(latent_dim, num_kp * 32)
        self.gcn    = GraphConvLayer(32, 2, num_kp)
        self.num_kp = num_kp

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, seq, latent_dim)
        batch, seq, d = z.shape
        z = z.view(batch * seq, d)
        x = self.expand(z).view(batch * seq, self.num_kp, 32)
        x = self.gcn(x)           # (batch*seq, num_kp, 2)
        x = x.view(batch, seq, self.num_kp, 2)
        return x


class GCAE(nn.Module):
    """Full GCAE — used for training. Encoder used alone for tokenisation."""

    def __init__(self, num_kp: int = 17, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = GCAEEncoder(num_kp, latent_dim)
        self.decoder = GCAEDecoder(latent_dim, num_kp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent tokens without decoding: (batch, seq, latent_dim)."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Transformer anomaly detector
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ShopFormerTransformer(nn.Module):
    """
    Transformer encoder-decoder that reconstructs GCAE tokens.
    High reconstruction error → anomalous (shoplifting) behavior.
    """

    def __init__(
        self,
        token_dim:   int = 64,
        num_heads:   int = 8,
        num_layers:  int = 4,
        ff_dim:      int = 256,
        dropout:     float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.pe = PositionalEncoding(token_dim, max_seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (batch, seq, token_dim) → reconstructed tokens same shape."""
        x   = self.pe(tokens)
        mem = self.encoder(x)
        out = self.decoder(x, mem)
        return out

    def reconstruction_error(self, tokens: torch.Tensor) -> torch.Tensor:
        """Per-sequence MSE between input and reconstruction: (batch,)"""
        with torch.no_grad():
            recon = self(tokens)
        return ((tokens - recon) ** 2).mean(dim=[1, 2])


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class ShopFormerInference:
    """
    Wraps GCAE tokenizer + Transformer for real-time use.
    Loads checkpoints; normalises anomaly scores to [0, 1] via sigmoid.
    """

    def __init__(self) -> None:
        self.device     = torch.device(settings.shopformer_device)
        self.num_kp     = settings.shopformer_num_kp
        self.latent_dim = settings.shopformer_latent_dim
        self._loaded    = False

        self.gcae_encoder: Optional[GCAEEncoder] = None
        self.transformer:  Optional[ShopFormerTransformer] = None

    def load(self) -> None:
        """Load weights from disk. Call once at service startup."""
        import os

        gcae_path = settings.shopformer_gcae_path
        tfm_path  = settings.shopformer_model_path

        gcae = GCAE(num_kp=self.num_kp, latent_dim=self.latent_dim)
        if os.path.exists(gcae_path):
            state = torch.load(gcae_path, map_location=self.device)
            gcae.load_state_dict(state, strict=False)
            log.info(f"Loaded GCAE from {gcae_path}")
        else:
            log.warning(f"GCAE checkpoint not found at {gcae_path} — using random weights (for dev only)")

        gcae.eval()
        self.gcae_encoder = gcae.encoder.to(self.device)

        transformer = ShopFormerTransformer(
            token_dim  = self.latent_dim,
            num_heads  = min(settings.shopformer_num_heads, self.latent_dim // 8),
            num_layers = settings.shopformer_num_layers,
        )
        if os.path.exists(tfm_path):
            state = torch.load(tfm_path, map_location=self.device)
            transformer.load_state_dict(state, strict=False)
            log.info(f"Loaded Transformer from {tfm_path}")
        else:
            log.warning(f"Transformer checkpoint not found at {tfm_path} — using random weights (for dev only)")

        transformer.eval()
        self.transformer = transformer.to(self.device)
        self._loaded     = True
        log.info("ShopFormer ready")

    def score(self, pose_sequence: list[list[list[float]]]) -> tuple[float, float, list[float]]:
        """
        Args:
            pose_sequence: list of frames, each frame is list of [x, y] per keypoint
                           shape: (seq_len, num_kp, 2)
        Returns:
            (anomaly_score: 0-1, reconstruction_error: raw, embedding: list[float])
        """
        if not self._loaded:
            self.load()

        seq_len = len(pose_sequence)
        if seq_len < 4:
            return 0.0, 0.0, []

        # Normalise + tensorise
        arr = np.array(pose_sequence, dtype=np.float32)   # (seq, kp, 2)
        arr = _normalise_skeleton(arr)

        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)   # (1, seq, kp, 2)

        with torch.no_grad():
            tokens = self.gcae_encoder(x)                         # (1, seq, latent_dim)
            error  = self.transformer.reconstruction_error(tokens) # (1,)

        raw_error     = float(error[0].cpu())
        anomaly_score = float(torch.sigmoid(torch.tensor(raw_error - 1.0)))
        embedding     = tokens[0].mean(dim=0).cpu().tolist()       # mean-pool over seq

        return anomaly_score, raw_error, embedding


def _normalise_skeleton(arr: np.ndarray) -> np.ndarray:
    """
    Centre skeleton on hip midpoint and scale by torso height.
    Makes scores camera-resolution-invariant.
    Matches ShopFormer training preprocessing.
    """
    # arr: (seq, kp, 2) — COCO order; kp 11,12 = hips, 5,6 = shoulders
    KP_L_HIP = 11; KP_R_HIP = 12
    KP_L_SH  = 5;  KP_R_SH  = 6

    # Per-frame normalisation
    out = arr.copy()
    for t in range(arr.shape[0]):
        hip_cx  = (arr[t, KP_L_HIP, 0] + arr[t, KP_R_HIP, 0]) / 2
        hip_cy  = (arr[t, KP_L_HIP, 1] + arr[t, KP_R_HIP, 1]) / 2
        sh_cy   = (arr[t, KP_L_SH, 1]  + arr[t, KP_R_SH, 1])  / 2

        torso   = max(abs(hip_cy - sh_cy), 1e-6)
        out[t, :, 0] = (arr[t, :, 0] - hip_cx) / torso
        out[t, :, 1] = (arr[t, :, 1] - hip_cy) / torso

    return out
