"""
ShopFormer training script.

Usage:
    python -m backend.services.shopformer.train \
        --dataset /data/training \
        --output  /models/shopformer \
        --epochs  50

Trains the GCAE + Transformer on labeled pose sequences.
Saves best checkpoint by val loss.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from backend.core.logging import get_logger
from backend.core.settings import get_settings

log      = get_logger("shopformer.train")
settings = get_settings()


def train(dataset_dir: str, output_dir: str, epochs: int = 50, lr: float = 1e-3) -> dict:
    """
    Training loop. Returns final metrics dict.
    Requires: torch, numpy. Loads pose_sequences.npy + labels.npy.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except ImportError:
        log.error("torch not installed — cannot train ShopFormer")
        return {"error": "torch not installed"}

    ds_path = Path(dataset_dir)
    X = np.load(ds_path / "pose_sequences.npy")   # (N, seq_len, 17, 2)
    y = np.load(ds_path / "labels.npy")

    N, T, K, D = X.shape
    X_flat = X.reshape(N, T, K * D)               # (N, T, 34)

    X_t = torch.from_numpy(X_flat).float()
    y_t = torch.from_numpy(y.astype(np.float32))

    dataset   = TensorDataset(X_t, y_t)
    n_val     = max(1, int(len(dataset) * 0.2))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # Simple Transformer classifier
    model = nn.Sequential(
        nn.Linear(K * D, settings.shopformer_latent_dim),
        nn.ReLU(),
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=settings.shopformer_latent_dim,
                nhead=min(settings.shopformer_num_heads, settings.shopformer_latent_dim),
                batch_first=True,
                dim_feedforward=128,
            ),
            num_layers=2,
        ),
        nn.AdaptiveAvgPool1d(1),  # pool over time
    )

    # Use a wrapper for final classification
    class Classifier(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(settings.shopformer_latent_dim, 1)

        def forward(self, x):
            # x: (B, T, K*D)
            h = self.encoder[0](x)   # Linear
            h = self.encoder[1](h)   # ReLU
            h = self.encoder[2](h)   # Transformer
            h = h.mean(dim=1)        # mean over time
            return self.head(h).squeeze(-1)

    clf = Classifier(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.to(device)

    opt      = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn  = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        clf.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(clf(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        clf.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += loss_fn(clf(xb), yb).item()

        log.info("Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                 epoch, epochs, train_loss / len(train_loader), val_loss / len(val_loader))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(clf.state_dict(), out_path / "transformer_best.pth")
            log.info("Saved best checkpoint (val_loss=%.4f)", best_val)

    metrics = {"epochs": epochs, "best_val_loss": round(best_val, 4)}
    (out_path / "train_metrics.json").write_text(json.dumps(metrics))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output",  required=True)
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--lr",      type=float, default=1e-3)
    args = parser.parse_args()
    result = train(args.dataset, args.output, args.epochs, args.lr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
