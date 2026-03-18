"""
Export trained ShopFormer to ONNX for fast CPU inference.

Usage:
    python -m backend.services.shopformer.export \
        --checkpoint /models/shopformer/transformer_best.pth \
        --output     /models/shopformer/model.onnx \
        --seq-len    24
"""
from __future__ import annotations
import argparse
from pathlib import Path
from backend.core.logging import get_logger
from backend.core.settings import get_settings

log = get_logger("shopformer.export")
settings = get_settings()


def export(checkpoint: str, output: str, seq_len: int = 24) -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        log.error("torch not installed")
        return

    K = settings.shopformer_num_kp   # 17
    D = 2

    class _Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(K * D, settings.shopformer_latent_dim)
            self.relu = nn.ReLU()
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=settings.shopformer_latent_dim,
                    nhead=min(settings.shopformer_num_heads, settings.shopformer_latent_dim),
                    batch_first=True, dim_feedforward=128,
                ),
                num_layers=2,
            )
            self.head = nn.Linear(settings.shopformer_latent_dim, 1)

        def forward(self, x):
            h = self.relu(self.proj(x))
            h = self.encoder(h).mean(dim=1)
            return torch.sigmoid(self.head(h)).squeeze(-1)

    model = _Classifier()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, seq_len, K * D)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["pose_sequence"],
        output_names=["anomaly_score"],
        dynamic_axes={"pose_sequence": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    log.info("Exported ONNX model to %s", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--seq-len",    type=int, default=24)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.seq_len)


if __name__ == "__main__":
    main()
