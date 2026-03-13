"""
train.py — training loop and CLI for Quantum Drift Forecasting models.

Usage
-----
python -m src.train --model lstm --sequence-length 32 --horizon 8 \
                    --epochs 30 --model-path model_lstm.pt

The script:
  1. Loads / generates the hardware telemetry dataset.
  2. Builds windowed sequences for all qubits.
  3. Trains the selected model with a combined forecast + classification loss.
  4. Saves the best checkpoint (lowest validation MAE) to --model-path.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data import build_dataset, FEATURE_COLS
from src.models import build_model


# ── Dataset builder ────────────────────────────────────────────────────────
def to_tensors(X, y_f, y_l, device):
    return (
        torch.tensor(X,   dtype=torch.float32, device=device),
        torch.tensor(y_f, dtype=torch.float32, device=device),
        torch.tensor(y_l, dtype=torch.float32, device=device),
    )


# ── Combined loss ──────────────────────────────────────────────────────────
def compute_loss(
    forecast: torch.Tensor,
    drift_logit: torch.Tensor,
    y_forecast: torch.Tensor,
    y_label: torch.Tensor,
    alpha: float = 0.7,
) -> torch.Tensor:
    """Weighted sum of forecast MSE and binary cross-entropy drift classification."""
    mse  = nn.functional.mse_loss(forecast, y_forecast)
    bce  = nn.functional.binary_cross_entropy_with_logits(
        drift_logit.squeeze(-1), y_label
    )
    return alpha * mse + (1 - alpha) * bce


# ── Training loop ──────────────────────────────────────────────────────────
def train(
    model_name: str = "lstm",
    csv_path: str = "data/quantum_device_metrics.csv",
    seq_len: int = 32,
    horizon: int = 8,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_path: str = "model.pt",
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}  model={model_name}")

    # ── Data ──
    ds = build_dataset(csv_path=csv_path, seq_len=seq_len, horizon=horizon)
    input_dim = ds["train"][0].shape[-1]

    def make_loader(split, shuffle=False):
        Xt, yf, yl = to_tensors(*ds[split], device)
        return DataLoader(TensorDataset(Xt, yf, yl), batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val")

    # ── Model ──
    model = build_model(
        model_name,
        input_dim=input_dim,
        horizon=horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    history = {"train_loss": [], "val_mae": [], "val_bce": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for Xb, yf_b, yl_b in tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs}", leave=False):
            optimizer.zero_grad()
            forecast, drift_logit = model(Xb)
            loss = compute_loss(forecast, drift_logit, yf_b, yl_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(Xb)
        epoch_loss /= len(train_loader.dataset)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_mae_sum, val_bce_sum = 0.0, 0.0
        with torch.no_grad():
            for Xb, yf_b, yl_b in val_loader:
                forecast, drift_logit = model(Xb)
                val_mae_sum += (
                    (forecast - yf_b).abs().mean().item() * len(Xb)
                )
                val_bce_sum += nn.functional.binary_cross_entropy_with_logits(
                    drift_logit.squeeze(-1), yl_b
                ).item() * len(Xb)
        val_mae = val_mae_sum / len(val_loader.dataset)
        val_bce = val_bce_sum / len(val_loader.dataset)

        history["train_loss"].append(epoch_loss)
        history["val_mae"].append(val_mae)
        history["val_bce"].append(val_bce)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {"model_state": model.state_dict(), "config": {
                    "model_name": model_name, "input_dim": input_dim,
                    "horizon": horizon, "hidden_dim": hidden_dim,
                    "num_layers": num_layers, "dropout": dropout,
                }},
                model_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | train_loss={epoch_loss:.5f} "
                f"| val_mae={val_mae:.5f} | val_bce={val_bce:.5f}"
            )

    print(f"\n[train] Best val MAE: {best_val_mae:.5f}  →  saved to {model_path}")
    return history


# ── CLI ────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Train a quantum drift forecasting model")
    p.add_argument("--model",           default="lstm",
                   choices=["rnn", "lstm", "gru", "transformer"])
    p.add_argument("--csv",             default="data/quantum_device_metrics.csv")
    p.add_argument("--sequence-length", type=int,   default=32)
    p.add_argument("--horizon",         type=int,   default=8)
    p.add_argument("--hidden-dim",      type=int,   default=128)
    p.add_argument("--num-layers",      type=int,   default=2)
    p.add_argument("--dropout",         type=float, default=0.2)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--model-path",      default="model.pt")
    p.add_argument("--seed",            type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        model_name=args.model,
        csv_path=args.csv,
        seq_len=args.sequence_length,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_path=args.model_path,
        seed=args.seed,
    )
