"""
models.py — neural network architectures for Quantum Drift Forecasting.

Implements:
  - VanillaRNN   : single-layer Elman RNN
  - LSTMForecaster: multi-layer LSTM with dropout
  - GRUForecaster : multi-layer GRU with dropout
  - TransformerForecaster: encoder-only Transformer with sinusoidal positional encoding
  - AnomalyDetector: Transformer-based reconstruction model for unsupervised anomaly scoring

All models share a common forward(x) → (forecast, drift_logit) interface:
  - forecast   : tensor of shape (batch, horizon) — next-step T1 predictions
  - drift_logit: tensor of shape (batch, 1)        — raw logit for drift classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ── Shared output head ─────────────────────────────────────────────────────
class _ForecastHead(nn.Module):
    """Linear projection from hidden state to (horizon forecast, drift logit)."""

    def __init__(self, hidden_dim: int, horizon: int) -> None:
        super().__init__()
        self.forecast_proj = nn.Linear(hidden_dim, horizon)
        self.drift_proj    = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forecast_proj(h), self.drift_proj(h)


# ── Vanilla RNN ────────────────────────────────────────────────────────────
class VanillaRNN(nn.Module):
    """Single-layer Elman RNN baseline.

    Parameters
    ----------
    input_dim   : number of input features per time step
    hidden_dim  : RNN hidden state dimension
    horizon     : number of future steps to forecast
    dropout     : dropout probability (applied after RNN)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        horizon: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn  = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = _ForecastHead(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        _, h_n = self.rnn(x)          # h_n: (1, batch, hidden_dim)
        h = self.drop(h_n.squeeze(0)) # (batch, hidden_dim)
        return self.head(h)


# ── LSTM Forecaster ────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """Multi-layer LSTM with optional MC-Dropout for uncertainty estimation.

    Parameters
    ----------
    input_dim   : number of input features per time step
    hidden_dim  : LSTM hidden / cell state dimension
    num_layers  : number of stacked LSTM layers
    horizon     : forecast horizon (steps ahead)
    dropout     : dropout probability between LSTM layers and before head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = _ForecastHead(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, batch, hidden_dim)
        h = self.drop(h_n[-1])        # last layer, (batch, hidden_dim)
        return self.head(h)


# ── GRU Forecaster ─────────────────────────────────────────────────────────
class GRUForecaster(nn.Module):
    """Multi-layer GRU — computationally lighter than LSTM.

    Parameters
    ----------
    input_dim   : number of input features per time step
    hidden_dim  : GRU hidden state dimension
    num_layers  : number of stacked GRU layers
    horizon     : forecast horizon (steps ahead)
    dropout     : dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        horizon: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru  = nn.GRU(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = _ForecastHead(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden_dim)
        h = self.drop(h_n[-1])
        return self.head(h)


# ── Sinusoidal Positional Encoding ─────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


# ── Transformer Forecaster ─────────────────────────────────────────────────
class TransformerForecaster(nn.Module):
    """Encoder-only Transformer for multi-step drift forecasting.

    The input sequence is projected to ``d_model`` dimensions, augmented with
    sinusoidal positional encodings, and processed by a stack of multi-head
    self-attention layers.  The final [CLS]-like token (last time step) is
    passed through a shared output head.

    Parameters
    ----------
    input_dim   : number of input features per time step
    d_model     : Transformer embedding dimension (must be divisible by nhead)
    nhead       : number of attention heads
    num_layers  : number of Transformer encoder layers
    dim_ff      : feedforward sublayer hidden dimension
    horizon     : forecast horizon
    dropout     : dropout probability throughout
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        horizon: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.head     = _ForecastHead(d_model, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        h = self.pos_enc(self.input_proj(x))   # (batch, seq_len, d_model)
        h = self.norm(self.encoder(h))          # (batch, seq_len, d_model)
        cls = h[:, -1, :]                       # use last token as summary
        return self.head(cls)


# ── Transformer Anomaly Detector (reconstruction-based) ───────────────────
class AnomalyDetector(nn.Module):
    """Encoder-Decoder Transformer that reconstructs the input sequence.

    Anomaly scores are computed as the per-time-step mean squared
    reconstruction error.  High-error windows indicate distributional shift
    relative to the training regime.

    Parameters
    ----------
    input_dim  : number of input features
    d_model    : Transformer embedding dimension
    nhead      : attention heads
    num_layers : number of encoder and decoder layers
    dim_ff     : feedforward sublayer dimension
    dropout    : dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, input_dim)
        self.pos_enc     = SinusoidalPositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.norm    = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return reconstructed sequence of shape (batch, seq_len, input_dim)."""
        h  = self.pos_enc(self.input_proj(x))
        mem = self.norm(self.encoder(h))
        tgt = self.pos_enc(self.input_proj(x))  # teacher-force with input
        out = self.decoder(tgt, mem)
        return self.output_proj(out)

    @torch.no_grad()
    def anomaly_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample anomaly score (mean MSE across time and features)."""
        rec = self(x)
        return ((x - rec) ** 2).mean(dim=(1, 2))


# ── Factory ────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "rnn":         VanillaRNN,
    "lstm":        LSTMForecaster,
    "gru":         GRUForecaster,
    "transformer": TransformerForecaster,
}


def build_model(
    name: str,
    input_dim: int,
    horizon: int = 8,
    **kwargs,
) -> nn.Module:
    """Instantiate a forecasting model by name.

    Parameters
    ----------
    name      : one of 'rnn', 'lstm', 'gru', 'transformer'
    input_dim : number of input features
    horizon   : forecast horizon in steps
    **kwargs  : forwarded to the model constructor
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](input_dim=input_dim, horizon=horizon, **kwargs)
