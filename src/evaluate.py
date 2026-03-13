"""
evaluate.py — evaluation metrics and visualization for Quantum Drift Forecasting.

Functions
---------
forecast_metrics(y_true, y_pred)  → dict of MAE, RMSE, MAPE
classification_metrics(y_true, y_pred_logit) → dict of F1, precision, recall, AUC
plot_forecast(y_true, y_pred, title)
plot_anomaly_scores(scores, labels, title)
plot_attention_heatmap(attn_weights, title)
run_mc_dropout(model, x, n_passes) → (mean_pred, std_pred)
conformal_intervals(calibration_errors, alpha) → margin
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from typing import Tuple, Optional

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1e2130",
    "axes.edgecolor":   "#2d3148",
    "text.color":       "#e2e8f0",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "grid.color":       "#2d3148",
    "axes.grid":        True,
    "legend.facecolor": "#1e2130",
    "legend.edgecolor": "#2d3148",
})


# ── Forecast metrics ───────────────────────────────────────────────────────
def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and MAPE on arrays of shape (N,) or (N, H)."""
    mae  = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    mask = y_true != 0
    mape = float(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean()) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


# ── Classification metrics ─────────────────────────────────────────────────
def classification_metrics(
    y_true: np.ndarray,
    y_pred_logit: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, F1, and ROC-AUC for drift detection."""
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score
    )
    prob  = 1 / (1 + np.exp(-y_pred_logit))   # sigmoid
    pred  = (prob >= threshold).astype(int)
    y_int = y_true.astype(int)
    return {
        "Precision": float(precision_score(y_int, pred, zero_division=0)),
        "Recall":    float(recall_score(y_int, pred, zero_division=0)),
        "F1":        float(f1_score(y_int, pred, zero_division=0)),
        "ROC-AUC":   float(roc_auc_score(y_int, prob)) if len(np.unique(y_int)) > 1 else float("nan"),
    }


# ── Forecast visualization ─────────────────────────────────────────────────
def plot_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    title: str = "Drift Forecast vs Ground Truth",
    xlabel: str = "Time Step",
    ylabel: str = "T1 Coherence (µs, normalised)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true, label="Ground truth", color="#6366f1", linewidth=1.4, alpha=0.9)
    ax.plot(y_pred, label="Forecast",     color="#34d399", linewidth=1.4, linestyle="--")
    if y_lower is not None and y_upper is not None:
        ax.fill_between(range(len(y_pred)), y_lower, y_upper,
                        color="#34d399", alpha=0.15, label="90% interval")
    ax.set_title(title, color="#c7d2fe", fontsize=13, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


# ── Anomaly score visualization ────────────────────────────────────────────
def plot_anomaly_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Anomaly Scores — Reconstruction Error",
) -> plt.Figure:
    if threshold is None:
        threshold = float(np.percentile(scores, 90))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(scores, color="#818cf8", linewidth=1.2, label="Anomaly score")
    ax1.axhline(threshold, color="#f87171", linestyle="--", linewidth=1.2,
                label=f"Threshold ({threshold:.4f})")
    ax1.fill_between(range(len(scores)),
                     np.where(scores > threshold, scores, threshold),
                     threshold, color="#f87171", alpha=0.25)
    ax1.set_title(title, color="#c7d2fe", fontsize=13, pad=10)
    ax1.set_ylabel("Score")
    ax1.legend()

    ax2.imshow(labels[np.newaxis, :], aspect="auto", cmap="RdYlGn_r",
               vmin=0, vmax=1, extent=[0, len(labels), 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel("Time Step")
    ax2.set_title("Ground-truth drift labels", color="#94a3b8", fontsize=9, pad=4)
    fig.tight_layout()
    return fig


# ── Attention heatmap (for Transformer models) ─────────────────────────────
def plot_attention_heatmap(
    attn_weights: np.ndarray,
    title: str = "Self-Attention Weights",
) -> plt.Figure:
    """Plot an (seq_len × seq_len) attention weight matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_weights, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Attention weight")
    ax.set_title(title, color="#c7d2fe", fontsize=13, pad=10)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.tight_layout()
    return fig


# ── MC-Dropout uncertainty estimation ─────────────────────────────────────
def run_mc_dropout(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run ``n_passes`` forward passes with dropout active; return (mean, std).

    The model's Dropout layers must remain in training mode for this to
    produce stochastic outputs.  Call ``model.train()`` before passing here.
    """
    model.train()
    forecasts = []
    with torch.no_grad():
        for _ in range(n_passes):
            forecast, _ = model(x)
            forecasts.append(forecast.cpu().numpy())
    stacked = np.stack(forecasts, axis=0)   # (n_passes, batch, horizon)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ── Conformal prediction intervals ─────────────────────────────────────────
def conformal_margin(
    calibration_errors: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Return the (1-alpha) quantile of absolute calibration residuals.

    Under exchangeability, this provides a marginal coverage guarantee of
    at least (1-alpha) on new test points.
    """
    return float(np.quantile(calibration_errors, 1 - alpha))


# ── Comparative bar chart ──────────────────────────────────────────────────
def plot_model_comparison(
    results: dict,
    metric: str = "MAE",
    title: str = "Model Comparison",
) -> plt.Figure:
    """Bar chart comparing a metric across multiple models.

    Parameters
    ----------
    results : dict mapping model_name → metrics_dict (from forecast_metrics)
    metric  : key to plot
    """
    names  = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = ["#6366f1", "#34d399", "#f59e0b", "#f87171"][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=colors, width=0.55)
    ax.bar_label(bars, fmt="%.5f", padding=3, color="#e2e8f0", fontsize=9)
    ax.set_title(title, color="#c7d2fe", fontsize=13, pad=10)
    ax.set_ylabel(metric)
    fig.tight_layout()
    return fig
