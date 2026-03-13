"""
data.py — data generation, loading, and preprocessing for Quantum Drift Forecasting.

Provides:
  - Synthetic multi-qubit hardware telemetry generation
  - CSV loading and per-qubit time-series extraction
  - Sliding-window sequence construction for supervised training
  - Train/validation/test splitting with no data leakage
"""

import math
import random
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


# ── Constants ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "T1_us",
    "T2_us",
    "gate_fidelity_1q",
    "gate_fidelity_2q",
    "readout_error",
    "gate_error_per_clifford",
    "cross_resonance_phase_rad",
]
TARGET_COL = "T1_us"       # default single-step forecast target
LABEL_COL  = "drift_label"


# ── Synthetic data generation ───────────────────────────────────────────────
def generate_synthetic_dataset(
    n_qubits: int = 5,
    n_steps: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic multi-qubit hardware telemetry dataset.

    Each row corresponds to a (timestamp, qubit_id) pair and contains
    simulated values for T1, T2, gate fidelities, readout error, gate error,
    and a cross-resonance phase parameter.  A binary ``drift_label`` column
    marks time steps where at least one metric has crossed an operational
    threshold.
    """
    rng = random.Random(seed)
    rows: List[dict] = []

    for step in range(n_steps):
        t = step * 0.5  # time in hours
        for q in range(n_qubits):
            # T1 relaxation time (µs) — slow sinusoidal drift + linear decay + noise
            t1_base = 80 + 20 * math.sin(2 * math.pi * t / 48 + q * 0.8)
            t1 = max(10.0, t1_base - 0.05 * t + rng.gauss(0, 2))

            # T2 coherence time (µs)
            t2 = max(5.0, 0.9 * t1 * rng.gauss(1.0, 0.04))

            # 1-qubit gate fidelity
            gf1 = min(1.0, max(0.90, 0.9990 - 0.00002 * t + rng.gauss(0, 0.0003)))

            # 2-qubit gate fidelity
            gf2 = min(1.0, max(0.85, 0.9850 - 0.00004 * t + rng.gauss(0, 0.0008)))

            # Readout assignment error
            ro_err = min(0.15, max(0.001, 0.012 + 0.00003 * t + rng.gauss(0, 0.001)))

            # Gate error per Clifford
            gate_err = min(0.01, max(0.0001, 0.0005 + 0.000005 * t + rng.gauss(0, 0.00005)))

            # Cross-resonance phase (rad) — tunable coupling parameter drift
            cr_phase = (
                1.5708
                + 0.001 * t
                + 0.05 * math.sin(2 * math.pi * t / 24)
                + rng.gauss(0, 0.01)
            )

            drift_label = int(t1 < 50 or gf1 < 0.998)

            rows.append(
                {
                    "timestamp_hr": round(t, 2),
                    "qubit_id": q,
                    "T1_us": round(t1, 4),
                    "T2_us": round(t2, 4),
                    "gate_fidelity_1q": round(gf1, 6),
                    "gate_fidelity_2q": round(gf2, 6),
                    "readout_error": round(ro_err, 6),
                    "gate_error_per_clifford": round(gate_err, 7),
                    "cross_resonance_phase_rad": round(cr_phase, 6),
                    "drift_label": drift_label,
                }
            )

    return pd.DataFrame(rows)


def load_or_generate(csv_path: str = "data/quantum_device_metrics.csv") -> pd.DataFrame:
    """Load the hardware telemetry CSV, or generate and save it if absent."""
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = generate_synthetic_dataset()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


# ── Per-qubit time series extraction ──────────────────────────────────────
def extract_qubit_series(
    df: pd.DataFrame,
    qubit_id: int,
    features: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays for a single qubit, sorted by timestamp.

    ``X`` has shape (T, n_features); ``y`` is the scalar ``drift_label``
    sequence of length T.
    """
    features = features or FEATURE_COLS
    sub = df[df["qubit_id"] == qubit_id].sort_values("timestamp_hr")
    X = sub[features].values.astype(np.float32)
    y = sub[LABEL_COL].values.astype(np.float32)
    return X, y


# ── Min-max normalisation ──────────────────────────────────────────────────
def normalize(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit min-max scaler on training data; apply to val and test."""
    x_min = X_train.min(axis=0, keepdims=True)
    x_max = X_train.max(axis=0, keepdims=True)
    rng = np.where(x_max - x_min == 0, 1.0, x_max - x_min)

    def scale(X):
        return (X - x_min) / rng

    return scale(X_train), scale(X_val), scale(X_test), x_min.squeeze(), x_max.squeeze()


# ── Sliding-window sequence builder ────────────────────────────────────────
def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 32,
    horizon: int = 8,
    target_col_idx: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build supervised (input_seq, target_seq, label_seq) triples.

    Returns
    -------
    X_seq : ndarray of shape (N, seq_len, n_features)
    y_seq : ndarray of shape (N, horizon)        — next-step forecasting targets
    lbl   : ndarray of shape (N,)                — drift label at the forecast horizon
    """
    N = len(X) - seq_len - horizon + 1
    if N <= 0:
        raise ValueError(
            f"Sequence too short: len(X)={len(X)} < seq_len+horizon={seq_len + horizon}"
        )
    X_seq = np.stack([X[i : i + seq_len] for i in range(N)]).astype(np.float32)
    y_seq = np.stack(
        [X[i + seq_len : i + seq_len + horizon, target_col_idx] for i in range(N)]
    ).astype(np.float32)
    lbl = np.array(
        [y[i + seq_len + horizon - 1] for i in range(N)], dtype=np.float32
    )
    return X_seq, y_seq, lbl


# ── Train / val / test split ───────────────────────────────────────────────
def temporal_split(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    lbl: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple:
    """Chronological (non-shuffled) 70/15/15 split."""
    N = len(X_seq)
    n_train = int(N * train_frac)
    n_val   = int(N * val_frac)

    return (
        X_seq[:n_train],       y_seq[:n_train],       lbl[:n_train],
        X_seq[n_train:n_train + n_val], y_seq[n_train:n_train + n_val], lbl[n_train:n_train + n_val],
        X_seq[n_train + n_val:], y_seq[n_train + n_val:], lbl[n_train + n_val:],
    )


# ── All-qubit combined dataset builder ────────────────────────────────────
def build_dataset(
    csv_path: str = "data/quantum_device_metrics.csv",
    seq_len: int = 32,
    horizon: int = 8,
    features: Optional[List[str]] = None,
) -> dict:
    """Full pipeline: load → per-qubit extract → sequence → normalise → split.

    Returns a dict with keys 'train', 'val', 'test', each containing
    (X, y_forecast, y_label) arrays, plus 'x_min' and 'x_max' scalers.
    """
    features = features or FEATURE_COLS
    df = load_or_generate(csv_path)
    n_qubits = df["qubit_id"].nunique()

    all_splits = {"train": [], "val": [], "test": []}
    x_min_all, x_max_all = None, None

    for q in range(n_qubits):
        X, y = extract_qubit_series(df, q, features)
        X_seq, y_seq, lbl = make_sequences(X, y, seq_len=seq_len, horizon=horizon)
        parts = temporal_split(X_seq, y_seq, lbl)
        Xtr, ytr, ltr, Xv, yv, lv, Xte, yte, lte = parts

        # Normalise using training statistics
        Xtr_n, Xv_n, Xte_n, xmin, xmax = normalize(
            Xtr.reshape(-1, Xtr.shape[-1]),
            Xv.reshape(-1, Xv.shape[-1]),
            Xte.reshape(-1, Xte.shape[-1]),
        )
        Xtr_n = Xtr_n.reshape(Xtr.shape)
        Xv_n  = Xv_n.reshape(Xv.shape)
        Xte_n = Xte_n.reshape(Xte.shape)

        if x_min_all is None:
            x_min_all, x_max_all = xmin, xmax
        else:
            x_min_all = np.minimum(x_min_all, xmin)
            x_max_all = np.maximum(x_max_all, xmax)

        all_splits["train"].append((Xtr_n, ytr, ltr))
        all_splits["val"].append((Xv_n, yv, lv))
        all_splits["test"].append((Xte_n, yte, lte))

    combined = {}
    for split in ("train", "val", "test"):
        Xs = np.concatenate([t[0] for t in all_splits[split]], axis=0)
        ys = np.concatenate([t[1] for t in all_splits[split]], axis=0)
        ls = np.concatenate([t[2] for t in all_splits[split]], axis=0)
        combined[split] = (Xs, ys, ls)

    combined["x_min"] = x_min_all
    combined["x_max"] = x_max_all
    combined["feature_names"] = features
    return combined
