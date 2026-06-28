"""forecasting.py -- coherence forecasters, including the CST-kernel forecaster.

Two families are evaluated on the seeded telemetry generator:

* classical baselines -- persistence, climatology, an autoregressive ridge on the
  T1 history, and a multivariate ridge on the full window (the commutative,
  ``n = 1`` member of the kernel family); and
* the causal spectral-truncation (CST) kernel forecaster -- multivariate ridge in
  the feature space that augments the raw window with the order-``n`` causal
  lagged cross-channel coordinates.

The CST forecaster contains the multivariate ridge as its ``n = 1`` special case,
so the truncation order ``n`` interpolates from the commutative baseline upward.
On the smooth, mean-reverting marginal telemetry of the generator the linear
baseline is already near-optimal, so forecast skill is essentially flat in ``n``
(the noncommutative coordinates neither help nor hurt) -- the honest, objective
counterpart of the detection result, where noncommutativity is decisive.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .data import (
    FEATURE_COLS,
    extract_qubit_series,
    generate_synthetic_dataset,
    make_sequences,
)
from .kernels import cst_feature_map

__all__ = ["pooled_windows", "forecast_baselines", "cstk_forecast_skill"]

N_QUBITS = 5
N_STEPS = 200
SEQ_LEN = 32
HORIZON = 8
RIDGE_ALPHA = 5.0


def pooled_windows(seed: int, n_qubits=N_QUBITS, n_steps=N_STEPS,
                   seq_len=SEQ_LEN, horizon=HORIZON):
    """Sliding windows pooled across qubits for one telemetry seed."""
    df = generate_synthetic_dataset(n_qubits=n_qubits, n_steps=n_steps, seed=seed)
    Xs, ys = [], []
    for q in range(n_qubits):
        X, y = extract_qubit_series(df, q, FEATURE_COLS)
        a, b, _ = make_sequences(X, y, seq_len=seq_len, horizon=horizon, target_col_idx=0)
        Xs.append(a)
        ys.append(b)
    return Xs, ys


def _leakfree_split(Xs_list, ys_list, train_frac=0.8):
    Xtr, ytr, Xte, yte = [], [], [], []
    for Xs, ys in zip(Xs_list, ys_list):
        n_tr = int(len(Xs) * train_frac)
        Xtr.append(Xs[:n_tr]); ytr.append(ys[:n_tr])
        Xte.append(Xs[n_tr:]); yte.append(ys[n_tr:])
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xte), np.concatenate(yte))


def forecast_baselines(seed: int):
    """Per-horizon MAE for persistence, climatology, AR-ridge, multivariate ridge."""
    Xs, ys = pooled_windows(seed)
    Xtr, ytr, Xte, yte = _leakfree_split(Xs, ys)
    H = ytr.shape[1]
    pers = Xte[:, -1, 0][:, None].repeat(H, axis=1)
    clim = ytr.mean(axis=0)[None, :].repeat(len(yte), axis=0)
    ar = Ridge(alpha=RIDGE_ALPHA).fit(Xtr[:, :, 0], ytr).predict(Xte[:, :, 0])
    mv = Ridge(alpha=RIDGE_ALPHA).fit(Xtr.reshape(len(Xtr), -1), ytr).predict(Xte.reshape(len(Xte), -1))
    mae = lambda p: np.abs(yte - p).mean(axis=0)
    return {"pers": mae(pers), "clim": mae(clim), "ar": mae(ar), "mv": mae(mv)}


def cstk_forecast_skill(seed: int, n_grid=(1, 2, 3, 4)):
    """Eight-step forecast skill over persistence for the CST forecaster vs ``n``.

    Returns ``{n: skill_at_horizon_8}``.  ``n = 1`` reproduces the multivariate
    ridge (commutative) baseline up to the added lag-0 covariance coordinates.
    """
    Xs, ys = pooled_windows(seed)
    Xtr, ytr, Xte, yte = _leakfree_split(Xs, ys)
    H = ytr.shape[1]
    pers = Xte[:, -1, 0][:, None].repeat(H, axis=1)
    pers_mae8 = np.abs(yte - pers).mean(axis=0)[-1]
    out = {}
    for n in n_grid:
        Ftr = np.concatenate([Xtr.reshape(len(Xtr), -1), cst_feature_map(Xtr, n)], axis=1)
        Fte = np.concatenate([Xte.reshape(len(Xte), -1), cst_feature_map(Xte, n)], axis=1)
        pr = Ridge(alpha=RIDGE_ALPHA).fit(Ftr, ytr).predict(Fte)
        mae8 = np.abs(yte - pr).mean(axis=0)[-1]
        out[n] = 1.0 - mae8 / pers_mae8
    return out
