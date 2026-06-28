"""detection.py -- unsupervised drift detection by C*-algebraic reconstruction.

The detector fits a low-rank nominal subspace in the feature space of a kernel
and scores a test window by its residual energy off that subspace.  Swapping the
feature map turns the same reconstruction detector into different monitors:

* ``raw``     -- the level-flatten detector of the original manuscript;
* ``cst``     -- the causal spectral-truncation (noncommutative) kernel, order n;
* ``periodic``-- the periodic spectral-truncation kernel of Hashimoto et al.;
* ``commutative`` -- the instantaneous (lag-0, ``n = 1``) covariance kernel.

The headline experiment is the *truncation sweep*: on correlated (crosstalk)
drift the commutative monitors sit at chance while the causal noncommutative
kernel detects, with a clear optimal truncation ``n`` -- the central novelty.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .kernels import cst_feature_map, periodic_truncation_feature_map

__all__ = [
    "feature_backend",
    "reconstruction_scores",
    "detector_auc",
    "truncation_sweep",
    "rank_truncation_grid",
    "baseline_comparison",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature backends
# ─────────────────────────────────────────────────────────────────────────────
def feature_backend(windows: np.ndarray, kind: str, n: int = 1) -> np.ndarray:
    """Map a batch of windows to the feature matrix used by the detector."""
    if kind == "raw":
        return windows.reshape(len(windows), -1)
    if kind == "commutative":
        return cst_feature_map(windows, 1)
    if kind == "cst":
        return cst_feature_map(windows, n)
    if kind == "periodic":
        return periodic_truncation_feature_map(windows, n)
    raise ValueError(f"unknown feature backend {kind!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Low-rank reconstruction detector
# ─────────────────────────────────────────────────────────────────────────────
def reconstruction_scores(F_train_nominal: np.ndarray, F_test: np.ndarray, k: int) -> np.ndarray:
    """Rank-``k`` reconstruction residual scores in a fixed feature space.

    The feature matrices are standardised on the nominal training statistics, a
    rank-``k`` principal subspace is fit on the (centred) nominal features, and
    each test window is scored by its mean squared residual outside that subspace.
    Larger score = less well explained by nominal operation = more anomalous.
    """
    mu = F_train_nominal.mean(0)
    sd = F_train_nominal.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Ftr = (F_train_nominal - mu) / sd
    Fte = (F_test - mu) / sd
    center = Ftr.mean(0)
    _, _, Vt = np.linalg.svd(Ftr - center, full_matrices=False)
    Wk = Vt[: max(1, k)]
    B = Fte - center
    proj = (B @ Wk.T) @ Wk
    return ((B - proj) ** 2).mean(1)


def detector_auc(windows, labels, train_idx, test_idx, kind, n=1, k=3):
    """ROC-AUC of the reconstruction detector on one train/test split."""
    nominal = train_idx[labels[train_idx] == 0]
    F_nom = feature_backend(windows[nominal], kind, n)
    F_te = feature_backend(windows[test_idx], kind, n)
    if len(np.unique(labels[test_idx])) < 2:
        return np.nan
    scores = reconstruction_scores(F_nom, F_te, k)
    return roc_auc_score(labels[test_idx], scores)


def _splits(N, n_splits, train_frac, seed):
    rng = np.random.default_rng(seed)
    n_tr = int(train_frac * N)
    return [(p[:n_tr], p[n_tr:]) for p in (rng.permutation(N) for _ in range(n_splits))]


# ─────────────────────────────────────────────────────────────────────────────
# Experiments
# ─────────────────────────────────────────────────────────────────────────────
def truncation_sweep(windows, labels, n_grid, kind="cst", k="best",
                     k_grid=(1, 2, 3, 5, 8, 12, 16), n_splits=8, train_frac=0.7,
                     seed=0):
    """AUC versus truncation order ``n`` (mean +/- s.d. over randomised splits).

    If ``k == 'best'`` the per-``n`` best rank over ``k_grid`` is reported (the
    detector's achievable AUC); otherwise the fixed rank ``k`` is used.
    """
    labels = labels.astype(int)
    splits = _splits(len(windows), n_splits, train_frac, seed)
    mean, sd, best_k = [], [], []
    for n in n_grid:
        ks = k_grid if k == "best" else [k]
        best_mean, best_sd, bk = -1.0, 0.0, ks[0]
        for kk in ks:
            vals = [detector_auc(windows, labels, tr, te, kind, n, kk) for tr, te in splits]
            vals = [v for v in vals if not np.isnan(v)]
            m = float(np.mean(vals))
            if m > best_mean:
                best_mean, best_sd, bk = m, float(np.std(vals)), kk
        mean.append(best_mean)
        sd.append(best_sd)
        best_k.append(bk)
    return {"n_grid": list(n_grid), "auc_mean": mean, "auc_sd": sd, "best_k": best_k}


def rank_truncation_grid(windows, labels, n_grid, k_grid, kind="cst",
                         n_splits=8, train_frac=0.7, seed=0):
    """2-D grid of mean AUC over (truncation ``n``, bottleneck rank ``k``)."""
    labels = labels.astype(int)
    splits = _splits(len(windows), n_splits, train_frac, seed)
    grid = np.empty((len(n_grid), len(k_grid)))
    for i, n in enumerate(n_grid):
        for j, k in enumerate(k_grid):
            vals = [detector_auc(windows, labels, tr, te, kind, n, k) for tr, te in splits]
            vals = [v for v in vals if not np.isnan(v)]
            grid[i, j] = float(np.mean(vals))
    return {"n_grid": list(n_grid), "k_grid": list(k_grid), "auc": grid}


def baseline_comparison(windows, labels, n_cst, n_periodic, k=8,
                        n_splits=8, train_frac=0.7, seed=0):
    """AUC of competing detectors on the same correlated-drift benchmark.

    Returns a dict ``method -> (mean, sd)`` for the level-flatten detector, the
    commutative (instantaneous) kernel, the periodic spectral-truncation kernel
    (Hashimoto et al.), and the causal spectral-truncation kernel (ours).
    """
    labels = labels.astype(int)
    splits = _splits(len(windows), n_splits, train_frac, seed)

    def evaluate(kind, n):
        vals = [detector_auc(windows, labels, tr, te, kind, n, k) for tr, te in splits]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)), float(np.std(vals))

    return {
        "raw level (flatten)": evaluate("raw", 1),
        "commutative kernel (n=1)": evaluate("commutative", 1),
        f"periodic spectral-truncation (n={n_periodic})": evaluate("periodic", n_periodic),
        f"causal spectral-truncation (ours, n={n_cst})": evaluate("cst", n_cst),
    }
