"""kernels.py -- causal spectral-truncation C*-algebraic kernels for telemetry.

This module implements the central algorithmic contribution of the manuscript: a
family of C*-algebra-valued positive-definite kernels on multivariate telemetry
windows whose *noncommutativity* is controlled by a truncation parameter ``n``.
The construction is a causal analogue of the spectral-truncation kernels of
Hashimoto et al. (NeurIPS 2024): where their truncation map projects functions on
the periodic torus onto Toeplitz matrices of Fourier coefficients, we project a
*causal* time series onto banded lower-triangular Toeplitz (FIR-convolution)
operators, so the representation respects the arrow of time and the products that
appear in the kernel mix ordered cross-channel lags.

Mathematical summary
---------------------
A window is ``X in R^{L x C}`` (``L`` time steps, ``C`` channels).  Let ``S`` be
the lower (causal) shift on ``R^L`` ((S v)_t = v_{t-1}, v_{-1} := 0).  The
*causal spectral-truncation map* of order ``n`` represents the window inside the
finite-dimensional C*-algebra ``M_C`` (the channel algebra) through the truncated
shift powers ``{S^tau : 0 <= tau < n}``: it returns the operator-valued kernel

    K_n(X, Y) = sum_{tau=0}^{n-1} w_tau * X^T S^tau Y   in   M_C,
    K_n(X, Y)[c, c'] = w_tau-weighted sum of <X[:, c], S^tau Y[:, c']>,

i.e. the ``C x C`` matrix of *ordered* causal lagged cross-channel correlations
(``w_tau = 1 / (L - tau)``).  Truncating the shift algebra to ``n`` powers is the
causal counterpart of truncating the Fourier multiplier algebra to ``n`` modes in
Hashimoto et al.; ``n = 1`` keeps only the diagonal (instantaneous) part and is
*commutative* (``X^T Y`` is symmetric), whereas for ``tau >= 1`` the shift does
not commute with channel selection, so ``K_n(X, Y)[c, c'] != K_n(X, Y)[c', c]``.
That asymmetry is the noncommutativity: it lets the kernel see interactions
``X[:, c](t) Y[:, c'](t - tau)`` between *different* points of the data domain,
which a commutative (``n = 1``) kernel cannot.  The scalar kernel is the
Hilbert-Schmidt inner product of these C*-algebra elements,

    k_n(X, Y) = < Phi_n(X), Phi_n(Y) >_F = vec Phi_n(X) . vec Phi_n(Y),

with ``Phi_n(X)[c, c', tau] = <X[:, c], S^tau X[:, c']> / (L - tau)`` the lagged
cross-correlation tensor; it is positive definite for every ``n`` (explicit
feature map) and reduces to a commutative instantaneous-covariance kernel at
``n = 1``.

References
----------
Hashimoto, Y. et al. Spectral truncation kernels: noncommutativity in
C*-algebraic kernel machines. NeurIPS 2024 (arXiv:2405.17823).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "lower_shift",
    "cst_feature_map",
    "cst_operator_kernel",
    "cst_scalar_gram",
    "periodic_truncation_feature_map",
    "rbf_gram",
]


# ─────────────────────────────────────────────────────────────────────────────
# Truncated causal shift (the generator of the noncommutativity)
# ─────────────────────────────────────────────────────────────────────────────
def lower_shift(L: int, tau: int = 1) -> np.ndarray:
    """Return the ``L x L`` causal shift power ``S^tau`` ((S v)_t = v_{t-1}).

    The order-``n`` causal spectral truncation represents a window through the
    powers ``{S^tau : 0 <= tau < n}``.  ``S`` is nilpotent and does not commute
    with channel selection; that non-commutation is the source of the kernel's
    noncommutativity and the reason ``K_n(X, Y)`` is not symmetric for ``n > 1``.
    """
    S = np.zeros((L, L))
    if 0 <= tau < L:
        idx = np.arange(L - tau)
        S[idx + tau, idx] = 1.0
    return S


# ─────────────────────────────────────────────────────────────────────────────
# Causal spectral-truncation feature map (windows -> noncommutative tensor)
# ─────────────────────────────────────────────────────────────────────────────
def cst_feature_map(windows: np.ndarray, n: int, center: bool = True) -> np.ndarray:
    """Vectorised causal-truncation feature map ``Phi_n`` for a batch of windows.

    Parameters
    ----------
    windows : ndarray, shape ``(N, L, C)``
        ``N`` telemetry windows of length ``L`` over ``C`` channels.
    n : int
        Truncation order (number of causal lags ``tau = 0, ..., n - 1``).  ``n =
        1`` yields the commutative instantaneous channel covariance; ``n > 1``
        adds *ordered* lagged cross-channel correlations (the noncommutative
        coordinates).
    center : bool
        Subtract the per-window temporal mean of each channel before forming
        products (focuses the kernel on covariance structure, not level).

    Returns
    -------
    ndarray, shape ``(N, C * C * n)``
        The flattened tensor ``Phi_n(X)[c, c', tau] = (1 / (L - tau))
        sum_{t >= tau} X[t, c] X[t - tau, c']``.  Note ``Phi_n[c, c', tau] !=
        Phi_n[c', c, tau]`` for ``tau >= 1`` -- the entries are matrix elements of
        the *noncommuting* products ``M_c S^tau``.
    """
    X = np.asarray(windows, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"expected windows of shape (N, L, C), got {X.shape}")
    N, L, C = X.shape
    n = int(max(1, min(n, L)))
    Xc = X - X.mean(axis=1, keepdims=True) if center else X
    feats = np.empty((N, C, C, n))
    for tau in range(n):
        if tau == 0:
            g = np.einsum("ntc,ntd->ncd", Xc, Xc) / L
        else:
            # <X[:, c], S^tau X[:, c']>: align X[t, c] with X[t - tau, c'].
            g = np.einsum("ntc,ntd->ncd", Xc[:, tau:, :], Xc[:, : L - tau, :]) / (L - tau)
        feats[:, :, :, tau] = g
    return feats.reshape(N, C * C * n)


def cst_operator_kernel(x: np.ndarray, y: np.ndarray, n: int, center: bool = True) -> np.ndarray:
    """Operator-valued kernel ``K_n(x, y) in M_C`` for two single windows.

    Returns the ``C x C`` C*-algebra element whose ``(c, c')`` entry is the
    order-``n`` causal lagged cross-correlation summed over lags.  This is the
    RKHM (reproducing kernel Hilbert C*-module) kernel value; the scalar Gram is
    its Hilbert-Schmidt norm contribution.
    """
    x = np.asarray(x, float)[None]
    y = np.asarray(y, float)[None]
    L, C = x.shape[1], x.shape[2]
    n = int(max(1, min(n, L)))
    xc = x - x.mean(1, keepdims=True) if center else x
    yc = y - y.mean(1, keepdims=True) if center else y
    K = np.zeros((C, C))
    for tau in range(n):
        if tau == 0:
            K += np.einsum("ntc,ntd->cd", xc, yc) / L
        else:
            K += np.einsum("ntc,ntd->cd", xc[:, tau:, :], yc[:, : L - tau, :]) / (L - tau)
    return K


def cst_scalar_gram(A: np.ndarray, B: np.ndarray, n: int, center: bool = True) -> np.ndarray:
    """Positive-definite scalar Gram ``k_n(A_i, B_j) = <Phi_n(A_i), Phi_n(B_j)>``.

    ``A`` and ``B`` are batches of windows ``(N_A, L, C)`` and ``(N_B, L, C)``;
    returns the ``(N_A, N_B)`` Gram matrix of the linear kernel on the
    causal-truncation feature map.  Positive-definiteness is immediate from the
    explicit feature map ``Phi_n``.
    """
    Fa = cst_feature_map(A, n, center=center)
    Fb = cst_feature_map(B, n, center=center)
    return Fa @ Fb.T


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: periodic spectral-truncation kernel (Hashimoto et al., NeurIPS'24)
# ─────────────────────────────────────────────────────────────────────────────
def periodic_truncation_feature_map(windows: np.ndarray, n: int, center: bool = True) -> np.ndarray:
    """Periodic (non-causal) spectral-truncation feature map -- faithful baseline.

    Implements the original torus construction: each channel is mapped to the
    ``n x n`` Toeplitz matrix of its truncated Fourier coefficients,
    ``R_n^{per}(a)[i, j] = ahat_{i - j}``, and the noncommutative product kernel
    uses the ordered cross-channel products ``R_n^{per}(a_c)^* R_n^{per}(a_{c'})``.
    The feature is the real/imaginary parts of those product matrices.  This is the
    periodic counterpart of :func:`cst_feature_map`; it ignores causality (it
    treats each window as a function on the circle), which is exactly the
    inductive bias the causal kernel removes.
    """
    X = np.asarray(windows, dtype=float)
    N, L, C = X.shape
    n = int(max(1, min(n, L)))
    Xc = X - X.mean(axis=1, keepdims=True) if center else X
    # Truncated Fourier coefficients ahat_{-(n-1)}..ahat_{n-1} per window-channel.
    fft = np.fft.fft(Xc, axis=1) / L              # (N, L, C) complex
    # Symmetric coefficient bank indexed by signed lag d in [-(n-1), n-1];
    # ahat_{-m} = conj(ahat_m) for a real signal.
    pos = fft[:, :n, :]                           # (N, n, C): lags 0..n-1
    neg = np.conj(fft[:, 1:n, :])[:, ::-1, :]     # (N, n-1, C): lags -(n-1)..-1
    bank = np.concatenate([neg, pos], axis=1)     # (N, 2n-1, C), index d+ (n-1)

    idx = np.arange(n)
    d = idx[:, None] - idx[None, :] + (n - 1)     # (n, n) -> indices into bank
    # Vectorised Toeplitz: T[:, i, j, c] = bank[:, d[i, j], c].
    Ts = [bank[:, d, c] for c in range(C)]        # list of (N, n, n)
    blocks = []
    for c in range(C):
        for cp in range(C):
            P = np.conj(np.transpose(Ts[c], (0, 2, 1))) @ Ts[cp]   # (N, n, n) product
            blocks.append(P.reshape(N, -1).real)
            blocks.append(P.reshape(N, -1).imag)
    return np.concatenate(blocks, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: commutative RBF kernel on instantaneous channel covariances
# ─────────────────────────────────────────────────────────────────────────────
def rbf_gram(A: np.ndarray, B: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """RBF Gram on the *commutative* instantaneous-covariance feature (``n = 1``).

    A standard commutative kernel baseline: featurise each window by its
    instantaneous (lag-0) channel covariance ``Phi_1`` and apply a Gaussian RBF.
    Carries no ordered cross-lag information, so it is blind to purely
    noncommutative (lagged cross-channel) drift.
    """
    Fa = cst_feature_map(A, 1)
    Fb = cst_feature_map(B, 1)
    # Standardise on A for a scale-free bandwidth.
    mu, sd = Fa.mean(0), Fa.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    Fa = (Fa - mu) / sd
    Fb = (Fb - mu) / sd
    sq = (Fa**2).sum(1)[:, None] + (Fb**2).sum(1)[None, :] - 2 * Fa @ Fb.T
    if gamma is None:
        gamma = 1.0 / Fa.shape[1]
    return np.exp(-gamma * np.maximum(sq, 0.0))
