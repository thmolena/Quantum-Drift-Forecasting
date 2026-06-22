"""
make_paper_figures.py — regenerate the four publication figures referenced by
submission/main.tex (fig1_dynamics, fig2_forecasting, fig3_benchmark, fig4_anomaly).

All content is derived from the repository's own telemetry generator
(``src.data.generate_synthetic_dataset``, identical to the paper's Methods) and
the exact metric values recorded by the executed sequence-model notebooks (which
are also the values printed in the manuscript's Tables 2-4). No numbers are
fabricated: Figs 1, 2 and 4 are computed live from the generator following the
Methods section; Fig 3 charts the published benchmark metrics.

Run from the repository root:
    KMP_DUPLICATE_LIB_OK=TRUE MPLBACKEND=Agg python scripts/make_paper_figures.py
"""

import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve

# Make the repo's own data module importable.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data import generate_synthetic_dataset, FEATURE_COLS  # noqa: E402

OUT = os.path.join(REPO, "submission", "figures")
os.makedirs(OUT, exist_ok=True)

# Manuscript styling: blue accent matches the journal accent colour in main.tex.
ACCENT = "#1A4E8A"
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── Telemetry-generator constants (mirror the Methods section) ───────────────
N_QUBITS = 5
N_STEPS = 200          # 200 half-hour steps == 100 hours
SEQ_LEN = 32
HORIZON = 8
SEEDS = [42, 43, 44, 45, 46]
T1_THRESHOLD = 50.0    # micro-seconds operational threshold


def qubit_matrix(seed):
    """Return T1 matrix of shape (n_steps, n_qubits) and the time axis (hours)."""
    df = generate_synthetic_dataset(n_qubits=N_QUBITS, n_steps=N_STEPS, seed=seed)
    t = np.sort(df["timestamp_hr"].unique())
    T1 = np.stack(
        [df[df["qubit_id"] == q].sort_values("timestamp_hr")["T1_us"].values
         for q in range(N_QUBITS)],
        axis=1,
    )  # (n_steps, n_qubits)
    return T1, t


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 — Coherence drift: dynamics and predictable structure
# ═════════════════════════════════════════════════════════════════════════════
def figure1():
    T1, t = qubit_matrix(seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) T1 trajectories for five qubits with the 50 us threshold.
    cmap = plt.get_cmap("viridis")
    for q in range(N_QUBITS):
        axes[0].plot(t, T1[:, q], lw=1.1, color=cmap(q / max(N_QUBITS - 1, 1)),
                     label=f"Qubit {q}")
    axes[0].axhline(T1_THRESHOLD, ls="--", color="#444444", lw=1.0,
                    label=r"$50\,\mu$s threshold")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel(r"$T_1$ relaxation time ($\mu$s)")
    axes[0].set_title("(a) Coherence-time trajectories")
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)

    # (b) Mean autocorrelation of detrended T1 across qubits (normalised autocov).
    max_lag = 60  # lags in steps; 1 step == 0.5 h
    acfs = []
    for q in range(N_QUBITS):
        x = T1[:, q].astype(float)
        # Linear detrend.
        coeffs = np.polyfit(np.arange(len(x)), x, 1)
        x = x - np.polyval(coeffs, np.arange(len(x)))
        x = x - x.mean()
        denom = np.dot(x, x)
        ac = np.array([np.dot(x[: len(x) - k], x[k:]) / denom
                       for k in range(max_lag + 1)])
        acfs.append(ac)
    acfs = np.array(acfs)
    mean_ac = acfs.mean(axis=0)
    sd_ac = acfs.std(axis=0)
    lags_h = np.arange(max_lag + 1) * 0.5

    axes[1].plot(lags_h, mean_ac, color=ACCENT, lw=1.8, label="Mean across qubits")
    axes[1].fill_between(lags_h, mean_ac - sd_ac, mean_ac + sd_ac,
                         color=ACCENT, alpha=0.20, label=r"$\pm 1$ s.d.")
    axes[1].axhline(0.5, ls=":", color="#888888", lw=1.0)
    axes[1].axhline(0.0, color="#444444", lw=0.8)
    axes[1].set_xlabel("Lag (hours)")
    axes[1].set_ylabel(r"Autocorrelation of $T_1$")
    axes[1].set_title("(b) Slow, forecastable structure")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT, "fig1_dynamics.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] mean ACF>0.5 holds to ~{lags_h[mean_ac > 0.5][-1]:.1f} h  ->  {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 — Forecasting baselines (persistence, climatology, AR-ridge, mv-ridge)
# ═════════════════════════════════════════════════════════════════════════════
def _build_windows(seed):
    """Per-qubit sliding windows; returns lists keyed by leak-free 80/20 split."""
    df = generate_synthetic_dataset(n_qubits=N_QUBITS, n_steps=N_STEPS, seed=seed)
    Xtr_full, ytr_full, Xte_full, yte_full = [], [], [], []
    Xtr_t1, Xte_t1 = [], []
    for q in range(N_QUBITS):
        sub = df[df["qubit_id"] == q].sort_values("timestamp_hr")
        feats = sub[FEATURE_COLS].values.astype(float)           # (T, 7)
        t1 = sub["T1_us"].values.astype(float)                   # (T,)
        T = len(t1)
        n_win = T - SEQ_LEN - HORIZON + 1
        # Leak-free 80/20 chronological split on the window index.
        split = int(n_win * 0.8)
        for i in range(n_win):
            win = feats[i:i + SEQ_LEN]                            # (32, 7)
            t1_hist = t1[i:i + SEQ_LEN]                           # (32,)
            tgt = t1[i + SEQ_LEN:i + SEQ_LEN + HORIZON]          # (8,)
            if i < split:
                Xtr_full.append(win.ravel()); ytr_full.append(tgt)
                Xtr_t1.append(t1_hist)
            else:
                Xte_full.append(win.ravel()); yte_full.append(tgt)
                Xte_t1.append(t1_hist)
    return (np.array(Xtr_full), np.array(ytr_full), np.array(Xte_full),
            np.array(yte_full), np.array(Xtr_t1), np.array(Xte_t1))


def _forecast_maes(seed):
    """Return per-horizon MAE arrays (len=HORIZON) for the four forecasters."""
    Xtr, ytr, Xte, yte, Xtr_t1, Xte_t1 = _build_windows(seed)

    # Persistence: carry the last observed T1 forward across the horizon.
    last_obs = Xte_t1[:, -1:]                                     # (N, 1)
    pred_pers = np.repeat(last_obs, HORIZON, axis=1)

    # Climatology: predict the training-mean horizon trajectory.
    clim = ytr.mean(axis=0, keepdims=True)                       # (1, 8)
    pred_clim = np.repeat(clim, len(yte), axis=0)

    # AR-ridge on the 32-step T1 history.
    ar = Ridge(alpha=5.0).fit(Xtr_t1, ytr)
    pred_ar = ar.predict(Xte_t1)

    # Multivariate ridge on the flattened 32x7 telemetry window.
    mv = Ridge(alpha=5.0).fit(Xtr, ytr)
    pred_mv = mv.predict(Xte)

    def mae_per_h(pred):
        return np.mean(np.abs(pred - yte), axis=0)

    return {
        "Persistence": mae_per_h(pred_pers),
        "Climatology": mae_per_h(pred_clim),
        "AR-ridge ($T_1$ history)": mae_per_h(pred_ar),
        "Multivariate ridge": mae_per_h(pred_mv),
    }


def figure2():
    names = ["Persistence", "Climatology", "AR-ridge ($T_1$ history)",
             "Multivariate ridge"]
    stacks = {n: [] for n in names}
    for s in SEEDS:
        res = _forecast_maes(s)
        for n in names:
            stacks[n].append(res[n])
    mean = {n: np.mean(stacks[n], axis=0) for n in names}
    sd = {n: np.std(stacks[n], axis=0) for n in names}

    horizons = np.arange(1, HORIZON + 1)
    colors = {"Persistence": "#888888", "Climatology": "#d62728",
              "AR-ridge ($T_1$ history)": "#2ca02c", "Multivariate ridge": ACCENT}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) MAE vs horizon.
    for n in names:
        axes[0].plot(horizons, mean[n], marker="o", ms=4, lw=1.6,
                     color=colors[n], label=n)
        axes[0].fill_between(horizons, mean[n] - sd[n], mean[n] + sd[n],
                             color=colors[n], alpha=0.15)
    axes[0].set_xlabel("Forecast horizon (steps)")
    axes[0].set_ylabel(r"MAE ($\mu$s)")
    axes[0].set_title("(a) Error vs forecast horizon")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_xticks(horizons)

    # (b) Skill over persistence for the two learned forecasters.
    pers = mean["Persistence"]
    for n in ["AR-ridge ($T_1$ history)", "Multivariate ridge"]:
        skill = 1.0 - mean[n] / pers
        axes[1].plot(horizons, 100 * skill, marker="s", ms=4, lw=1.8,
                     color=colors[n], label=n)
    axes[1].axhline(0.0, color="#444444", lw=0.8)
    axes[1].set_xlabel("Forecast horizon (steps)")
    axes[1].set_ylabel("Skill over persistence (%)")
    axes[1].set_title("(b) Skill widens with horizon")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_xticks(horizons)

    fig.tight_layout()
    path = os.path.join(OUT, "fig2_forecasting.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    # Report the headline numbers for cross-checking against Table 1.
    print("[fig2] MAE@1step (us):",
          {n: round(float(mean[n][0]), 2) for n in names})
    print("[fig2] MAE@8step (us):",
          {n: round(float(mean[n][-1]), 2) for n in names})
    mv_skill8 = 1 - mean["Multivariate ridge"][-1] / pers[-1]
    print(f"[fig2] multivariate-ridge skill @8 steps = {100*mv_skill8:.0f}%  ->  {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 — Benchmark (charts the published notebook-recorded metrics)
# ═════════════════════════════════════════════════════════════════════════════
def figure3():
    # Values transcribed from the manuscript Tables 2-4 / executed-notebook
    # outputs (quantum_drift_combined.ipynb). These are the recorded metrics.
    models = ["RNN", "LSTM", "GRU", "Transformer"]
    mcol = {"RNN": "#94a3b8", "LSTM": "#f59e0b", "GRU": "#10b981",
            "Transformer": "#6366f1"}

    # (a) Thermal incident detection: ROC-AUC and F1 (Extended Data Table edtab:thermal).
    thermal_auc = {"RNN": 0.408, "LSTM": 0.423, "GRU": 0.718}
    thermal_f1 = {"RNN": 0.000, "LSTM": 0.000, "GRU": 0.257}

    # (b) Three-dataset average ROC-AUC (Extended Data Table edtab:cross) + Transformer* periodic.
    avg_auc = {"GRU": 0.660, "LSTM": 0.628, "Transformer": 0.196}
    transformer_periodic = 0.799

    # (c) Parameter counts (Extended Data Table edtab:thermal).
    params = {"RNN": 5645, "LSTM": 116845, "GRU": 87949, "Transformer": 116845}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # Panel (a): grouped bars ROC-AUC (with chance line) and F1.
    tm = ["RNN", "LSTM", "GRU"]
    x = np.arange(len(tm))
    w = 0.38
    axes[0].bar(x - w / 2, [thermal_auc[m] for m in tm], width=w,
                color=[mcol[m] for m in tm], label="ROC-AUC")
    axes[0].bar(x + w / 2, [thermal_f1[m] for m in tm], width=w,
                color=[mcol[m] for m in tm], alpha=0.45, hatch="//", label="F1")
    axes[0].axhline(0.5, ls="--", color="#444444", lw=1.0, label="Chance (AUC 0.5)")
    for i, m in enumerate(tm):
        axes[0].text(i - w / 2, thermal_auc[m] + 0.01, f"{thermal_auc[m]:.2f}",
                     ha="center", va="bottom", fontsize=7.5)
        axes[0].text(i + w / 2, thermal_f1[m] + 0.01, f"{thermal_f1[m]:.2f}",
                     ha="center", va="bottom", fontsize=7.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(tm)
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("(a) Thermal incident detection")
    axes[0].legend(fontsize=7.5, loc="upper left")

    # Panel (b): three-dataset average ROC-AUC + hatched Transformer* (periodic).
    bm = ["GRU", "LSTM", "Transformer"]
    xb = np.arange(len(bm) + 1)
    vals = [avg_auc[m] for m in bm] + [transformer_periodic]
    cols = [mcol[m] for m in bm] + [mcol["Transformer"]]
    bars = axes[1].bar(xb, vals, color=cols, width=0.6)
    bars[-1].set_hatch("xx")
    bars[-1].set_alpha(0.55)
    axes[1].axhline(0.5, ls="--", color="#444444", lw=1.0)
    labels = bm + ["Transformer$^{*}$"]
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    axes[1].set_xticks(xb); axes[1].set_xticklabels(labels, rotation=20, fontsize=8)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title("(b) Cross-domain avg ROC-AUC")

    # Panel (c): parameter counts on a log scale.
    pm = ["RNN", "LSTM", "GRU"]
    axes[2].bar(pm, [params[m] for m in pm], color=[mcol[m] for m in pm], width=0.6)
    axes[2].set_yscale("log")
    for i, m in enumerate(pm):
        axes[2].text(i, params[m] * 1.08, f"{params[m]:,}", ha="center",
                     va="bottom", fontsize=7.5)
    axes[2].set_ylabel("Trainable parameters (log)")
    axes[2].set_title("(c) Parameter efficiency")

    fig.tight_layout()
    path = os.path.join(OUT, "fig3_benchmark.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] charted published thermal/cross-domain/param metrics  ->  {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 — Unsupervised drift detection by reconstruction error (PCA, rank-k)
# ═════════════════════════════════════════════════════════════════════════════
def _recon_windows(seed):
    """Flattened windows + drift labels.

    Window label is the drift label at the forecast endpoint
    (``i + SEQ_LEN + HORIZON - 1``), matching ``src.data.make_sequences``.
    Features are standardised per qubit so reconstruction error is balanced
    across the seven telemetry channels; the nominal manifold is therefore
    tight, and an over-complete code memorises drifted windows (the collapse in
    Fig. 4b).
    """
    df = generate_synthetic_dataset(n_qubits=N_QUBITS, n_steps=N_STEPS, seed=seed)
    Xs, ys = [], []
    for q in range(N_QUBITS):
        sub = df[df["qubit_id"] == q].sort_values("timestamp_hr")
        feats = sub[FEATURE_COLS].values.astype(float)
        lab = sub["drift_label"].values.astype(int)
        # Standardise per-feature so reconstruction error is balanced.
        mu, sd = feats.mean(0), feats.std(0)
        sd[sd == 0] = 1.0
        feats = (feats - mu) / sd
        T = len(lab)
        n_win = T - SEQ_LEN - HORIZON + 1
        for i in range(n_win):
            Xs.append(feats[i:i + SEQ_LEN].ravel())
            ys.append(int(lab[i + SEQ_LEN + HORIZON - 1]))
    return np.array(Xs), np.array(ys)


def _pca_recon_auc(X, y, rank, rng):
    """Fit rank-k PCA on nominal windows of a random 70/30 split; AUC of residual."""
    n = len(X)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    Xtr, Xte, yte = X[tr], X[te], y[te]
    nominal = Xtr[y[tr] == 0]
    if len(nominal) < rank + 1 or len(np.unique(yte)) < 2:
        return None, None, None
    mu = nominal.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(nominal - mu, full_matrices=False)
    comps = Vt[:rank]                                            # (rank, D)
    centred = Xte - mu
    recon = centred @ comps.T @ comps
    score = np.mean((centred - recon) ** 2, axis=1)             # per-window MSE
    auc = roc_auc_score(yte, score)
    return auc, yte, score


def figure4():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) ROC curve for the rank-3 detector (representative seed/split).
    X, y = _recon_windows(seed=42)
    rng = np.random.default_rng(0)
    auc3, yte, score = _pca_recon_auc(X, y, rank=3, rng=rng)
    fpr, tpr, _ = roc_curve(yte, score)
    axes[0].plot(fpr, tpr, color=ACCENT, lw=2.0,
                 label=f"Rank-3 detector (AUC = {auc3:.2f})")
    axes[0].plot([0, 1], [0, 1], ls="--", color="#888888", lw=1.0,
                 label="Chance")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].set_title("(a) Rank-3 reconstruction ROC")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)

    # (b) ROC-AUC vs bottleneck rank over eight randomised splits.
    ranks = list(range(1, SEQ_LEN + 1))
    means, sds = [], []
    for k in ranks:
        aucs = []
        for s in range(8):
            rng = np.random.default_rng(100 + s)
            a, _, _ = _pca_recon_auc(X, y, rank=k, rng=rng)
            if a is not None:
                aucs.append(a)
        means.append(np.mean(aucs))
        sds.append(np.std(aucs))
    means = np.array(means); sds = np.array(sds)
    axes[1].plot(ranks, means, color=ACCENT, lw=1.8, marker="o", ms=3)
    axes[1].fill_between(ranks, means - sds, means + sds, color=ACCENT, alpha=0.20,
                         label=r"$\pm 1$ s.d. (8 splits)")
    axes[1].axhline(0.5, ls="--", color="#444444", lw=1.0, label="Chance")
    axes[1].set_xlabel("Reconstruction bottleneck rank $k$")
    axes[1].set_ylabel("Detection ROC-AUC")
    axes[1].set_title("(b) Tight manifold separates drift")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    path = os.path.join(OUT, "fig4_anomaly.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] rank-1 AUC={means[0]:.2f}, rank-3 AUC={auc3:.2f}  ->  {path}")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nAll figures written to:", OUT)
