#!/usr/bin/env python3
"""
make_paper_figures.py — single reproducible script that regenerates the four
figures used in submission/main.tex.

Design goals
------------
* CPU-only, matplotlib, fixed seeds.  No GPU, no PyTorch required.
* Reuses the repository's own telemetry generator and sequence utilities in
  ``src/`` (the exact T1 formula and feature set described in the Methods).
* Figures 1, 2 and 4 plus Table 1 (forecasting benchmark) are computed *live*
  from the seeded telemetry generator — these are deterministic and reproduce
  exactly on every run.
* Figure 3 reproduces the sequence-model benchmark metrics recorded by the
  executed notebooks (machine-temperature incident detection, three-dataset
  cross-domain average, periodic-regime specialisation, parameter counts).
  These values are *not* re-trained here; they are transcribed verbatim from the
  notebook outputs (see NOTEBOOK_METRICS below) and rendered, matching the
  Methods statement that Fig. 3 / Tables 2-3 come from the executed notebooks.

Usage
-----
    python submission/code/make_paper_figures.py
    # or, from anywhere:
    KMP_DUPLICATE_LIB_OK=TRUE python /abs/path/make_paper_figures.py

Outputs four PNGs into submission/figures/:
    fig1_dynamics.png, fig2_forecasting.png, fig3_benchmark.png, fig4_anomaly.png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless / CPU-only
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve

# ── Repository imports ───────────────────────────────────────────────────────
# Resolve the submission root (this file lives at submission/code/).
SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

from qdriftforecast.data import (  # noqa: E402
    FEATURE_COLS,
    extract_qubit_series,
    generate_synthetic_dataset,
    make_sequences,
)

# ── Global configuration (fixed seeds, telemetry spec from Methods) ──────────
N_QUBITS = 5          # Q = 5
N_STEPS = 200         # N = 200 half-hour steps  → 100 h
DT_HOURS = 0.5        # half-hour sampling
SEQ_LEN = 32
HORIZON = 8
T1_THRESHOLD = 50.0   # µs operational threshold

FORECAST_SEEDS = [0, 1, 2, 3, 4]   # five telemetry-generator seeds (Table 1 / Fig 2)
DETECTOR_SEED = 42                  # telemetry seed for the reconstruction study (Fig 4)
N_DETECTOR_SPLITS = 8              # randomised 70/30 splits for the rank sweep
RANK_GRID = list(range(1, SEQ_LEN + 1))  # bottleneck rank 1..32

OUT_DIR = SUBMISSION_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication styling (light theme, print-friendly) ────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 8.5,
    "legend.framealpha": 0.9,
    "figure.dpi": 150,
})

# Consistent palette across figures.
C_PERS = "#6b7280"   # persistence (grey)
C_CLIM = "#b45309"   # climatology (amber)
C_AR = "#2563eb"     # AR-ridge (blue)
C_MV = "#059669"     # multivariate ridge (green)
C_RNN = "#9ca3af"
C_LSTM = "#f59e0b"
C_GRU = "#059669"
C_TRANSF = "#7c3aed"
QUBIT_COLORS = ["#2563eb", "#059669", "#b45309", "#dc2626", "#7c3aed"]


# ─────────────────────────────────────────────────────────────────────────────
# Shared data helpers
# ─────────────────────────────────────────────────────────────────────────────
def pooled_windows(seed: int):
    """Return (X_windows, y_horizon, drift_label) pooled over all qubits.

    X_windows : (N, SEQ_LEN, n_features)
    y_horizon : (N, HORIZON)           — future T1 targets
    labels    : (N,)                   — drift label at the forecast horizon
    """
    df = generate_synthetic_dataset(n_qubits=N_QUBITS, n_steps=N_STEPS, seed=seed)
    Xs_list, ys_list, lbl_list = [], [], []
    for q in range(N_QUBITS):
        X, y = extract_qubit_series(df, q, FEATURE_COLS)
        Xs, ys, lbl = make_sequences(X, y, seq_len=SEQ_LEN, horizon=HORIZON, target_col_idx=0)
        Xs_list.append(Xs)
        ys_list.append(ys)
        lbl_list.append(lbl)
    return Xs_list, ys_list, lbl_list


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — coherence dynamics + predictable structure
# ─────────────────────────────────────────────────────────────────────────────
def figure1():
    df = generate_synthetic_dataset(n_qubits=N_QUBITS, n_steps=N_STEPS, seed=DETECTOR_SEED)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) T1 trajectories for 5 qubits over 100 h, with 50 µs threshold.
    for q in range(N_QUBITS):
        sub = df[df["qubit_id"] == q].sort_values("timestamp_hr")
        ax_a.plot(sub["timestamp_hr"].values, sub["T1_us"].values,
                  color=QUBIT_COLORS[q], linewidth=1.1, alpha=0.9, label=f"Qubit {q}")
    ax_a.axhline(T1_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1.3,
                 label=f"{T1_THRESHOLD:.0f} µs threshold")
    ax_a.set_xlabel("Time (hours)")
    ax_a.set_ylabel(r"$T_1$ relaxation time (µs)")
    ax_a.set_title("(a) Coherence trajectories")
    ax_a.legend(ncol=2, loc="upper right", fontsize=7.5)
    ax_a.set_xlim(0, (N_STEPS - 1) * DT_HOURS)

    # (b) Mean autocorrelation of detrended T1 vs lag, ±1 s.d. band across qubits.
    max_lag = 40
    acfs = []
    for q in range(N_QUBITS):
        s = df[df["qubit_id"] == q].sort_values("timestamp_hr")["T1_us"].values.astype(float)
        idx = np.arange(len(s))
        s = s - np.polyval(np.polyfit(idx, s, 1), idx)  # remove linear secular trend
        s = s - s.mean()
        denom = (s * s).sum()
        ac = np.array([(s[:len(s) - k] * s[k:]).sum() / denom for k in range(max_lag)])
        acfs.append(ac)
    acfs = np.array(acfs)
    lags_h = np.arange(max_lag) * DT_HOURS
    mean_ac = acfs.mean(0)
    sd_ac = acfs.std(0)
    ax_b.plot(lags_h, mean_ac, color=C_AR, linewidth=1.6, label="Mean autocorrelation")
    ax_b.fill_between(lags_h, mean_ac - sd_ac, mean_ac + sd_ac, color=C_AR, alpha=0.18,
                      label=r"$\pm1$ s.d. across qubits")
    ax_b.axhline(0.5, color="#dc2626", linestyle=":", linewidth=1.1, label="0.5 level")
    ax_b.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax_b.set_xlabel("Lag (hours)")
    ax_b.set_ylabel("Autocorrelation of $T_1$")
    ax_b.set_title("(b) Slow, forecastable structure")
    ax_b.legend(loc="upper right")
    ax_b.set_xlim(0, lags_h[-1])

    fig.tight_layout()
    out = OUT_DIR / "fig1_dynamics.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # Diagnostics for text consistency.
    below = np.where(mean_ac < 0.5)[0]
    first_below_h = (below[0] * DT_HOURS) if len(below) else lags_h[-1]
    print(f"[fig1] ACF stays >0.5 until ~{first_below_h:.1f} h -> {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 + TABLE 1 — forecasting baselines (live, 5 seeds, leak-free 80/20)
# ─────────────────────────────────────────────────────────────────────────────
def _forecast_one_seed(seed: int):
    Xs_list, ys_list, _ = pooled_windows(seed)
    # Leak-free per-qubit 80/20 chronological split, then pool.
    Xtr, ytr, Xte, yte = [], [], [], []
    for Xs, ys in zip(Xs_list, ys_list):
        n = len(Xs)
        n_tr = int(n * 0.8)
        Xtr.append(Xs[:n_tr]); ytr.append(ys[:n_tr])
        Xte.append(Xs[n_tr:]); yte.append(ys[n_tr:])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte)

    # persistence: carry last observed T1 forward across the horizon.
    pers = Xte[:, -1, 0][:, None].repeat(HORIZON, axis=1)
    # climatology: training-mean horizon trajectory.
    clim = ytr.mean(axis=0)[None, :].repeat(len(yte), axis=0)
    # AR-ridge on 32-step T1 history.
    ar = Ridge(alpha=5.0).fit(Xtr[:, :, 0], ytr).predict(Xte[:, :, 0])
    # multivariate ridge on flattened 32x7 window.
    mv = Ridge(alpha=5.0).fit(Xtr.reshape(len(Xtr), -1), ytr).predict(Xte.reshape(len(Xte), -1))

    mae = lambda p: np.abs(yte - p).mean(axis=0)  # noqa: E731
    return mae(pers), mae(clim), mae(ar), mae(mv)


def figure2_and_table1():
    acc = {"pers": [], "clim": [], "ar": [], "mv": []}
    for seed in FORECAST_SEEDS:
        p, c, a, m = _forecast_one_seed(seed)
        acc["pers"].append(p); acc["clim"].append(c); acc["ar"].append(a); acc["mv"].append(m)
    for k in acc:
        acc[k] = np.array(acc[k])  # (n_seeds, HORIZON)

    horizons = np.arange(1, HORIZON + 1)
    means = {k: acc[k].mean(0) for k in acc}
    sds = {k: acc[k].std(0) for k in acc}

    # Skill over persistence per horizon for the two learned forecasters.
    pm = means["pers"]
    skill_ar = 1 - acc["ar"] / acc["pers"]      # per-seed skill for s.d. band
    skill_mv = 1 - acc["mv"] / acc["pers"]
    skill_ar_m, skill_ar_s = skill_ar.mean(0), skill_ar.std(0)
    skill_mv_m, skill_mv_s = skill_mv.mean(0), skill_mv.std(0)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.0))

    # (a) MAE vs horizon, mean ± 1 s.d. bands.
    series = [("Persistence", "pers", C_PERS, "-"),
              ("Climatology", "clim", C_CLIM, ":"),
              ("AR-ridge ($T_1$ history)", "ar", C_AR, "--"),
              ("Multivariate ridge", "mv", C_MV, "-")]
    for label, key, color, ls in series:
        ax_a.plot(horizons, means[key], color=color, linestyle=ls, linewidth=1.7, marker="o",
                  markersize=3.5, label=label)
        ax_a.fill_between(horizons, means[key] - sds[key], means[key] + sds[key],
                          color=color, alpha=0.15)
    ax_a.set_xlabel("Forecast horizon (steps ahead)")
    ax_a.set_ylabel("Mean absolute error (µs)")
    ax_a.set_title("(a) Forecast error vs horizon")
    ax_a.legend(loc="center right")
    ax_a.set_xticks(horizons)

    # (b) Skill over persistence for the two learned forecasters.
    ax_b.plot(horizons, skill_ar_m * 100, color=C_AR, linestyle="--", linewidth=1.7,
              marker="o", markersize=3.5, label="AR-ridge")
    ax_b.fill_between(horizons, (skill_ar_m - skill_ar_s) * 100, (skill_ar_m + skill_ar_s) * 100,
                      color=C_AR, alpha=0.15)
    ax_b.plot(horizons, skill_mv_m * 100, color=C_MV, linewidth=1.7,
              marker="o", markersize=3.5, label="Multivariate ridge")
    ax_b.fill_between(horizons, (skill_mv_m - skill_mv_s) * 100, (skill_mv_m + skill_mv_s) * 100,
                      color=C_MV, alpha=0.15)
    ax_b.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax_b.set_xlabel("Forecast horizon (steps ahead)")
    ax_b.set_ylabel("Skill over persistence (%)")
    ax_b.set_title("(b) Skill widens with horizon")
    ax_b.legend(loc="lower right")
    ax_b.set_xticks(horizons)

    fig.tight_layout()
    out = OUT_DIR / "fig2_forecasting.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # ── Print Table 1 values for the manuscript ──
    def fmt(key, h):
        return f"{means[key][h]:.2f}±{sds[key][h]:.2f}"
    sk = lambda key: 1 - means[key][7] / pm[7]  # noqa: E731
    print(f"[table1] seeds={FORECAST_SEEDS}")
    print(f"  Persistence   MAE@1={fmt('pers',0)}  MAE@8={fmt('pers',7)}  skill@8=---")
    print(f"  Climatology   MAE@1={fmt('clim',0)}  MAE@8={fmt('clim',7)}  skill@8={sk('clim')*100:.0f}%")
    print(f"  AR-ridge      MAE@1={fmt('ar',0)}  MAE@8={fmt('ar',7)}  skill@8={sk('ar')*100:.0f}%")
    print(f"  Multivar      MAE@1={fmt('mv',0)}  MAE@8={fmt('mv',7)}  skill@8={sk('mv')*100:.0f}%")
    print(f"  Skill@1: AR={skill_ar_m[0]*100:.0f}%  MV={skill_mv_m[0]*100:.0f}%  -> {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — sequence-model benchmark (metrics transcribed from executed notebooks)
# ─────────────────────────────────────────────────────────────────────────────
# These values are recorded verbatim from the executed notebooks:
#   notebooks/rnn_drift_forecast.ipynb        (thermal incident detection)
#   notebooks/quantum_drift_combined.ipynb    (three-dataset average)
#   notebooks/transformer_calibration.ipynb   (periodic-regime ROC-AUC, param count)
# They are NOT re-trained in this script (training is CPU/GPU-agnostic PyTorch).
NOTEBOOK_METRICS = {
    # thermal incident detection (machine_temperature_system_failure)
    "thermal": {
        "Elman RNN":   {"roc_auc": 0.4083, "f1": 0.0000, "params": 5645},
        "LSTM":        {"roc_auc": 0.4234, "f1": 0.0000, "params": 116845},
        "GRU":         {"roc_auc": 0.7182, "f1": 0.2574, "params": 87949},
    },
    # three-dataset cross-domain average ROC-AUC
    "cross": {
        "GRU":         0.6603,
        "LSTM":        0.6278,
        "Transformer": 0.1955,
    },
    # transformer on the periodic calibration-like regime in isolation
    "transformer_periodic": 0.7987,
    # transformer parameter count (transformer_calibration notebook)
    "transformer_params": 226765,
}


def figure3():
    th = NOTEBOOK_METRICS["thermal"]
    cross = NOTEBOOK_METRICS["cross"]
    tf_periodic = NOTEBOOK_METRICS["transformer_periodic"]
    tf_params = NOTEBOOK_METRICS["transformer_params"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) Thermal incident detection: ROC-AUC and F1 grouped bars.
    models_a = ["Elman RNN", "LSTM", "GRU"]
    roc = [th[m]["roc_auc"] for m in models_a]
    f1 = [th[m]["f1"] for m in models_a]
    x = np.arange(len(models_a))
    w = 0.38
    b1 = ax_a.bar(x - w / 2, roc, w, label="ROC-AUC", color=C_GRU, alpha=0.85)
    b2 = ax_a.bar(x + w / 2, f1, w, label="F1", color=C_TRANSF, alpha=0.85)
    ax_a.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.0, label="Chance (ROC-AUC)")
    ax_a.bar_label(b1, fmt="%.2f", padding=2, fontsize=7.5)
    ax_a.bar_label(b2, fmt="%.2f", padding=2, fontsize=7.5)
    ax_a.set_xticks(x); ax_a.set_xticklabels(models_a)
    ax_a.set_ylabel("Score")
    ax_a.set_ylim(0, 0.85)
    ax_a.set_title("(a) Thermal incident detection")
    ax_a.legend(loc="upper left", fontsize=7.5)

    # (b) Three-dataset average ROC-AUC + Transformer* periodic (hatched).
    labels_b = ["GRU", "LSTM", "Transformer", "Transformer*\n(periodic)"]
    vals_b = [cross["GRU"], cross["LSTM"], cross["Transformer"], tf_periodic]
    colors_b = [C_GRU, C_LSTM, C_TRANSF, C_TRANSF]
    bars_b = ax_b.bar(labels_b, vals_b, color=colors_b, alpha=0.85, width=0.6)
    bars_b[-1].set_hatch("////")
    bars_b[-1].set_alpha(0.55)
    ax_b.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.0, label="Chance")
    ax_b.bar_label(bars_b, fmt="%.2f", padding=2, fontsize=7.5)
    ax_b.set_ylabel("ROC-AUC")
    ax_b.set_ylim(0, 0.9)
    ax_b.set_title("(b) Cross-domain ranking quality")
    ax_b.legend(loc="upper right", fontsize=7.5)
    ax_b.tick_params(axis="x", labelsize=8)

    # (c) Parameter counts (log scale).
    labels_c = ["Elman RNN", "GRU", "LSTM", "Transformer"]
    params_c = [th["Elman RNN"]["params"], th["GRU"]["params"],
                th["LSTM"]["params"], tf_params]
    colors_c = [C_RNN, C_GRU, C_LSTM, C_TRANSF]
    bars_c = ax_c.bar(labels_c, params_c, color=colors_c, alpha=0.85, width=0.6)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("Parameter count (log scale)")
    ax_c.set_title("(c) Model size")
    ax_c.yaxis.set_major_locator(LogLocator(base=10))
    ax_c.bar_label(bars_c, labels=[f"{p:,}" for p in params_c], padding=3, fontsize=7.5)
    ax_c.tick_params(axis="x", labelsize=8)
    ax_c.set_ylim(1e3, 5e5)

    fig.tight_layout()
    out = OUT_DIR / "fig3_benchmark.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] rendered from notebook metrics -> {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — reconstruction detector (live, telemetry seed 42, 8 random splits)
# ─────────────────────────────────────────────────────────────────────────────
def _reconstruction_error(Xtr_nominal, Xte, k):
    """Per-feature min-max scaling (fit on nominal windows) + rank-k PCA recon MSE."""
    n_feat = Xtr_nominal.shape[-1]
    flat_tr = Xtr_nominal.reshape(-1, n_feat)
    mn = flat_tr.min(0)
    mx = flat_tr.max(0)
    rng = np.where(mx - mn == 0, 1.0, mx - mn)
    Xtr = ((Xtr_nominal - mn) / rng).reshape(len(Xtr_nominal), -1)
    Xte = ((Xte - mn) / rng).reshape(len(Xte), -1)
    center = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - center, full_matrices=False)
    Wk = Vt[:k]
    B = Xte - center
    proj = (B @ Wk.T) @ Wk
    return ((B - proj) ** 2).mean(1)


def figure4():
    Xs_list, _, lbl_list = pooled_windows(DETECTOR_SEED)
    Xall = np.concatenate(Xs_list)
    Lall = np.concatenate(lbl_list).astype(int)
    n = len(Xall)
    n_tr = int(0.7 * n)

    # Randomised 70/30 splits; detector fit on nominal training windows only.
    rng = np.random.default_rng(0)
    splits = [rng.permutation(n) for _ in range(N_DETECTOR_SPLITS)]

    # (a) ROC curve for rank-3 detector on the first held-out split.
    tr, te = splits[0][:n_tr], splits[0][n_tr:]
    nom = tr[Lall[tr] == 0]
    err3 = _reconstruction_error(Xall[nom], Xall[te], 3)
    fpr, tpr, _ = roc_curve(Lall[te], err3)
    auc3_single = roc_auc_score(Lall[te], err3)

    # (b) ROC-AUC vs bottleneck rank, mean ± 1 s.d. over the splits.
    rank_auc = {k: [] for k in RANK_GRID}
    for sp in splits:
        tr, te = sp[:n_tr], sp[n_tr:]
        nom = tr[Lall[tr] == 0]
        if len(np.unique(Lall[te])) < 2:
            continue
        for k in RANK_GRID:
            rank_auc[k].append(roc_auc_score(Lall[te], _reconstruction_error(Xall[nom], Xall[te], k)))
    auc_mean = np.array([np.mean(rank_auc[k]) for k in RANK_GRID])
    auc_sd = np.array([np.std(rank_auc[k]) for k in RANK_GRID])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.0))

    ax_a.plot(fpr, tpr, color=C_MV, linewidth=2.0, label=f"Rank-3 detector (AUC = {auc3_single:.2f})")
    ax_a.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.0, label="Chance")
    ax_a.set_xlabel("False positive rate")
    ax_a.set_ylabel("True positive rate")
    ax_a.set_title("(a) Reconstruction detector ROC")
    ax_a.legend(loc="lower right")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1.02)

    ax_b.plot(RANK_GRID, auc_mean, color=C_AR, linewidth=1.8, marker="o", markersize=3.0,
              label="ROC-AUC")
    ax_b.fill_between(RANK_GRID, auc_mean - auc_sd, auc_mean + auc_sd, color=C_AR, alpha=0.18,
                      label=r"$\pm1$ s.d. (8 splits)")
    ax_b.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.0, label="Chance")
    ax_b.set_xlabel("Reconstruction bottleneck rank")
    ax_b.set_ylabel("Detection ROC-AUC")
    ax_b.set_title("(b) Drift lives off a low-rank manifold")
    ax_b.legend(loc="upper right")
    ax_b.set_xlim(1, SEQ_LEN)

    fig.tight_layout()
    out = OUT_DIR / "fig4_anomaly.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"[fig4] rank-3 single-split AUC={auc3_single:.4f}  "
          f"rank-1 8-split={auc_mean[0]:.4f}±{auc_sd[0]:.4f}  "
          f"rank-3 8-split={auc_mean[2]:.4f}±{auc_sd[2]:.4f} -> {out.name}")


def main():
    print(f"Writing figures to: {OUT_DIR}")
    figure1()
    figure2_and_table1()
    figure3()
    figure4()
    print("Done. 4 figures written.")


if __name__ == "__main__":
    main()
