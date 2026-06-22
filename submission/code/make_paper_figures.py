#!/usr/bin/env python3
"""
make_paper_figures.py — single reproducible script that regenerates the five
figures used in submission/main.tex (one programmatic method-overview schematic
plus four quantitative figures), styled to Nature Machine Intelligence display
conventions: vector PDF with embedded editable text (pdf.fonttype = 42),
sans-serif typeface, colour-blind-safe Okabe-Ito palette, top/right spines
removed, bold lower-case panel labels (a, b, c) for multi-panel figures, no
in-panel titles (every description lives in the LaTeX caption), and error bars or
shaded 95%/+-1 s.d. bands wherever a mean is plotted.

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

Outputs five vector PDFs into submission/figures/ and the manuscript table
bodies into submission/tables/:
    fig0_overview.pdf, fig1_dynamics.pdf, fig2_forecasting.pdf,
    fig3_benchmark.pdf, fig4_anomaly.pdf
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Determinism: pin the build epoch BEFORE matplotlib is imported so its PDF
# backend stamps a fixed CreationDate, making every figure PDF byte-identical
# across runs (combined with metadata={"CreationDate": None} at savefig time).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")

import matplotlib

matplotlib.use("Agg")  # headless / CPU-only
import matplotlib.pyplot as plt
from cycler import cycler
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
TABLE_DIR = SUBMISSION_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = SUBMISSION_ROOT / "code" / "generated_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Nature Machine Intelligence display styling ──────────────────────────────
# Conventions applied throughout (Nature Portfolio artwork & formatting guidance):
#   * Vector PDF with embedded, editable text (pdf.fonttype = 42).
#   * Sans-serif typeface (Arial/Helvetica family), 7-8 pt range.
#   * No in-panel titles -- every description lives in the LaTeX caption.
#   * Bold lower-case panel labels (a, b, c) for multi-panel figures.
#   * Colour-blind-safe qualitative palette (Okabe & Ito / Wong, Nat. Methods
#     2011): safe under deuteranopia/protanopia, avoids the red--green trap.
#   * Error bars / 95%-shaded CI wherever a mean is plotted; the caption states
#     n and what the interval represents.
#   * Top/right spines removed for an uncluttered Nature-style frame.
#
# Okabe-Ito colour-blind-safe qualitative palette.
NMI_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,   # embed TrueType so text stays selectable/editable
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",  # keep in-figure math sans-serif
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.0,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.prop_cycle": cycler(color=NMI_PALETTE),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    # Deterministic vector output (no per-run hashing or timestamp drift).
    "svg.hashsalt": "qdriftforecast",
})

# Consistent Okabe-Ito assignments across figures.
C_PERS = "#000000"   # persistence (black)
C_CLIM = "#E69F00"   # climatology (orange)
C_AR = "#0072B2"     # AR-ridge (blue)
C_MV = "#009E73"     # multivariate ridge (bluish green)
C_RNN = "#56B4E9"    # Elman RNN (sky blue)
C_LSTM = "#E69F00"   # LSTM (orange)
C_GRU = "#009E73"    # GRU (bluish green)
C_TRANSF = "#CC79A7" # Transformer (reddish purple)
C_CHANCE = "#D55E00" # chance / threshold reference (vermillion)
QUBIT_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.03) -> None:
    """Bold lower-case panel label in the upper-left, Nature convention."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


def save_figure(fig, name: str):
    """Write *fig* as a byte-deterministic vector PDF, then close it.

    Two ingredients make the output reproducible to the byte:
      * ``SOURCE_DATE_EPOCH`` (pinned at import) fixes the PDF ``CreationDate``;
      * ``metadata={"CreationDate": None}`` removes the per-run ``ModDate``.
    Vector PDF (not raster PNG) is the Nature-portfolio house format for line and
    bar charts: text stays selectable and the art scales without pixelation.
    """
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return out

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

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) T1 trajectories for 5 qubits over 100 h, with 50 µs threshold.
    for q in range(N_QUBITS):
        sub = df[df["qubit_id"] == q].sort_values("timestamp_hr")
        ax_a.plot(sub["timestamp_hr"].values, sub["T1_us"].values,
                  color=QUBIT_COLORS[q], linewidth=1.1, alpha=0.9, label=f"qubit {q}")
    ax_a.axhline(T1_THRESHOLD, color=C_CHANCE, linestyle="--", linewidth=1.1,
                 label=f"{T1_THRESHOLD:.0f} µs threshold")
    ax_a.set_xlabel("time (hours)")
    ax_a.set_ylabel(r"$T_1$ relaxation time (µs)")
    ax_a.legend(ncol=3, loc="lower center", fontsize=5.6, handlelength=1.1,
                columnspacing=0.9, borderaxespad=0.3)
    ax_a.set_xlim(0, (N_STEPS - 1) * DT_HOURS)
    ax_a.set_ylim(45, None)  # leave room below the curves for the legend
    panel_label(ax_a, "a")

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
    ax_b.plot(lags_h, mean_ac, color=C_AR, linewidth=1.6, label="mean autocorrelation")
    ax_b.fill_between(lags_h, mean_ac - sd_ac, mean_ac + sd_ac, color=C_AR, alpha=0.18,
                      linewidth=0, label=r"$\pm1$ s.d. across qubits")
    ax_b.axhline(0.5, color=C_CHANCE, linestyle=":", linewidth=1.1, label="0.5 level")
    ax_b.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax_b.set_xlabel("lag (hours)")
    ax_b.set_ylabel("autocorrelation of $T_1$")
    ax_b.legend(loc="upper right")
    ax_b.set_xlim(0, lags_h[-1])
    panel_label(ax_b, "b")

    fig.tight_layout()
    out = save_figure(fig, "fig1_dynamics.pdf")

    # Diagnostics for text consistency.
    below = np.where(mean_ac < 0.5)[0]
    first_below_h = (below[0] * DT_HOURS) if len(below) else lags_h[-1]
    (DATA_DIR / "coherence_dynamics_summary.json").write_text(json.dumps({
        "seed": DETECTOR_SEED,
        "n_qubits": N_QUBITS,
        "n_steps": N_STEPS,
        "sampling_hours": DT_HOURS,
        "t1_threshold_us": T1_THRESHOLD,
        "first_mean_autocorrelation_below_0_5_hours": first_below_h,
    }, indent=2) + "\n")
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

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # (a) MAE vs horizon, mean ± 1 s.d. bands.
    series = [("persistence", "pers", C_PERS, "-"),
              ("climatology", "clim", C_CLIM, ":"),
              ("AR-ridge ($T_1$ history)", "ar", C_AR, "--"),
              ("multivariate ridge", "mv", C_MV, "-")]
    for label, key, color, ls in series:
        ax_a.plot(horizons, means[key], color=color, linestyle=ls, linewidth=1.5, marker="o",
                  markersize=3.0, label=label)
        ax_a.fill_between(horizons, means[key] - sds[key], means[key] + sds[key],
                          color=color, alpha=0.15, linewidth=0)
    ax_a.set_xlabel("forecast horizon (steps ahead)")
    ax_a.set_ylabel("mean absolute error (µs)")
    ax_a.legend(loc="center right", handlelength=1.6)
    ax_a.set_xticks(horizons)
    panel_label(ax_a, "a")

    # (b) Skill over persistence for the two learned forecasters.
    ax_b.plot(horizons, skill_ar_m * 100, color=C_AR, linestyle="--", linewidth=1.5,
              marker="o", markersize=3.0, label="AR-ridge")
    ax_b.fill_between(horizons, (skill_ar_m - skill_ar_s) * 100, (skill_ar_m + skill_ar_s) * 100,
                      color=C_AR, alpha=0.15, linewidth=0)
    ax_b.plot(horizons, skill_mv_m * 100, color=C_MV, linewidth=1.5,
              marker="o", markersize=3.0, label="multivariate ridge")
    ax_b.fill_between(horizons, (skill_mv_m - skill_mv_s) * 100, (skill_mv_m + skill_mv_s) * 100,
                      color=C_MV, alpha=0.15, linewidth=0)
    ax_b.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax_b.set_xlabel("forecast horizon (steps ahead)")
    ax_b.set_ylabel("skill over persistence (%)")
    ax_b.legend(loc="lower right", handlelength=1.6)
    ax_b.set_xticks(horizons)
    panel_label(ax_b, "b")

    fig.tight_layout()
    out = save_figure(fig, "fig2_forecasting.pdf")

    # ── Print Table 1 values for the manuscript ──
    def fmt(key, h):
        return f"{means[key][h]:.2f}±{sds[key][h]:.2f}"
    sk = lambda key: 1 - means[key][7] / pm[7]  # noqa: E731
    table_rows = [
        ("Persistence", "pers", "---", False),
        ("Climatology", "clim", f"{sk('clim')*100:.0f}\\%", False),
        (r"AR-ridge (\T\ history)", "ar", f"{sk('ar')*100:.0f}\\%", False),
        ("Multivariate ridge", "mv", f"{sk('mv')*100:.0f}\\%", True),
    ]
    table_lines = [
        r"\begin{tabular}{lccc}",
        r"  \toprule",
        r"  Forecaster & MAE @ 1 step (\si{\micro\second}) & MAE @ 8 steps (\si{\micro\second}) & Skill @ 8 steps \\",
        r"  \midrule",
    ]
    for label, key, skill, bold in table_rows:
        first = f"{means[key][0]:.2f} \\pm {sds[key][0]:.2f}"
        eighth = f"{means[key][7]:.2f} \\pm {sds[key][7]:.2f}"
        if bold:
            first = rf"\mathbf{{{first}}}"
            eighth = rf"\mathbf{{{eighth}}}"
            skill = rf"$\mathbf{{{skill}}}$"
        table_lines.append(f"  {label:<22} & ${first}$ & ${eighth}$ & {skill} \\\\")
    table_lines += [r"  \bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "forecasting_benchmark.tex").write_text("\n".join(table_lines) + "\n")
    (DATA_DIR / "forecasting_benchmark.json").write_text(json.dumps({
        "seeds": FORECAST_SEEDS,
        "horizons": horizons.tolist(),
        "mae_mean": {k: v.tolist() for k, v in means.items()},
        "mae_sd": {k: v.tolist() for k, v in sds.items()},
        "skill_ar_mean": skill_ar_m.tolist(),
        "skill_ar_sd": skill_ar_s.tolist(),
        "skill_multivariate_mean": skill_mv_m.tolist(),
        "skill_multivariate_sd": skill_mv_s.tolist(),
    }, indent=2) + "\n")
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

CROSS_DOMAIN_METRICS = {
    "GRU": {"mae": 1337.3, "rmse": 1628.8, "roc_auc": 0.6603},
    "LSTM": {"mae": 1528.8, "rmse": 1895.0, "roc_auc": 0.6278},
    "Transformer": {"mae": 1436.2, "rmse": 1791.6, "roc_auc": 0.1955},
}


def figure3():
    th = NOTEBOOK_METRICS["thermal"]
    cross = NOTEBOOK_METRICS["cross"]
    tf_periodic = NOTEBOOK_METRICS["transformer_periodic"]
    tf_params = NOTEBOOK_METRICS["transformer_params"]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.2, 2.7))

    # (a) Thermal incident detection: ROC-AUC and F1 grouped bars.
    models_a = ["Elman RNN", "LSTM", "GRU"]
    roc = [th[m]["roc_auc"] for m in models_a]
    f1 = [th[m]["f1"] for m in models_a]
    x = np.arange(len(models_a))
    w = 0.38
    b1 = ax_a.bar(x - w / 2, roc, w, label="ROC-AUC", color=C_GRU)
    b2 = ax_a.bar(x + w / 2, f1, w, label="F1", color=C_TRANSF)
    ax_a.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.0, label="chance (ROC-AUC)")
    ax_a.bar_label(b1, fmt="%.2f", padding=2, fontsize=6.0)
    ax_a.bar_label(b2, fmt="%.2f", padding=2, fontsize=6.0)
    ax_a.set_xticks(x); ax_a.set_xticklabels(models_a, rotation=15, ha="right")
    ax_a.set_ylabel("score")
    ax_a.set_ylim(0, 0.85)
    ax_a.legend(loc="upper left", fontsize=6.0, handlelength=1.2)
    panel_label(ax_a, "a")

    # (b) Three-dataset average ROC-AUC + Transformer* periodic (hatched).
    labels_b = ["GRU", "LSTM", "Transformer", "Transformer*\n(periodic)"]
    vals_b = [cross["GRU"], cross["LSTM"], cross["Transformer"], tf_periodic]
    colors_b = [C_GRU, C_LSTM, C_TRANSF, C_TRANSF]
    bars_b = ax_b.bar(labels_b, vals_b, color=colors_b, width=0.6)
    bars_b[-1].set_hatch("////")
    bars_b[-1].set_alpha(0.55)
    ax_b.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.0, label="chance")
    ax_b.bar_label(bars_b, fmt="%.2f", padding=2, fontsize=6.0)
    ax_b.set_ylabel("ROC-AUC")
    ax_b.set_ylim(0, 0.95)
    ax_b.legend(loc="upper left", fontsize=6.0, handlelength=1.2)
    ax_b.tick_params(axis="x", labelsize=6.0, rotation=20)
    panel_label(ax_b, "b")

    # (c) Parameter counts (log scale).
    labels_c = ["Elman RNN", "GRU", "LSTM", "Transformer"]
    params_c = [th["Elman RNN"]["params"], th["GRU"]["params"],
                th["LSTM"]["params"], tf_params]
    colors_c = [C_RNN, C_GRU, C_LSTM, C_TRANSF]
    bars_c = ax_c.bar(labels_c, params_c, color=colors_c, width=0.6)
    ax_c.set_yscale("log")
    ax_c.set_ylabel("parameter count (log scale)")
    ax_c.yaxis.set_major_locator(LogLocator(base=10))
    ax_c.bar_label(bars_c, labels=[f"{p:,}" for p in params_c], padding=3, fontsize=6.0)
    ax_c.tick_params(axis="x", labelsize=6.0, rotation=20)
    ax_c.set_ylim(1e3, 5e5)
    panel_label(ax_c, "c")

    fig.tight_layout()
    out = save_figure(fig, "fig3_benchmark.pdf")
    thermal_lines = [
        r"\begin{tabular}{lccccr}",
        r"  \toprule",
        r"  Model & MAE (\si{\micro\second}) & RMSE (\si{\micro\second}) & F1 & ROC-AUC & Params \\",
        r"  \midrule",
        r"  Elman RNN   & 61.37 & 64.93 & 0.000 & 0.408 & \textbf{\num{5645}} \\",
        r"  LSTM        & \textbf{51.48} & \textbf{55.00} & 0.000 & 0.423 & \num{116845} \\",
        r"  GRU         & 51.79 & 55.35 & \textbf{0.257} & \textbf{0.718} & \num{87949} \\",
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    (TABLE_DIR / "thermal_benchmark.tex").write_text("\n".join(thermal_lines) + "\n")
    cross_lines = [
        r"\begin{tabular}{lccc@{\hskip 2.4em}lc}",
        r"  \toprule",
        r"  \multicolumn{4}{c}{Three-dataset average} & \multicolumn{2}{c}{Periodic regime} \\",
        r"  \cmidrule(r){1-4}\cmidrule(l){5-6}",
        r"  Model & MAE & RMSE & ROC-AUC & Model & ROC-AUC \\",
        r"  \midrule",
        r"  GRU         & \textbf{1337.3} & \textbf{1628.8} & \textbf{0.660} & Transformer & \textbf{0.799} \\",
        r"  LSTM        & 1528.8 & 1895.0 & 0.628 & GRU (avg.)   & 0.660 \\",
        r"  Transformer & 1436.2 & 1791.6 & 0.196 & ---          & ---   \\",
        r"  \bottomrule",
        r"\end{tabular}",
    ]
    (TABLE_DIR / "cross_domain_benchmark.tex").write_text("\n".join(cross_lines) + "\n")
    (DATA_DIR / "sequence_notebook_metrics.json").write_text(json.dumps({
        "thermal": th,
        "cross_domain": CROSS_DOMAIN_METRICS,
        "transformer_periodic_roc_auc": tf_periodic,
        "transformer_params": tf_params,
    }, indent=2) + "\n")
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

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax_a.plot(fpr, tpr, color=C_MV, linewidth=1.8, label=f"rank-3 detector (AUC = {auc3_single:.2f})")
    ax_a.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1.0, label="chance")
    ax_a.set_xlabel("false positive rate")
    ax_a.set_ylabel("true positive rate")
    ax_a.legend(loc="lower right")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1.02)
    panel_label(ax_a, "a")

    ax_b.plot(RANK_GRID, auc_mean, color=C_AR, linewidth=1.6, marker="o", markersize=3.0,
              label="ROC-AUC")
    ax_b.fill_between(RANK_GRID, auc_mean - auc_sd, auc_mean + auc_sd, color=C_AR, alpha=0.18,
                      linewidth=0, label=r"$\pm1$ s.d. (8 splits)")
    ax_b.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.0, label="chance")
    ax_b.set_xlabel("reconstruction bottleneck rank")
    ax_b.set_ylabel("detection ROC-AUC")
    ax_b.legend(loc="upper right")
    ax_b.set_xlim(1, SEQ_LEN)
    panel_label(ax_b, "b")

    fig.tight_layout()
    out = save_figure(fig, "fig4_anomaly.pdf")
    (DATA_DIR / "reconstruction_rank_sweep.json").write_text(json.dumps({
        "seed": DETECTOR_SEED,
        "n_splits": N_DETECTOR_SPLITS,
        "rank_grid": RANK_GRID,
        "rank_auc_mean": auc_mean.tolist(),
        "rank_auc_sd": auc_sd.tolist(),
        "rank3_single_split_auc": float(auc3_single),
    }, indent=2) + "\n")

    print(f"[fig4] rank-3 single-split AUC={auc3_single:.4f}  "
          f"rank-1 8-split={auc_mean[0]:.4f}±{auc_sd[0]:.4f}  "
          f"rank-3 8-split={auc_mean[2]:.4f}±{auc_sd[2]:.4f} -> {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 0 — method-overview schematic (programmatic boxes + arrows)
# ─────────────────────────────────────────────────────────────────────────────
def _box(ax, xy, w, h, text, fc, ec="#222222"):
    """Draw a rounded method-schematic box with centred wrapped text.

    Returns the (right-edge, left-edge) anchor points used to connect arrows.
    """
    from matplotlib.patches import FancyBboxPatch

    box = FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=7.0, zorder=5)
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                                shrinkA=2, shrinkB=2))


def figure0_overview():
    """Programmatic method-overview schematic (the NMI 'Figure 1' convention).

    A left-to-right pipeline: a physically motivated multi-qubit telemetry
    generator emits a multivariate, mean-reverting calibration stream; sliding
    windows feed (i) learned forecasters that predict the eight-step T1 horizon,
    (ii) sequence-model families benchmarked for incident detection, and (iii) a
    low-rank reconstruction detector that flags off-manifold drift -- all under a
    single leakage-free protocol scored by horizon-stable MAE and threshold-free
    ROC-AUC. No in-plot title; description lives in the LaTeX caption.
    """
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Light Okabe-Ito-tinted fills for the pipeline stages.
    blue, green, orange, purple, grey = (
        "#D6E6F2", "#D6EFE3", "#FBE6D4", "#ECDCE9", "#ECECEC",
    )

    # Stage 1: telemetry generator (left, full height anchor).
    r0, _ = _box(ax, (0.010, 0.355), 0.165, 0.30,
                 "multi-qubit\ntelemetry generator\n($Q=5$, 7 channels)", blue)
    # Stage 2: sliding windows + leak-free split.
    r1, l1 = _box(ax, (0.215, 0.355), 0.160, 0.30,
                  "sliding windows\n$32\\!\\to\\!8$,\nleak-free split", green)

    # Three parallel analysis branches (top / middle / bottom).
    r2t, l2t = _box(ax, (0.430, 0.700), 0.230, 0.235,
                    "forecasters\n(persistence, climatology,\nAR / multivariate ridge)", orange)
    r2m, l2m = _box(ax, (0.430, 0.380), 0.230, 0.235,
                    "sequence models\n(RNN, LSTM, GRU,\nTransformer)", purple)
    r2b, l2b = _box(ax, (0.430, 0.060), 0.230, 0.235,
                    "reconstruction detector\n(rank-$k$ nominal subspace)", grey)

    # Outcome boxes (right).
    r3t, l3t = _box(ax, (0.730, 0.700), 0.260, 0.235,
                    "horizon-stable MAE\n($72\\%$ skill at 8 steps)", orange)
    r3m, l3m = _box(ax, (0.730, 0.380), 0.260, 0.235,
                    "objective-aware\nROC-AUC / F1", purple)
    r3b, l3b = _box(ax, (0.730, 0.060), 0.260, 0.235,
                    "off-manifold\ndrift score", grey)

    # Arrows: generator -> windows -> fan out to three branches -> outcomes.
    _arrow(ax, r0, l1)
    fan = (r1[0], r1[1])
    for lt in (l2t, l2m, l2b):
        _arrow(ax, fan, lt)
    _arrow(ax, r2t, l3t)
    _arrow(ax, r2m, l3m)
    _arrow(ax, r2b, l3b)

    # Top/bottom annotations: the unifying protocol and theory.
    ax.text(0.5, 0.985,
            "single leakage-free protocol  $\\bullet$  fixed seeds, CPU-reproducible",
            ha="center", va="center", fontsize=6.6, color="#333333")
    ax.text(0.5, 0.012,
            "predictability from autocorrelation  $\\bullet$  drift is an off-manifold,"
            " low-rank phenomenon",
            ha="center", va="center", fontsize=6.4, color="#555555")

    fig.tight_layout()
    out = save_figure(fig, "fig0_overview.pdf")
    print(f"[fig0] method-overview schematic -> {out.name}")


def main():
    print(f"Writing figures to: {OUT_DIR}")
    figure0_overview()
    figure1()
    figure2_and_table1()
    figure3()
    figure4()
    print("Done. 5 figures written (1 schematic + 4 quantitative).")


if __name__ == "__main__":
    main()
