#!/usr/bin/env python3
"""make_paper_figures.py -- regenerate every figure and table in main.tex.

All figures and tables are computed *live* from the installed ``qdriftforecast``
package on commodity CPU hardware, deterministically from fixed integer seeds.
The pipeline produces (into ``submission/figures/`` and ``submission/tables/``):

    fig0_overview.pdf      method-overview schematic of the C*-kernel pipeline
    fig1_dynamics.pdf      coherence dynamics, autocorrelation, drift signature
    fig2_forecasting.pdf   forecasting MAE/skill vs horizon; skill vs truncation
    fig3_noncommutative.pdf  noncommutative detection: AUC vs n, (n,k) grid, bars
    fig4_detection.pdf     ROC curves and the over-complete collapse

    forecasting_benchmark.tex   forecasting baselines
    detection_benchmark.tex     correlated-drift detector comparison
    truncation_sweep.tex        detection AUC vs truncation order n

plus machine-readable JSON summaries in ``submission/code/generated_data/``.

Styled to Nature Machine Intelligence display conventions: vector PDF with
embedded editable text, sans-serif typeface, colour-blind-safe Okabe-Ito palette,
top/right spines removed, bold lower-case panel labels, error/shaded bands.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "1700000000")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import LogLocator  # noqa: F401  (kept for optional log axes)
from sklearn.metrics import roc_auc_score, roc_curve

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]

from qdriftforecast.data import (  # noqa: E402
    FEATURE_COLS,
    generate_correlated_drift_windows,
    generate_synthetic_dataset,
)
from qdriftforecast.detection import (  # noqa: E402
    baseline_comparison,
    feature_backend,
    rank_truncation_grid,
    reconstruction_scores,
    truncation_sweep,
)
from qdriftforecast.forecasting import (  # noqa: E402
    cstk_forecast_skill,
    forecast_baselines,
)
from qdriftforecast.kernels import cst_feature_map  # noqa: E402

# ── Global configuration (fixed seeds) ───────────────────────────────────────
N_QUBITS = 5
N_STEPS = 200
DT_HOURS = 0.5
SEQ_LEN = 32
HORIZON = 8
T1_THRESHOLD = 50.0

FORECAST_SEEDS = [0, 1, 2, 3, 4]
DYN_SEED = 42

# Correlated-drift detection benchmark.
DRIFT_N_EACH = 700
DRIFT_SEED = 0
N_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
K_GRID = [1, 2, 3, 5, 8, 12, 16]
N_GRID_HEAT = [1, 2, 3, 4, 6, 8]
K_GRID_HEAT = [1, 2, 3, 5, 8, 12]
N_SPLITS = 8
N_CST = 2          # informative-lag-optimal truncation order (empirical optimum)
N_PERIODIC = 2

OUT_DIR = SUBMISSION_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = SUBMISSION_ROOT / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = SUBMISSION_ROOT / "code" / "generated_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Nature Machine Intelligence display styling ──────────────────────────────
NMI_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
               "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "axes.grid": False, "lines.linewidth": 1.3, "lines.markersize": 3.0,
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.prop_cycle": cycler(color=NMI_PALETTE),
    "figure.dpi": 300, "savefig.dpi": 300, "svg.hashsalt": "qdriftforecast",
})

C_PERS = "#000000"; C_CLIM = "#E69F00"; C_AR = "#0072B2"; C_MV = "#009E73"
C_CST = "#0072B2"; C_COMM = "#000000"; C_PER = "#CC79A7"; C_RAW = "#888888"
C_CHANCE = "#D55E00"; C_PEAK = "#009E73"
QUBIT_COLORS = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"]


def panel_label(ax, letter, x=-0.16, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="right")


def save_figure(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 0 -- method-overview schematic
# ═════════════════════════════════════════════════════════════════════════════
def _box(ax, xy, w, h, text, fc, ec="#222222"):
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((xy[0], xy[1]), w, h,
                         boxstyle="round,pad=0.012,rounding_size=0.02",
                         linewidth=1.0, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=7.0, zorder=5)
    return (xy[0] + w, xy[1] + h / 2), (xy[0], xy[1] + h / 2)


def _arrow(ax, p0, p1):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#444444",
                                shrinkA=2, shrinkB=2))


def figure0_overview():
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    blue, green, orange, purple, grey = ("#D6E6F2", "#D6EFE3", "#FBE6D4",
                                         "#ECDCE9", "#ECECEC")
    r0, _ = _box(ax, (0.010, 0.355), 0.170, 0.30,
                 "multi-qubit\ntelemetry\n($Q=5$, 7 channels)", blue)
    r1, l1 = _box(ax, (0.220, 0.355), 0.190, 0.30,
                  "causal spectral-\ntruncation kernel\n$K_n=\\sum_{\\tau<n}w_\\tau X^{\\top}S^{\\tau}Y$", green)
    r2t, l2t = _box(ax, (0.470, 0.700), 0.250, 0.235,
                    "commutative ($n{=}1$)\ninstantaneous monitor", purple)
    r2b, l2b = _box(ax, (0.470, 0.070), 0.250, 0.235,
                    "noncommutative ($n{>}1$)\nlagged cross-channel monitor", orange)
    r3t, l3t = _box(ax, (0.770, 0.700), 0.220, 0.235,
                    "forecasting +\nmarginal drift", purple)
    r3b, l3b = _box(ax, (0.770, 0.070), 0.220, 0.235,
                    "correlated\ncrosstalk drift", orange)
    _arrow(ax, r0, l1)
    fan = (r1[0], r1[1])
    _arrow(ax, fan, l2t); _arrow(ax, fan, l2b)
    _arrow(ax, r2t, l3t); _arrow(ax, r2b, l3b)
    ax.text(0.5, 0.985,
            "truncation order $n$ controls noncommutativity  $\\bullet$  fixed seeds, CPU-reproducible",
            ha="center", va="center", fontsize=6.6, color="#333333")
    ax.text(0.5, 0.012,
            "commutative monitors are blind to lagged cross-channel drift  $\\bullet$  an optimal $n$ certifies detectability",
            ha="center", va="center", fontsize=6.4, color="#555555")
    fig.tight_layout()
    out = save_figure(fig, "fig0_overview.pdf")
    print(f"[fig0] overview schematic -> {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 -- coherence dynamics, autocorrelation, noncommutative drift signature
# ═════════════════════════════════════════════════════════════════════════════
def figure1():
    df = generate_synthetic_dataset(n_qubits=N_QUBITS, n_steps=N_STEPS, seed=DYN_SEED)
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.2, 2.6))

    # (a) T1 trajectories.
    for q in range(N_QUBITS):
        sub = df[df["qubit_id"] == q].sort_values("timestamp_hr")
        ax_a.plot(sub["timestamp_hr"].values, sub["T1_us"].values,
                  color=QUBIT_COLORS[q], linewidth=1.0, alpha=0.9, label=f"qubit {q}")
    ax_a.axhline(T1_THRESHOLD, color=C_CHANCE, linestyle="--", linewidth=1.0)
    ax_a.set_xlabel("time (hours)")
    ax_a.set_ylabel(r"$T_1$ relaxation time (µs)")
    ax_a.legend(ncol=2, loc="lower center", fontsize=5.2, handlelength=1.0,
                columnspacing=0.8, borderaxespad=0.2)
    ax_a.set_xlim(0, (N_STEPS - 1) * DT_HOURS); ax_a.set_ylim(45, None)
    panel_label(ax_a, "a")

    # (b) Autocorrelation of detrended T1.
    max_lag = 40
    acfs = []
    for q in range(N_QUBITS):
        s = df[df["qubit_id"] == q].sort_values("timestamp_hr")["T1_us"].values.astype(float)
        idx = np.arange(len(s))
        s = s - np.polyval(np.polyfit(idx, s, 1), idx)
        s = s - s.mean()
        denom = (s * s).sum()
        ac = np.array([(s[:len(s) - k] * s[k:]).sum() / denom for k in range(max_lag)])
        acfs.append(ac)
    acfs = np.array(acfs)
    lags_h = np.arange(max_lag) * DT_HOURS
    mean_ac, sd_ac = acfs.mean(0), acfs.std(0)
    ax_b.plot(lags_h, mean_ac, color=C_AR, linewidth=1.6)
    ax_b.fill_between(lags_h, mean_ac - sd_ac, mean_ac + sd_ac, color=C_AR,
                      alpha=0.18, linewidth=0)
    ax_b.axhline(0.5, color=C_CHANCE, linestyle=":", linewidth=1.0)
    ax_b.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax_b.set_xlabel("lag (hours)")
    ax_b.set_ylabel(r"autocorrelation of $T_1$")
    ax_b.set_xlim(0, lags_h[-1])
    panel_label(ax_b, "b")

    # (c) Noncommutative drift signature: lagged cross-channel correlation gap.
    X, y = generate_correlated_drift_windows(n_each=DRIFT_N_EACH, seed=DRIFT_SEED)
    C = len(FEATURE_COLS)
    nmax = 4
    f = cst_feature_map(X, nmax).reshape(len(X), C, C, nmax)
    gap = np.abs(f[y == 1].mean(0) - f[y == 0].mean(0))  # (C, C, nmax)
    # mean |mean-gap| per lag tau, summarising where the drift signal lives.
    per_lag = gap.reshape(C * C, nmax).mean(0)
    bars = ax_c.bar(np.arange(nmax), per_lag, color=[C_CHANCE] + [C_PEAK] * (nmax - 1),
                    width=0.6)
    ax_c.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.0)
    ax_c.set_xticks(np.arange(nmax))
    ax_c.set_xticklabels([f"$\\tau={t}$" for t in range(nmax)])
    ax_c.set_xlabel("cross-channel correlation lag")
    ax_c.set_ylabel("nominal–drift mean gap")
    panel_label(ax_c, "c")

    fig.tight_layout()
    out = save_figure(fig, "fig1_dynamics.pdf")
    below = np.where(mean_ac < 0.5)[0]
    first_below_h = (below[0] * DT_HOURS) if len(below) else lags_h[-1]
    (DATA_DIR / "coherence_dynamics_summary.json").write_text(json.dumps({
        "seed": DYN_SEED, "n_qubits": N_QUBITS, "n_steps": N_STEPS,
        "sampling_hours": DT_HOURS, "t1_threshold_us": T1_THRESHOLD,
        "first_mean_autocorrelation_below_0_5_hours": float(first_below_h),
        "drift_signature_gap_per_lag": per_lag.tolist(),
    }, indent=2) + "\n")
    print(f"[fig1] ACF>0.5 until ~{first_below_h:.1f} h; lag-0 gap={per_lag[0]:.3f} "
          f"lag-1 gap={per_lag[1]:.3f} -> {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 + Table 1 -- forecasting baselines (live, 5 seeds) + skill vs n
# ═════════════════════════════════════════════════════════════════════════════
def figure2_and_table1():
    acc = {"pers": [], "clim": [], "ar": [], "mv": []}
    for seed in FORECAST_SEEDS:
        r = forecast_baselines(seed)
        for k in acc:
            acc[k].append(r[k])
    for k in acc:
        acc[k] = np.array(acc[k])
    horizons = np.arange(1, HORIZON + 1)
    means = {k: acc[k].mean(0) for k in acc}
    sds = {k: acc[k].std(0) for k in acc}
    pm = means["pers"]
    skill_ar = 1 - acc["ar"] / acc["pers"]
    skill_mv = 1 - acc["mv"] / acc["pers"]
    skill_ar_m, skill_ar_s = skill_ar.mean(0), skill_ar.std(0)
    skill_mv_m, skill_mv_s = skill_mv.mean(0), skill_mv.std(0)

    # Skill vs truncation order n (CST forecaster), 5-seed mean.
    sk_n = {}
    for seed in FORECAST_SEEDS:
        r = cstk_forecast_skill(seed, n_grid=(1, 2, 3, 4))
        for n in r:
            sk_n.setdefault(n, []).append(r[n])
    n_vals = sorted(sk_n)
    sk_n_m = np.array([np.mean(sk_n[n]) for n in n_vals])
    sk_n_s = np.array([np.std(sk_n[n]) for n in n_vals])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.2, 2.6))
    series = [("persistence", "pers", C_PERS, "-"),
              ("climatology", "clim", C_CLIM, ":"),
              ("AR-ridge", "ar", C_AR, "--"),
              ("multivariate ridge", "mv", C_MV, "-")]
    for label, key, color, ls in series:
        ax_a.plot(horizons, means[key], color=color, linestyle=ls, linewidth=1.5,
                  marker="o", markersize=3.0, label=label)
        ax_a.fill_between(horizons, means[key] - sds[key], means[key] + sds[key],
                          color=color, alpha=0.15, linewidth=0)
    ax_a.set_xlabel("forecast horizon (steps ahead)")
    ax_a.set_ylabel("mean absolute error (µs)")
    ax_a.legend(loc="center right", handlelength=1.6, fontsize=6.2)
    ax_a.set_xticks(horizons)
    panel_label(ax_a, "a")

    ax_b.plot(horizons, skill_ar_m * 100, color=C_AR, linestyle="--", linewidth=1.5,
              marker="o", markersize=3.0, label="AR-ridge")
    ax_b.fill_between(horizons, (skill_ar_m - skill_ar_s) * 100,
                      (skill_ar_m + skill_ar_s) * 100, color=C_AR, alpha=0.15, linewidth=0)
    ax_b.plot(horizons, skill_mv_m * 100, color=C_MV, linewidth=1.5, marker="o",
              markersize=3.0, label="multivariate ridge")
    ax_b.fill_between(horizons, (skill_mv_m - skill_mv_s) * 100,
                      (skill_mv_m + skill_mv_s) * 100, color=C_MV, alpha=0.15, linewidth=0)
    ax_b.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax_b.set_xlabel("forecast horizon (steps ahead)")
    ax_b.set_ylabel("skill over persistence (%)")
    ax_b.legend(loc="lower right", handlelength=1.6)
    ax_b.set_xticks(horizons)
    panel_label(ax_b, "b")

    # (c) Forecast skill at 8 steps vs truncation order n (flat: commutative suffices).
    ax_c.plot(n_vals, sk_n_m * 100, color=C_CST, linewidth=1.5, marker="s", markersize=3.5)
    ax_c.fill_between(n_vals, (sk_n_m - sk_n_s) * 100, (sk_n_m + sk_n_s) * 100,
                      color=C_CST, alpha=0.15, linewidth=0)
    ax_c.set_xlabel("truncation order $n$")
    ax_c.set_ylabel("8-step skill (%)")
    ax_c.set_xticks(n_vals)
    lo = (sk_n_m.min() - 0.05) * 100
    hi = (sk_n_m.max() + 0.05) * 100
    ax_c.set_ylim(lo, hi)
    panel_label(ax_c, "c")

    fig.tight_layout()
    out = save_figure(fig, "fig2_forecasting.pdf")

    def sk(key):
        return 1 - means[key][7] / pm[7]
    rows = [("Persistence", "pers", "---", False),
            ("Climatology", "clim", f"{sk('clim')*100:.0f}\\%", False),
            (r"AR-ridge (\T\ history)", "ar", f"{sk('ar')*100:.0f}\\%", False),
            ("Multivariate ridge (CST $n{=}1$)", "mv", f"{sk('mv')*100:.0f}\\%", True)]
    lines = [r"\begin{tabular}{lccc}", r"  \toprule",
             r"  Forecaster & MAE @ 1 step (\si{\micro\second}) & MAE @ 8 steps (\si{\micro\second}) & Skill @ 8 steps \\",
             r"  \midrule"]
    for label, key, skill, bold in rows:
        first = f"{means[key][0]:.2f} \\pm {sds[key][0]:.2f}"
        eighth = f"{means[key][7]:.2f} \\pm {sds[key][7]:.2f}"
        if bold:
            first = rf"\mathbf{{{first}}}"; eighth = rf"\mathbf{{{eighth}}}"
            skill = rf"$\mathbf{{{skill}}}$"
        lines.append(f"  {label:<32} & ${first}$ & ${eighth}$ & {skill} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "forecasting_benchmark.tex").write_text("\n".join(lines) + "\n")
    (DATA_DIR / "forecasting_benchmark.json").write_text(json.dumps({
        "seeds": FORECAST_SEEDS, "horizons": horizons.tolist(),
        "mae_mean": {k: v.tolist() for k, v in means.items()},
        "mae_sd": {k: v.tolist() for k, v in sds.items()},
        "skill_ar_mean": skill_ar_m.tolist(), "skill_ar_sd": skill_ar_s.tolist(),
        "skill_multivariate_mean": skill_mv_m.tolist(),
        "skill_multivariate_sd": skill_mv_s.tolist(),
        "cstk_skill8_vs_n": {int(n): float(m) for n, m in zip(n_vals, sk_n_m)},
        "cstk_skill8_vs_n_sd": {int(n): float(s) for n, s in zip(n_vals, sk_n_s)},
    }, indent=2) + "\n")
    print(f"[fig2] MV skill@8={sk('mv')*100:.0f}%  CST skill@8 vs n="
          f"{[f'{m*100:.0f}' for m in sk_n_m]} -> {out.name}")
    return {"skill8_mv": float(sk("mv"))}


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 + Table 2 -- noncommutative detection of correlated drift
# ═════════════════════════════════════════════════════════════════════════════
def figure3_and_table2():
    X, y = generate_correlated_drift_windows(n_each=DRIFT_N_EACH, seed=DRIFT_SEED)

    sweep = truncation_sweep(X, y, N_GRID, kind="cst", n_splits=N_SPLITS, seed=DRIFT_SEED)
    auc_n = np.array(sweep["auc_mean"]); sd_n = np.array(sweep["auc_sd"])
    comm = auc_n[0]

    grid = rank_truncation_grid(X, y, N_GRID_HEAT, K_GRID_HEAT, kind="cst",
                                n_splits=N_SPLITS, seed=DRIFT_SEED)
    bc = baseline_comparison(X, y, n_cst=N_CST, n_periodic=N_PERIODIC, k=8,
                             n_splits=N_SPLITS, seed=DRIFT_SEED)
    # Raw-level reference (n-independent), drawn as a flat line in panel (a).
    raw_auc = float(bc["raw level (flatten)"][0])

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.4, 2.7))

    # (a) AUC vs truncation order n.
    ax_a.plot(sweep["n_grid"], auc_n, color=C_CST, linewidth=1.6, marker="o",
              markersize=3.2, label="causal CST kernel")
    ax_a.fill_between(sweep["n_grid"], auc_n - sd_n, auc_n + sd_n, color=C_CST,
                      alpha=0.18, linewidth=0)
    peak = int(np.argmax(auc_n))
    ax_a.scatter([sweep["n_grid"][peak]], [auc_n[peak]], color=C_PEAK, zorder=6, s=22)
    ax_a.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.0, label="chance")
    ax_a.axhline(raw_auc, color=C_RAW, linestyle=":", linewidth=1.0, label="raw-level monitor")
    ax_a.set_xlabel("truncation order $n$")
    ax_a.set_ylabel("detection ROC-AUC")
    ax_a.set_ylim(0.4, 1.0)
    ax_a.legend(loc="upper right", fontsize=6.0, handlelength=1.4)
    panel_label(ax_a, "a")

    # (b) (n, k) heatmap.
    im = ax_b.imshow(grid["auc"], aspect="auto", origin="lower", cmap="viridis",
                     vmin=0.5, vmax=1.0)
    ax_b.set_xticks(range(len(grid["k_grid"]))); ax_b.set_xticklabels(grid["k_grid"])
    ax_b.set_yticks(range(len(grid["n_grid"]))); ax_b.set_yticklabels(grid["n_grid"])
    ax_b.set_xlabel("bottleneck rank $k$")
    ax_b.set_ylabel("truncation order $n$")
    cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cb.set_label("ROC-AUC", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    panel_label(ax_b, "b")

    # (c) Baseline comparison bars.
    names = ["raw\nlevel", "commutative\n($n{=}1$)", "periodic\nSTK", "causal CST\n(ours)"]
    vals = [bc["raw level (flatten)"][0], bc["commutative kernel (n=1)"][0],
            bc[f"periodic spectral-truncation (n={N_PERIODIC})"][0],
            bc[f"causal spectral-truncation (ours, n={N_CST})"][0]]
    errs = [bc["raw level (flatten)"][1], bc["commutative kernel (n=1)"][1],
            bc[f"periodic spectral-truncation (n={N_PERIODIC})"][1],
            bc[f"causal spectral-truncation (ours, n={N_CST})"][1]]
    colors = [C_RAW, C_COMM, C_PER, C_CST]
    bars = ax_c.bar(names, vals, yerr=errs, color=colors, width=0.62,
                    error_kw=dict(lw=0.8, capsize=2))
    ax_c.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.0, label="chance")
    ax_c.bar_label(bars, fmt="%.2f", padding=2, fontsize=6.0)
    ax_c.set_ylabel("detection ROC-AUC")
    ax_c.set_ylim(0, 1.0)
    ax_c.tick_params(axis="x", labelsize=5.8)
    ax_c.legend(loc="upper left", fontsize=6.0)
    panel_label(ax_c, "c")

    fig.tight_layout()
    out = save_figure(fig, "fig3_noncommutative.pdf")

    # Table 2: detection benchmark.
    order = ["raw level (flatten)", "commutative kernel (n=1)",
             f"periodic spectral-truncation (n={N_PERIODIC})",
             f"causal spectral-truncation (ours, n={N_CST})"]
    pretty = {
        "raw level (flatten)": "Raw level (flatten)",
        "commutative kernel (n=1)": r"Commutative kernel ($n=1$)",
        f"periodic spectral-truncation (n={N_PERIODIC})":
            rf"Periodic spectral-truncation~\citep{{hashimoto2024spectral}} ($n={N_PERIODIC}$)",
        f"causal spectral-truncation (ours, n={N_CST})":
            rf"\textbf{{Causal spectral-truncation (ours, $n={N_CST}$)}}",
    }
    tlines = [r"\begin{tabular}{lc}", r"  \toprule",
              r"  Detector & Detection ROC-AUC \\", r"  \midrule"]
    best_name = order[-1]
    for nm in order:
        m, s = bc[nm]
        cell = f"{m:.3f} $\\pm$ {s:.3f}"
        if nm == best_name:
            cell = rf"$\mathbf{{{m:.3f} \pm {s:.3f}}}$"
        tlines.append(f"  {pretty[nm]} & {cell} \\\\")
    tlines += [r"  \bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "detection_benchmark.tex").write_text("\n".join(tlines) + "\n")

    # Extended Data table: truncation sweep.
    s2 = [r"\begin{tabular}{cc@{\hskip 2.4em}cc}", r"  \toprule",
          r"  Order $n$ & ROC-AUC & Order $n$ & ROC-AUC \\", r"  \midrule"]
    half = (len(N_GRID) + 1) // 2
    for i in range(half):
        left = rf"{N_GRID[i]} & ${auc_n[i]:.3f} \pm {sd_n[i]:.3f}$"
        if i + half < len(N_GRID):
            j = i + half
            cell_l = rf"\mathbf{{{auc_n[i]:.3f} \pm {sd_n[i]:.3f}}}" if i == peak else f"{auc_n[i]:.3f} \\pm {sd_n[i]:.3f}"
            cell_r = rf"\mathbf{{{auc_n[j]:.3f} \pm {sd_n[j]:.3f}}}" if j == peak else f"{auc_n[j]:.3f} \\pm {sd_n[j]:.3f}"
            s2.append(rf"  {N_GRID[i]} & ${cell_l}$ & {N_GRID[j]} & ${cell_r}$ \\")
        else:
            cell_l = rf"\mathbf{{{auc_n[i]:.3f} \pm {sd_n[i]:.3f}}}" if i == peak else f"{auc_n[i]:.3f} \\pm {sd_n[i]:.3f}"
            s2.append(rf"  {N_GRID[i]} & ${cell_l}$ & & \\")
    s2 += [r"  \bottomrule", r"\end{tabular}"]
    (TABLE_DIR / "truncation_sweep.tex").write_text("\n".join(s2) + "\n")

    (DATA_DIR / "detection_benchmark.json").write_text(json.dumps({
        "n_each": DRIFT_N_EACH, "seed": DRIFT_SEED, "n_splits": N_SPLITS,
        "truncation_sweep": {"n_grid": sweep["n_grid"],
                             "auc_mean": auc_n.tolist(), "auc_sd": sd_n.tolist(),
                             "best_k": sweep["best_k"]},
        "peak_n": int(sweep["n_grid"][peak]), "peak_auc": float(auc_n[peak]),
        "commutative_auc": float(comm), "raw_auc": float(raw_auc),
        "grid": {"n_grid": grid["n_grid"], "k_grid": grid["k_grid"],
                 "auc": grid["auc"].tolist()},
        "baseline_comparison": {k: list(v) for k, v in bc.items()},
    }, indent=2) + "\n")
    print(f"[fig3] n=1 AUC={comm:.3f}  peak n={sweep['n_grid'][peak]} AUC={auc_n[peak]:.3f}  "
          f"periodic={bc[f'periodic spectral-truncation (n={N_PERIODIC})'][0]:.3f}  "
          f"raw={raw_auc:.3f} -> {out.name}")
    return {"comm_auc": comm, "peak_n": int(sweep["n_grid"][peak]),
            "peak_auc": float(auc_n[peak]),
            "periodic_auc": bc[f"periodic spectral-truncation (n={N_PERIODIC})"][0],
            "cst_auc": bc[f"causal spectral-truncation (ours, n={N_CST})"][0],
            "raw_auc": raw_auc}


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 -- ROC curves and over-complete collapse
# ═════════════════════════════════════════════════════════════════════════════
def figure4():
    X, y = generate_correlated_drift_windows(n_each=DRIFT_N_EACH, seed=DRIFT_SEED)
    yi = y.astype(int)
    n_tr = int(0.7 * len(X))
    rng = np.random.default_rng(DRIFT_SEED)
    perm = rng.permutation(len(X))
    tr, te = perm[:n_tr], perm[n_tr:]
    nom = tr[yi[tr] == 0]

    def roc_for(kind, n, k):
        F_nom = feature_backend(X[nom], kind, n)
        F_te = feature_backend(X[te], kind, n)
        sc = reconstruction_scores(F_nom, F_te, k)
        fpr, tpr, _ = roc_curve(yi[te], sc)
        return fpr, tpr, roc_auc_score(yi[te], sc)

    fpr2, tpr2, auc2 = roc_for("cst", N_CST, 8)
    fpr1, tpr1, auc1 = roc_for("commutative", 1, 8)

    # Over-complete collapse: AUC vs truncation order at fixed rank (re-uses sweep).
    sw = truncation_sweep(X, y, N_GRID, kind="cst", k=8, n_splits=N_SPLITS, seed=DRIFT_SEED)
    auc_k8 = np.array(sw["auc_mean"]); sd_k8 = np.array(sw["auc_sd"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax_a.plot(fpr2, tpr2, color=C_CST, linewidth=1.8,
              label=f"causal CST $n={N_CST}$ (AUC = {auc2:.2f})")
    ax_a.plot(fpr1, tpr1, color=C_COMM, linewidth=1.4, linestyle="--",
              label=f"commutative $n=1$ (AUC = {auc1:.2f})")
    ax_a.plot([0, 1], [0, 1], color="#9ca3af", linestyle=":", linewidth=1.0, label="chance")
    ax_a.set_xlabel("false positive rate"); ax_a.set_ylabel("true positive rate")
    ax_a.legend(loc="lower right", fontsize=6.2)
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1.02)
    panel_label(ax_a, "a")

    ax_b.plot(sw["n_grid"], auc_k8, color=C_CST, linewidth=1.6, marker="o", markersize=3.0)
    ax_b.fill_between(sw["n_grid"], auc_k8 - sd_k8, auc_k8 + sd_k8, color=C_CST,
                      alpha=0.18, linewidth=0)
    ax_b.axhline(0.5, color=C_CHANCE, linestyle="--", linewidth=1.0, label="chance")
    ax_b.set_xlabel("truncation order $n$ (rank $k=8$)")
    ax_b.set_ylabel("detection ROC-AUC")
    ax_b.set_ylim(0.4, 1.0)
    ax_b.legend(loc="upper right")
    panel_label(ax_b, "b")

    fig.tight_layout()
    out = save_figure(fig, "fig4_detection.pdf")
    (DATA_DIR / "roc_overcomplete.json").write_text(json.dumps({
        "auc_cst_n2": float(auc2), "auc_commutative_n1": float(auc1),
        "n_grid": sw["n_grid"], "auc_k8_mean": auc_k8.tolist(),
        "auc_k8_sd": sd_k8.tolist(),
    }, indent=2) + "\n")
    print(f"[fig4] ROC CST n={N_CST} AUC={auc2:.3f} vs commutative AUC={auc1:.3f} -> {out.name}")


def main():
    print(f"Writing figures to: {OUT_DIR}")
    figure0_overview()
    figure1()
    figure2_and_table1()
    figure3_and_table2()
    figure4()
    print("Done. 5 figures + 3 table bodies regenerated from the package.")


if __name__ == "__main__":
    main()
