"""
generate_figures.py — regenerates the figures embedded in main.pdf.

Design constraints (top-venue style): every figure is a bar chart or a line
chart; CPU-only (numpy, pandas, scikit-learn, scipy, matplotlib); no GPU/PyTorch.
Figures 1, 2 and 4 are computed live from the reproducible telemetry generator in
src/data.py with multi-seed confidence bands; Figure 3 renders the sequence-model
benchmark metrics recorded by the executed notebooks.

Run from the repository root:
    python submission/figures/generate_figures.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data import generate_synthetic_dataset, build_dataset, FEATURE_COLS  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(REPO_ROOT, "data", "quantum_device_metrics.csv")
SEEDS = [42, 1, 2, 3, 4]
SEQ_LEN, HORIZON = 32, 8

INK = "#1b2733"; MUTED = "#5b6b7d"; GRID = "#dfe5ec"
C_PERS = "#9aa6b2"; C_CLIM = "#c2a35a"; C_AR = "#3f6fb0"; C_MV = "#1f8a70"
C_RNN = "#9aa6b2"; C_LSTM = "#3f6fb0"; C_GRU = "#1f8a70"; C_TFM = "#d98a2b"
C_ACCENT = "#1d3f6e"; C_DRIFT = "#b3424a"
MODEL_COLORS = {"VanillaRNN": C_RNN, "LSTM": C_LSTM, "GRU": C_GRU, "Transformer": C_TFM}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.titlecolor": INK, "text.color": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0, "legend.frameon": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def tag(ax, t, dx=-0.13, dy=1.05):
    ax.text(dx, dy, t, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", color=INK)


def save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {name}.png + {name}.pdf")


# ── Shared experiment: multi-baseline, multi-seed forecasting ────────────────
def _windows(series_feats, t1):
    """Sliding windows from a (T, F) feature array and its T1 column."""
    Xf, Xu, Y, last = [], [], [], []
    T = len(t1)
    for i in range(T - SEQ_LEN - HORIZON + 1):
        Xf.append(series_feats[i:i + SEQ_LEN].ravel())
        Xu.append(t1[i:i + SEQ_LEN])
        Y.append(t1[i + SEQ_LEN:i + SEQ_LEN + HORIZON])
        last.append(t1[i + SEQ_LEN - 1])
    return map(np.array, (Xf, Xu, Y, last))


def forecasting_experiment():
    """Return per-horizon MAE for four baselines, aggregated over SEEDS.

    Baselines: persistence, climatology, AR-ridge (univariate lags),
    multivariate ridge (all telemetry channels). Leak-free 80/20 temporal split
    per qubit; metrics aggregated over pooled test windows, repeated across seeds.
    """
    from sklearn.linear_model import Ridge
    names = ["persistence", "climatology", "AR-ridge", "multivariate ridge"]
    per_seed = {n: [] for n in names}
    for seed in SEEDS:
        df = generate_synthetic_dataset(n_qubits=5, n_steps=200, seed=seed)
        Xf_tr, Xu_tr, Y_tr, Xf_te, Xu_te, Y_te, last_te = [], [], [], [], [], [], []
        for q in sorted(df.qubit_id.unique()):
            sub = df[df.qubit_id == q].sort_values("timestamp_hr")
            feats = sub[FEATURE_COLS].values.astype(float)
            t1 = sub["T1_us"].values.astype(float)
            Xf, Xu, Y, last = _windows(feats, t1)
            k = int(0.8 * len(Y))
            Xf_tr.append(Xf[:k]); Xu_tr.append(Xu[:k]); Y_tr.append(Y[:k])
            Xf_te.append(Xf[k:]); Xu_te.append(Xu[k:]); Y_te.append(Y[k:]); last_te.append(last[k:])
        Xf_tr = np.concatenate(Xf_tr); Xu_tr = np.concatenate(Xu_tr); Y_tr = np.concatenate(Y_tr)
        Xf_te = np.concatenate(Xf_te); Xu_te = np.concatenate(Xu_te); Y_te = np.concatenate(Y_te)
        last_te = np.concatenate(last_te)

        clim = Y_tr.mean(axis=0)
        ar = Ridge(alpha=5.0).fit(Xu_tr, Y_tr).predict(Xu_te)
        mv = Ridge(alpha=5.0).fit(Xf_tr, Y_tr).predict(Xf_te)
        preds = {
            "persistence": np.repeat(last_te[:, None], HORIZON, axis=1),
            "climatology": np.repeat(clim[None, :], len(Y_te), axis=0),
            "AR-ridge": ar,
            "multivariate ridge": mv,
        }
        for n in names:
            per_seed[n].append(np.abs(preds[n] - Y_te).mean(axis=0))
    stats = {n: (np.mean(per_seed[n], 0), np.std(per_seed[n], 0)) for n in names}
    return names, stats


def figure1_dynamics():
    """Line charts: coherence trajectories and their autocorrelation structure."""
    df = generate_synthetic_dataset(n_qubits=5, n_steps=200, seed=42)
    qubits = sorted(df.qubit_id.unique())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.16, wspace=0.26)

    cmap = plt.cm.viridis(np.linspace(0.12, 0.85, len(qubits)))
    for q, col in zip(qubits, cmap):
        sub = df[df.qubit_id == q].sort_values("timestamp_hr")
        axa.plot(sub.timestamp_hr, sub.T1_us, lw=1.3, color=col, label=f"qubit {q}")
    axa.axhline(50, color=C_DRIFT, ls="--", lw=1.1)
    axa.text(99, 51, "operational threshold", color=C_DRIFT, fontsize=7.6, ha="right", va="bottom")
    axa.set_xlabel("time (hours)"); axa.set_ylabel(r"$T_1$ relaxation time (µs)")
    axa.set_title("Coherence drift trajectories")
    axa.legend(ncol=3, loc="lower left", columnspacing=1.1, handlelength=1.3, fontsize=7.2)
    tag(axa, "a", dx=-0.12)

    # Mean autocorrelation of T1 across qubits (detrended) — forecastable structure.
    nlags = 40
    acc = []
    for q in qubits:
        x = df[df.qubit_id == q].sort_values("timestamp_hr").T1_us.values.astype(float)
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[len(x) - 1:][:nlags + 1]
        acc.append(ac / ac[0])
    acc = np.array(acc)
    lags = np.arange(nlags + 1) * 0.5  # hours
    m, s = acc.mean(0), acc.std(0)
    axb.axhline(0, color=MUTED, lw=0.8, ls=":")
    axb.plot(lags, m, color=C_ACCENT, lw=2.2, label="mean ACF")
    axb.fill_between(lags, m - s, m + s, color=C_ACCENT, alpha=0.15, lw=0, label="±1 s.d. across qubits")
    axb.set_xlabel("lag (hours)"); axb.set_ylabel(r"autocorrelation of $T_1$")
    axb.set_title("Slow temporal structure enables forecasting")
    axb.legend(loc="upper right"); axb.set_xlim(0, lags[-1])
    tag(axb, "b", dx=-0.14)

    fig.suptitle("Quantum-hardware coherence drift: dynamics and predictable structure",
                 fontsize=12, fontweight="bold", y=0.99)
    save(fig, "fig1_dynamics")


def figure2_forecasting(names, stats):
    """Line charts: forecast error vs horizon with CI bands, and skill over persistence."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.16, wspace=0.26)
    hx = np.arange(1, HORIZON + 1)
    colors = {"persistence": C_PERS, "climatology": C_CLIM, "AR-ridge": C_AR,
              "multivariate ridge": C_MV}
    styles = {"persistence": "--", "climatology": "-.", "AR-ridge": "-", "multivariate ridge": "-"}
    for n in names:
        m, s = stats[n]
        axa.plot(hx, m, color=colors[n], lw=2.0, ls=styles[n], marker="o", ms=3.5, label=n)
        axa.fill_between(hx, m - s, m + s, color=colors[n], alpha=0.12, lw=0)
    axa.set_xlabel("forecast horizon (steps ahead)"); axa.set_ylabel("mean absolute error (µs)")
    axa.set_title("Forecast error vs horizon"); axa.set_xticks(hx)
    axa.legend(loc="upper left"); tag(axa, "a", dx=-0.13)

    # Skill score: 1 - MAE_model / MAE_persistence (%) for the learned forecasters.
    mp, _ = stats["persistence"]
    for n in ["AR-ridge", "multivariate ridge"]:
        m, s = stats[n]
        skill = 100 * (1 - m / mp)
        axb.plot(hx, skill, color=colors[n], lw=2.2, ls=styles[n], marker="o", ms=4, label=n)
    axb.axhline(0, color=MUTED, lw=1.0, ls=":")
    axb.text(1, 1.5, "persistence baseline", color=MUTED, fontsize=7.4, ha="left", va="bottom")
    axb.set_xlabel("forecast horizon (steps ahead)"); axb.set_ylabel("skill over persistence (%)")
    axb.set_title("Forecast skill grows with horizon"); axb.set_xticks(hx)
    axb.set_ylim(-5, 80); axb.legend(loc="lower right"); tag(axb, "b", dx=-0.14)

    fig.suptitle("Learned forecasters beat naive baselines, with the gap widening over the horizon",
                 fontsize=12, fontweight="bold", y=0.99)
    save(fig, "fig2_forecasting")


def figure3_benchmark():
    """Bar charts: objective-aware sequence-model benchmark (recorded metrics)."""
    thermal = {
        "VanillaRNN": dict(f1=0.0000, auc=0.4083, params=5645),
        "LSTM": dict(f1=0.0000, auc=0.4234, params=116845),
        "GRU": dict(f1=0.2574, auc=0.7182, params=87949),
    }
    cross_auc = {"GRU": 0.6603, "LSTM": 0.6278, "Transformer": 0.1955}
    tfm_periodic = 0.7987

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.17, wspace=0.42)
    axa, axb, axc = axes
    m3 = ["VanillaRNN", "LSTM", "GRU"]; c3 = [MODEL_COLORS[m] for m in m3]

    x = np.arange(len(m3)); w = 0.38
    axa.bar(x - w / 2, [thermal[m]["auc"] for m in m3], w, color=c3, edgecolor="white")
    axa.bar(x + w / 2, [thermal[m]["f1"] for m in m3], w, color=c3, alpha=0.45,
            edgecolor="white", hatch="///")
    axa.axhline(0.5, color=MUTED, ls=":", lw=1.0)
    axa.text(2.45, 0.5, "chance", color=MUTED, fontsize=7.2, va="bottom", ha="right")
    for xi, m in zip(x, m3):
        axa.text(xi - w / 2, thermal[m]["auc"] + 0.02, f"{thermal[m]['auc']:.2f}", ha="center", fontsize=7.2)
        axa.text(xi + w / 2, thermal[m]["f1"] + 0.02, f"{thermal[m]['f1']:.2f}", ha="center", fontsize=7.2)
    axa.set_xticks(x); axa.set_xticklabels(m3, rotation=12); axa.set_ylim(0, 0.85)
    axa.set_ylabel("score"); axa.set_title("Thermal incident detection")
    axa.legend(handles=[Line2D([0], [0], color=MUTED, lw=6, label="ROC-AUC"),
                        Line2D([0], [0], color=MUTED, lw=6, alpha=0.45, label="F1")], loc="upper left")
    tag(axa, "a", dx=-0.2)

    order = ["GRU", "LSTM", "Transformer"]
    axb.bar(order, [cross_auc[m] for m in order], color=[MODEL_COLORS[m] for m in order],
            edgecolor="white", width=0.6)
    for m in order:
        axb.text(m, cross_auc[m] + 0.02, f"{cross_auc[m]:.2f}", ha="center", fontsize=8)
    axb.axhline(0.5, color=MUTED, ls=":", lw=1.0)
    axb.bar(["Transformer*"], [tfm_periodic], color=C_TFM, alpha=0.5, edgecolor=C_TFM,
            hatch="///", width=0.6)
    axb.text("Transformer*", tfm_periodic + 0.02, f"{tfm_periodic:.2f}", ha="center", fontsize=8, color=C_TFM)
    axb.set_ylim(0, 0.9); axb.set_ylabel("ROC-AUC")
    axb.set_title("Generalist vs specialist")
    axb.tick_params(axis="x", labelrotation=18)
    axb.text(0.5, -0.34, "* Transformer on the periodic regime only", transform=axb.transAxes,
             ha="center", fontsize=7.0, color=MUTED)
    tag(axb, "b", dx=-0.2)

    params = [thermal[m]["params"] for m in m3]
    bars = axc.bar(m3, params, color=c3, edgecolor="white", width=0.6)
    axc.set_yscale("log"); axc.set_ylabel("trainable parameters")
    axc.set_title("Parameter efficiency")
    axc.tick_params(axis="x", labelrotation=12)
    for b, m in zip(bars, m3):
        axc.text(b.get_x() + b.get_width() / 2, params[m3.index(m)] * 1.15,
                 f"{thermal[m]['params']:,}\nAUC {thermal[m]['auc']:.2f}", ha="center",
                 fontsize=6.8, color=INK)
    axc.set_ylim(2e3, 4e5)
    tag(axc, "c", dx=-0.22)

    fig.suptitle("No architecture wins everywhere: the GRU is the parameter-efficient generalist",
                 fontsize=12, fontweight="bold", y=0.99)
    save(fig, "fig3_benchmark")


def figure4_anomaly():
    """Line charts: detection ROC and detectability vs reconstruction bottleneck rank."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    ds = build_dataset(csv_path=CSV, seq_len=SEQ_LEN, horizon=HORIZON)
    X = np.concatenate([ds[s][0] for s in ("train", "val", "test")], 0)
    lbl = np.concatenate([ds[s][2] for s in ("train", "val", "test")], 0)
    n, sl, nf = X.shape; Xf = X.reshape(n, sl * nf)

    def split_score(k, seed):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n); cut = int(0.7 * n)
        tr, te = perm[:cut], perm[cut:]
        nominal = Xf[tr][lbl[tr] == 0]
        p = PCA(n_components=k, random_state=0).fit(nominal)
        sc = ((Xf[te] - p.inverse_transform(p.transform(Xf[te]))) ** 2).mean(1)
        return lbl[te], sc

    K = 3
    lte, score = split_score(K, 7)
    fpr, tpr, _ = roc_curve(lte, score); roc_auc = auc(fpr, tpr)
    ranks = [1, 2, 3, 5, 8, 16, 32]
    by_rank = []
    for k in ranks:
        vals = [roc_auc_score(*split_score(k, s)) for s in range(8)]
        by_rank.append(vals)
    by_rank = np.array(by_rank)
    rank_m, rank_s = by_rank.mean(1), by_rank.std(1)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.16, wspace=0.27)
    axa.plot(fpr, tpr, color=C_ACCENT, lw=2.4, label=f"detector (AUC = {roc_auc:.2f})")
    axa.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1.0, label="chance")
    axa.set_xlabel("false-positive rate"); axa.set_ylabel("true-positive rate")
    axa.set_title("Drift-detection ROC (rank 3)"); axa.set_xlim(0, 1); axa.set_ylim(0, 1.02)
    axa.legend(loc="lower right"); tag(axa, "a", dx=-0.13)

    axb.plot(ranks, rank_m, color=C_TFM, lw=2.2, marker="o", ms=5)
    axb.fill_between(ranks, rank_m - rank_s, rank_m + rank_s, color=C_TFM, alpha=0.15, lw=0,
                     label="±1 s.d. across splits")
    axb.axhline(0.5, color=MUTED, ls=":", lw=1.0)
    axb.text(ranks[-1], 0.5, "chance", color=MUTED, fontsize=7.2, va="bottom", ha="right")
    axb.set_xscale("log", base=2); axb.set_xticks(ranks); axb.set_xticklabels([str(r) for r in ranks])
    axb.set_xlabel("reconstruction bottleneck rank"); axb.set_ylabel("detection ROC-AUC")
    axb.set_title("Drift lives off a low-rank manifold"); axb.set_ylim(0.35, 0.95)
    axb.legend(loc="upper right"); tag(axb, "b", dx=-0.14)

    fig.suptitle("Unsupervised drift detection by reconstruction error, controlled by bottleneck rank",
                 fontsize=12, fontweight="bold", y=0.99)
    save(fig, "fig4_anomaly")
    return roc_auc, dict(zip(ranks, zip(rank_m, rank_s)))


def print_numbers(names, stats):
    print("\n--- numbers for manuscript (CPU, seeded) ---")
    for n in names:
        m, s = stats[n]
        print(f"  {n:20s} MAE h1 {m[0]:6.2f}±{s[0]:.2f} | h8 {m[-1]:6.2f}±{s[-1]:.2f}")
    mp = stats["persistence"][0]
    mv = stats["multivariate ridge"][0]
    print(f"  multivariate-ridge skill@h8 = {100*(1-mv[-1]/mp[-1]):.1f}%")


def main():
    set_style()
    names, stats = forecasting_experiment()
    figure1_dynamics()
    figure2_forecasting(names, stats)
    figure3_benchmark()
    a, rk = figure4_anomaly()
    print_numbers(names, stats)
    print(f"  anomaly rank-3 AUC (Fig.4 split) = {a:.3f}")
    print(f"  anomaly AUC by rank: " + ", ".join(f"r{k}:{v[0]:.2f}±{v[1]:.2f}" for k, v in rk.items()))
    print("done.")


if __name__ == "__main__":
    main()
