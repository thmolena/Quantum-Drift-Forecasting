# Causal Spectral-Truncation Kernels: Noncommutativity Certifies Drift Detectability in Quantum Hardware

**Molena Huynh** · North Carolina State University · molena.huynh@jmp.com

**[Read the manuscript (PDF)](submission/main.pdf)** · [Full tutorial](index.html) · [Reproducibility package (`qdriftforecast`)](submission/code) · Python 3.9–3.13 · MIT License

Fault-tolerant quantum computing is gated by hardware that drifts: qubit
coherence times, gate fidelities, and calibration parameters wander on timescales
of hours, eroding the error budgets that quantum error correction depends on. The
most damaging drift mode is *correlated decoherence* — control crosstalk, shared
two-level-system baths, and correlated flux noise couple the errors of
neighbouring qubits and calibration channels, often with a propagation delay, so
that a disturbance reappears lagged in a different channel. Such an event can
leave every per-channel marginal and every instantaneous cross-channel covariance
unchanged while living entirely in the *ordered, lagged* cross-channel structure.
A monitor blind to that structure is blind to the drift.

> **AI for quantum, at HPC scale.** This recasts quantum-hardware reliability monitoring as *operator-algebraic machine learning* on calibration telemetry: a C\*-algebra-valued kernel whose truncation order tunes the noncommutativity it can represent, learning the ordered, lagged cross-channel structure that defines correlated drift (**ROC-AUC 0.84 vs 0.46** at chance for commutative monitors). It is positive-definite for every order and reduces to standard low-rank detection and ridge forecasting at order one, so it strictly generalizes established monitors; its kernel evaluations are batched linear algebra that maps onto GPU backends for fleet-scale telemetry. Byte-deterministic and `pip`-installable.

This repository recasts reliability monitoring as operator-algebraic learning on
multivariate calibration telemetry and introduces the **causal spectral-truncation
(CST) kernel**, a C\*-algebra-valued positive-definite kernel on telemetry windows
whose truncation order `n` controls the noncommutativity of the cross-channel
products it represents. It is the *causal* counterpart of the periodic
spectral-truncation kernels of Hashimoto et al. (NeurIPS 2024,
[arXiv:2405.17823](https://arxiv.org/abs/2405.17823)): in place of a periodic
Fourier-multiplier (Toeplitz) truncation we truncate the causal lower-shift
algebra, so the kernel respects the arrow of time and its noncommutativity encodes
ordered, lagged cross-channel correlations.

The principal result is a separation that only a noncommutative monitor can
resolve: a physically motivated correlated-decoherence (crosstalk) drift mode,
constructed with matched per-channel marginals and (in the stationary limit)
matched instantaneous cross-covariance, concentrates its signal in the ordered,
lagged cross-channel structure. The commutative monitors we evaluate are all at
chance, and the drift is detected only by the noncommutative kernel, with a sharp
optimal truncation order that the accompanying theory certifies.

## The causal spectral-truncation kernel

A telemetry window is `X ∈ ℝ^{L×C}` (`L = 32` time steps, `C = 7` channels). Let
`S` be the causal lower shift, `(Sv)_t = v_{t-1}`. The order-`n` causal
spectral-truncation kernel represents the window inside the channel C\*-algebra
`M_C` through the truncated shift powers `{S^τ : 0 ≤ τ < n}`:

```text
K_n(X, Y) = Σ_{τ=0}^{n-1} w_τ · Xᵀ S^τ Y  ∈  M_C,    w_τ = 1 / (L − τ).
```

Each entry is a weighted sum of ordered, lagged cross-channel correlations. At
`n = 1` the kernel reduces to the commutative instantaneous covariance `XᵀY`; for
`n ≥ 2` it adds asymmetric lagged cross-channel terms and becomes noncommutative
(`K_n(X,Y)_{cc'} ≠ K_n(X,Y)_{c'c}`, because `Sᵀ ≠ S`). The kernel is
positive-definite for every order. The commutative `n = 1` member coincides with
the low-rank reconstruction detector and the multivariate-ridge forecaster, so the
family is a strict generalisation of those established methods rather than a
competitor to them.

## Main result: noncommutativity is necessary for correlated-drift detection

We construct a correlated-decoherence benchmark in which a single shared white
latent is injected into four channels at channel-specific integer delays with
variance-preserving mixing. By construction every per-channel marginal, every
temporal autocorrelation, and (in the stationary limit) the lag-0 cross-covariance
are held identical to nominal operation; the perturbation is the ordered lag-`τ`
cross-channel covariance. The drift energy is therefore concentrated in lagged
structure that monitors built on marginals or instantaneous (commutative)
covariance cannot resolve: the commutative monitors evaluated here are all at
chance, and the drift is exposed only by the causal noncommutative (`n ≥ 2`)
kernel.

**Kernel comparison** (correlated-drift benchmark; ROC-AUC, mean ± s.d. over eight
randomised 70/30 splits, bottleneck rank `k = 8`):

| Representation | ROC-AUC |
|---|---|
| Raw level | 0.49 ± 0.03 |
| Commutative kernel (`n = 1`) | 0.46 ± 0.02 |
| Periodic spectral-truncation kernel (Hashimoto et al., `n = 2`) | 0.47 ± 0.03 |
| **Causal spectral-truncation kernel (ours, `n = 2`)** | **0.84 ± 0.02** |

The raw-level monitor, the commutative kernel, and the periodic
spectral-truncation kernel applied unchanged are all at chance. Only the causal
spectral-truncation kernel separates the classes. The periodic kernel fails not
because spectral truncation is wrong but because its circular truncation smears
the causal delay signature across the window boundary; the causal lower-shift
truncation aligns exactly with the propagation delays that define the drift.
Causality, not truncation alone, is what makes the kernel see the event. On a
held-out split the causal kernel at `n = 2` attains ROC-AUC 0.86 while the
commutative kernel traces the chance diagonal (ROC-AUC 0.46).

**Truncation sweep** (best ROC-AUC over bottleneck ranks, mean over eight splits):

| Truncation order `n` | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
|---|---|---|---|---|---|---|---|
| ROC-AUC | 0.51 | **0.90** | 0.83 | 0.78 | 0.69 | 0.65 | 0.57 |

Detection is unimodal in `n` with a sharp interior optimum at `n = 2`. The
commutative endpoint (`n = 1`) sits at chance, as the blindness theorem requires;
detection then peaks and decays back toward chance as the representation becomes
over-complete (each added lag beyond the informative one contributes only noise
coordinates that dilute the fixed-rank nominal subspace). Both endpoints —
commutative (`n = 1`) and over-complete (`n → L`) — are at chance. This is the
detection analogue of the representation–complexity trade-off that governs
spectral-truncation kernels in regression: there the truncation order trades
representation power against Rademacher complexity; here it trades captured drift
energy against the dimension of the nominal subspace.

## Secondary result: coherence forecasting

The same kernel forecasts the `T₁` relaxation time with horizon-growing skill.
Under a leakage-free 80/20 chronological split, aggregated over five
telemetry-generator seeds (MAE in microseconds):

| Forecaster | MAE @ 1 step | MAE @ 8 steps | Skill @ 8 steps |
|---|---|---|---|
| Persistence | 2.44 ± 0.16 | 6.79 ± 0.18 | — |
| Climatology | 13.01 | 12.40 | −83% |
| AR-ridge (`T₁` history) | 1.85 | 2.19 | 68% |
| Multivariate ridge (commutative `n = 1` member) | 1.80 ± 0.09 | 1.88 ± 0.06 | 72% |

Persistence degrades steeply with horizon while the learned forecasters stay
nearly flat, so skill over persistence grows from 26% one step ahead to 72% eight
steps ahead. Increasing the kernel's truncation order does not change this: the CST
forecaster's eight-step skill is flat in `n` (72% at `n = 1` to 71% at `n = 4`). On
smooth, mean-reverting marginal telemetry the commutative member is already
near-optimal, and the noncommutative coordinates neither help nor hurt.
Noncommutativity is unnecessary for smooth marginal forecasting and decisive only
for correlated-drift detection — the operational rule is to match the kernel's
noncommutativity to the geometry of the drift.

## Installation

```bash
pip install qdriftforecast
```

Installation from a local checkout of the package directory is equivalent:

```bash
pip install submission/code
```

The figure-and-table pipeline runs CPU-only on NumPy, pandas, SciPy,
scikit-learn, and Matplotlib. PyTorch is not required for the figure/table
pipeline; it is available through the optional `ml` extra
(`pip install "qdriftforecast[ml]"`).

## Reproduction

After installation, every figure and table regenerates deterministically through
the console entry point:

```bash
qdrift-reproduce
```

This is equivalent to `python -m qdriftforecast.reproduce`, or to running
`python make_paper_figures.py` from `submission/code/`. The command pins
`KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`, then writes five vector PDF
figures to `submission/figures/`, three LaTeX table bodies to `submission/tables/`,
and diagnostic JSON summaries to `submission/code/generated_data/`. Execution
completes in seconds on a single CPU. Determinism is byte-level: seeds and
`SOURCE_DATE_EPOCH` are pinned, and the vector PDF output is fixed across runs.
Determinism and seeds are documented in
[`submission/code/README.md`](submission/code/README.md).

Regenerated figures (`submission/figures/`):

| Output | Content |
|---|---|
| `fig0_overview.pdf` | Overview of the causal spectral-truncation monitoring pipeline |
| `fig1_dynamics.pdf` | `T₁` trajectories, detrended autocorrelation, and the noncommutative drift signature by lag |
| `fig2_forecasting.pdf` | Forecasting MAE versus horizon, skill over persistence, and CST skill versus `n` |
| `fig3_noncommutative.pdf` | Detection ROC-AUC versus truncation order, the `(n, k)` map, and the kernel comparison |
| `fig4_detection.pdf` | Operating characteristic on a held-out split and the over-complete collapse |

Regenerated table bodies (`submission/tables/`):

| Output | Content |
|---|---|
| `forecasting_benchmark.tex` | Forecasting baselines (MAE at 1 and 8 steps, skill) |
| `detection_benchmark.tex` | Correlated-drift detection kernel comparison at fixed rank |
| `truncation_sweep.tex` | Detection ROC-AUC versus truncation order |

## Repository Layout

```text
Quantum-Drift-Forecasting/
├── README.md
├── LICENSE
├── submission/                   # manuscript and reproduction package
│   ├── main.tex                  # self-contained LaTeX source (bibliography inlined)
│   ├── main.pdf                  # compiled journal-format article
│   ├── figures/                  # regenerated vector PDF figures
│   ├── tables/                   # regenerated LaTeX table bodies
│   └── code/                     # the qdriftforecast package
│       ├── pyproject.toml
│       ├── README.md
│       ├── make_paper_figures.py
│       ├── generated_data/       # diagnostic JSON summaries
│       └── qdriftforecast/
│           ├── kernels.py        # causal spectral-truncation kernel + periodic baseline
│           ├── detection.py      # reconstruction detector, truncation/rank sweeps, baseline comparison
│           ├── forecasting.py    # forecasting baselines, CST forecast skill
│           ├── data.py           # telemetry generator + correlated-drift benchmark
│           └── reproduce.py      # qdrift-reproduce entry point
└── data/                         # synthetic telemetry
```

## Manuscript

A self-contained research manuscript — *Causal spectral-truncation kernels:
noncommutativity certifies drift detectability in quantum hardware* — lives in
[`submission/`](submission/):

- [`main.pdf`](submission/main.pdf): the compiled, journal-format article with
  embedded figures, three results tables, the supporting theory (positive
  definiteness for every order, a blindness theorem for commutative monitors, and
  an optimal-truncation detection certificate), Methods, and an inlined
  bibliography.
- [`main.tex`](submission/main.tex): the self-contained LaTeX source.

All figures, tables, and reported uncertainties are computed live from the seeded
telemetry generator in
[`submission/code/qdriftforecast/data.py`](submission/code/qdriftforecast/data.py)
and regenerate from the released code on commodity CPU hardware.

## Citation

```bibtex
@article{huynh2026cstk,
  title   = {Causal spectral-truncation kernels: noncommutativity certifies
             drift detectability in quantum hardware},
  author  = {Huynh, Molena},
  year    = {2026},
  note    = {North Carolina State University. Correspondence:
             molena.huynh@jmp.com},
}
```

We extend, and on this causal detection task outperform, the periodic
spectral-truncation kernels of Hashimoto, Ghriss, Ikeda, and Kadri, *Spectral
truncation kernels: noncommutativity in C\*-algebraic kernel machines*, NeurIPS
2024 ([arXiv:2405.17823](https://arxiv.org/abs/2405.17823)).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
