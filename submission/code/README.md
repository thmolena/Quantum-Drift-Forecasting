# specops-cst

Reproduction package for *Causal Spectral-Truncation Kernels: Noncommutativity
Certifies Drift Detectability in Quantum Hardware* (Molena Huynh, North Carolina
State University; BibTeX key `huynh2026cst`). The package deterministically
regenerates every manuscript figure and result table from a seeded telemetry
generator and a controlled correlated-drift benchmark. Distribution name:
`specops-cst` (version 2.0.0); the importable package is `qdriftforecast`. Part
of the **spectral-truncation operators (SpecOps)** program, which treats spectral
truncation of operators as a unifying primitive for resource-accounted,
CPU-reproducible quantum-AI methods.

## Summary

A quantum computer is a stochastic device: every quantity known about it is
estimated from samples, so its calibration record is itself a noisy multivariate
time series, and deciding whether the machine has drifted out of specification is
a statistical inference problem rather than a gauge reading. This work introduces
the **causal spectral-truncation (CST) kernel**, a C\*-algebra-valued
positive-definite kernel on windows of calibration telemetry whose truncation
order `n` controls the noncommutativity of the cross-channel products it
represents, and proves that this noncommutativity certifies detectability of
correlated hardware drift. On a correlated-decoherence benchmark whose per-channel
marginals and stationary instantaneous covariance are matched to nominal
operation, every commutative monitor detects at chance (ROC-AUC ≈ 0.50), whereas
the noncommutative CST kernel reaches ROC-AUC 0.90 at a provably optimal
truncation order `n = 2`. The same kernel, at its commutative order, forecasts
relaxation-time coherence with 72% lower error than persistence eight steps ahead.
The pipeline is training-free at inference, installs with `pip`, and regenerates
every figure and table deterministically on a single CPU in seconds.

## Background and problem setting

The following is written for a reader in an adjacent field with no prior exposure
to this subarea.

**Quantum processors drift.** The physical parameters that determine whether a
superconducting quantum processor can run a useful circuit — the relaxation time
`T_1`, the dephasing time `T_2`, single- and two-qubit gate fidelities,
readout-assignment error, error-per-Clifford, and tunable-coupling phase — are not
constants. They evolve continuously with temperature, magnetic-flux noise,
two-level-system defect dynamics in the device oxides, and recalibration history.
In the near-term (NISQ) regime this drift caps the depth of usable circuits; in
the fault-tolerant regime it threatens the load-bearing assumption that physical
error rates remain below the error-correction threshold long enough for logical
errors to be suppressed. As devices scale to thousands of qubits, deciding *when*
a device has drifted and *which* components warrant the expensive intervention of
recalibration becomes a data problem beyond manual inspection.

**The damaging drift is correlated and lagged.** Control crosstalk, shared
two-level-system baths, and correlated flux noise couple the errors of neighbouring
qubits and of distinct calibration channels, frequently with a *propagation delay*:
a disturbance perturbing one qubit's coherence reappears, lagged, in a neighbour's
readout or coupling phase. This *correlated decoherence* is precisely the failure
mode that quantum error correction is least able to absorb — and precisely the mode
that per-channel, level-based monitors are worst equipped to see. Such an event can
leave every per-channel marginal distribution and every instantaneous (lag-zero)
cross-channel correlation unchanged while living entirely in the *ordered, lagged*
cross-channel structure. A monitor blind to that structure is blind to the drift.

**Why an operator-algebraic view.** A window of multivariate telemetry is naturally
an element of a matrix algebra over the channels; the cross-channel, cross-time
interactions a monitor can resolve are exactly the products in that algebra it
computes. *Commutative* kernels — whose feature products are insensitive to the
order or relative lag of their factors — cannot represent ordered, lagged
interactions at all. The recent spectral-truncation kernels of Hashimoto et al.
(NeurIPS 2024, arXiv:2405.17823) make this precise for periodic functional data by
truncating the algebra of multiplication operators on the torus to finite Toeplitz
matrices, obtaining a family of C\*-algebra-valued kernels whose noncommutativity is
governed by a single truncation parameter. That construction was built for periodic
regression; it does not respect the arrow of time and had not been brought to
time-ordered hardware telemetry or to unsupervised detection. Bridging that gap is
the object of this work.

## Contributions

1. **The causal spectral-truncation (CST) kernel.** A C\*-algebra-valued
   positive-definite kernel on telemetry windows,
   `K_n(X, Y) = Σ_{τ<n} w_τ · Xᵀ S^τ Y ∈ M_C`, where `S` is the causal lower shift
   `(Sv)_t = v_{t-1}` and the weights are `w_τ = 1/(L-τ)`. The truncation order `n`
   interpolates continuously from a commutative instantaneous monitor (`n = 1`,
   the plain covariance `XᵀX`) to a noncommutative monitor that represents ordered,
   lagged cross-channel correlations (`n ≥ 2`). It is the causal, time-respecting
   counterpart of the periodic spectral-truncation kernel of Hashimoto et al., and
   it subsumes the low-rank reconstruction detector and the multivariate ridge
   forecaster as commutative special cases, so the family strictly generalizes
   established monitors rather than competing with them.

2. **Noncommutativity is necessary, and certified to be so.** On a physically
   motivated correlated-decoherence benchmark engineered so the drift lives in
   ordered, lagged cross-channel structure, the raw-level monitor, the commutative
   (`n = 1`) kernel, and even the periodic spectral-truncation kernel (ported
   unchanged) all detect at chance (ROC-AUC ≈ 0.50), whereas the causal kernel
   detects at ROC-AUC 0.90 at a certifiable optimal order `n = 2` — the single
   largest effect in the study, a 0.39 swing from adding one lagged coordinate.

3. **A certified optimal truncation order.** Detection ROC-AUC is unimodal in `n`
   with an interior optimum where the marginal captured-drift energy falls below
   the marginal noise cost; because the correlated-decoherence energy concentrates
   at the first informative lag, the optimum is `n* = 2`, and both the commutative
   endpoint (`n = 1`) and the over-complete limit (`n → L`) sit at chance. This
   optimal-truncation certificate is the unsupervised-detection analogue of the
   representation-versus-complexity trade-off that governs spectral-truncation
   kernels in regression. The optimal order is a *built-in* diagnostic of the drift
   geometry, not a tuned hyperparameter.

4. **Forecasting, where noncommutativity is provably unnecessary.** At its
   commutative order the same kernel forecasts `T_1` coherence with 72% lower error
   than persistence eight steps ahead (skill widening from 26% one step ahead), a
   horizon-growing margin derived in closed form from mean reversion. The CST
   forecaster's eight-step skill is flat in `n` (72% at `n = 1`, 71% at `n = 4`):
   the noncommutative coordinates neither help nor hurt smooth marginal forecasting.

5. **An end-to-end reproducible toolbox.** The kernel is training-free at inference,
   deterministic, and hardware-agnostic; it installs with `pip` and regenerates
   every figure, table, and confidence interval from fixed integer seeds on a single
   CPU in seconds, byte-for-byte.

## Method

A telemetry window is a matrix `X ∈ ℝ^{L×C}` with `L = 32` time steps and `C = 7`
channels, centred per channel. The order-`n` CST kernel represents the window inside
the channel C\*-algebra `M_C = M_C(ℝ)` of `C×C` real matrices through the truncated
shift powers `{S^τ : 0 ≤ τ < n}`, so each matrix entry
`K_n(X, Y)_{cc'} = Σ_{τ<n} w_τ ⟨X_{:,c}, S^τ Y_{:,c'}⟩` is a weighted sum of ordered,
lagged cross-channel correlations. Because `S` is nilpotent and `Sᵀ ≠ S`, the kernel
is causal and, for `n ≥ 2`, noncommutative: `K_1` is the symmetric instantaneous
covariance, while `n ≥ 2` acquires antisymmetric lagged coordinates that no
commutative member can represent. The scalar realization
`k_n(X, Y) = ⟨Φ_n(X), Φ_n(Y)⟩`, with feature tensor
`Φ_n(X)_{cc'τ} = w_τ ⟨X_{:,c}, S^τ X_{:,c'}⟩`, is positive definite for every order.

Two heads share this representation. The **unsupervised detector** fits a rank-`k`
nominal subspace `P_k` in the CST feature space (from the top-`k` singular vectors of
standardized nominal features) and scores a window by its residual energy off that
subspace, `s_k(X) = ‖(I − P_k)(Φ_n(X) − Φ̄)‖²`; at `n = 1` this recovers the classical
low-rank reconstruction detector. The **forecaster** is a multivariate Tikhonov
(ridge) regression on the flattened window — the commutative `n = 1` member of the
family. Forecasting is scored by horizon-resolved mean absolute error (MAE) and
detection by the threshold-free area under the ROC curve (ROC-AUC). The correlated-
decoherence benchmark injects a shared white latent into four channels at
channel-specific integer delays `(0, 1, 2, 3)` under variance-preserving mixing, so
that per-channel marginals and the stationary instantaneous covariance match nominal
operation and the perturbation resides in the lag-`τ ≥ 1` cross-channel covariance.
Every quantitative claim rests on an elementary, self-contained theorem: positive
definiteness at every order; a blindness theorem for commutative monitors; the
optimal-truncation detection certificate; ROC-AUC scale invariance (Mann–Whitney);
and horizon-growing forecast skill from mean reversion.

## Main results

All values are transcribed from the manuscript (mean ± s.d.).

**Forecasting** (five telemetry-generator seeds; skill relative to persistence at
the eight-step horizon):

| Forecaster | MAE @ 1 step (µs) | MAE @ 8 steps (µs) | Skill @ 8 steps |
|---|---|---|---|
| Persistence | 2.44 ± 0.16 | 6.79 ± 0.18 | — |
| Climatology | 13.01 ± 0.05 | 12.40 ± 0.05 | −83% |
| AR-ridge (`T_1` history) | 1.85 ± 0.07 | 2.19 ± 0.06 | 68% |
| **Multivariate ridge (CST `n=1`)** | **1.80 ± 0.09** | **1.88 ± 0.06** | **72%** |

Persistence degrades steeply with horizon while the learned forecasters stay nearly
flat, so skill over persistence *grows* with lead time, from 26% one step ahead to
72% eight steps ahead. Increasing the truncation order leaves this unchanged (72% at
`n = 1`, 71% at `n = 4`).

**Correlated-drift detection** (eight randomized 70/30 splits, bottleneck rank
`k = 8`):

| Detector | Detection ROC-AUC |
|---|---|
| Raw level (flatten) | 0.49 ± 0.03 |
| Commutative kernel (`n=1`) | 0.46 ± 0.02 |
| Periodic spectral-truncation kernel (Hashimoto et al., `n=2`) | 0.47 ± 0.03 |
| **Causal spectral-truncation kernel (ours, `n=2`)** | **0.84 ± 0.02** |

The benchmark matches all per-channel marginals and the stationary instantaneous
covariance, so every commutative monitor is at chance; only the causal kernel
separates the classes (0.84 at matched rank, 0.90 at its optimum). The
nominal-versus-drift gap in the causal cross-channel features is 0.008 at lag `τ = 0`
and an order of magnitude larger, 0.089, at lag `τ = 1`, locating the signal exactly
where a commutative monitor cannot look.

**Truncation sweep.** Detection ROC-AUC is unimodal in `n`: 0.51 (chance) at `n = 1`,
peaking at 0.90 at `n = 2`, then decaying monotonically (0.83 at `n = 3`, 0.65 at
`n = 8`) back toward chance as the representation becomes over-complete — the
certified profile, with the two chance-level endpoints at `n = 1` and `n → L`.

## Significance

The work recasts quantum-hardware reliability monitoring as operator-algebraic
learning and distils the result into a single operational rule: *match the kernel's
noncommutativity to the geometry of the drift*. It provides both a theoretical
characterization of when noncommutative monitoring is necessary (correlated,
lagged drift) and superfluous (smooth marginal drift), and a practical, auditable
toolbox that runs on commodity CPUs. Because the family recovers established
forecasting and low-rank detection at its commutative order, it extends the reach of
existing monitors while inheriting their interpretability, and its per-window cost
grows only linearly in the noncommutativity it exploits — a design point suited to
per-qubit or per-coupler monitors instantiated thousands of times across a device.

## Installation

```bash
pip install specops-cst
```

Installation from a local checkout of this directory is equivalent:

```bash
cd submission/code
pip install .
```

The figure-and-table pipeline runs CPU-only on NumPy, pandas, SciPy,
scikit-learn, and Matplotlib. PyTorch is not required for the figure/table
pipeline; it is available through the optional `ml` extra
(`pip install "specops-cst[ml]"`). The supported interpreter range is Python
`>=3.9`.

## Reproduction command

The package installs a console entry point (with `qdrift-reproduce` as an alias):

```bash
cst-reproduce
```

The entry point is equivalent to `python -m qdriftforecast.reproduce`, or to
running `python make_paper_figures.py` from this directory. It pins
`KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` for deterministic numerics,
then executes `make_paper_figures.py`, which writes the figures, the table
bodies, and the diagnostic JSON summaries. Execution completes in seconds on a
laptop CPU.

## Package modules

The public API is exported from `qdriftforecast`:

| Module | Public functions | Role |
|---|---|---|
| `qdriftforecast/kernels.py` | `lower_shift`, `cst_feature_map`, `cst_operator_kernel`, `cst_scalar_gram`, `periodic_truncation_feature_map` | Causal spectral-truncation kernel and the periodic (Hashimoto et al.) baseline feature map |
| `qdriftforecast/detection.py` | `truncation_sweep`, `rank_truncation_grid`, `baseline_comparison` | Reconstruction detector over the kernel feature map; truncation/rank sweeps and the kernel comparison |
| `qdriftforecast/forecasting.py` | `forecast_baselines`, `cstk_forecast_skill` | Forecasting baselines and the CST forecaster's skill versus truncation order |
| `qdriftforecast/data.py` | `generate_synthetic_dataset`, `generate_correlated_drift_windows` | Seeded multi-qubit telemetry generator and the matched correlated-drift benchmark |

The unsupervised detector fits a rank-`k` nominal subspace in the kernel feature
space and scores a window by its residual energy off that subspace; at `n = 1` it
reduces to the commutative low-rank reconstruction detector.

## Regenerated figures and tables

`make_paper_figures.py` produces five vector PDF figures in
`submission/figures/`:

| Output | Content |
|---|---|
| `fig0_overview.pdf` | Overview of the causal spectral-truncation monitoring pipeline |
| `fig1_dynamics.pdf` | `T_1` trajectories, detrended autocorrelation, and the noncommutative drift signature by lag |
| `fig2_forecasting.pdf` | Forecasting MAE versus horizon, skill over persistence, and CST skill versus truncation order |
| `fig3_noncommutative.pdf` | Detection ROC-AUC versus truncation order, the (`n`, `k`) map, and the kernel comparison |
| `fig4_detection.pdf` | Operating characteristic on a held-out split and the over-complete collapse |

It writes three manuscript table bodies to `submission/tables/`:

| Output | Content |
|---|---|
| `forecasting_benchmark.tex` | Forecasting baselines (MAE at 1 and 8 steps, skill at 8 steps) |
| `detection_benchmark.tex` | Correlated-drift detection kernel comparison at fixed bottleneck rank `k = 8` |
| `truncation_sweep.tex` | Detection ROC-AUC versus truncation order `n` |

Diagnostic JSON summaries are written to `submission/code/generated_data/`:
`coherence_dynamics_summary.json`, `forecasting_benchmark.json`,
`detection_benchmark.json`, `reconstruction_rank_sweep.json`, and
`roc_overcomplete.json`.

## Determinism and seeds

The pipeline is reproducible to the byte. Every figure and table is computed live
from the seeded telemetry generator in `qdriftforecast.data`; identical inputs
yield identical outputs on every run.

- Telemetry-generator seeds for the forecasting benchmark (forecasting table,
  Fig. 2): `[0, 1, 2, 3, 4]`. Reported uncertainties are mean ± standard
  deviation across these five seeds.
- The correlated-drift detection benchmark (Figs. 3 and 4, detection table,
  truncation sweep) is evaluated over eight randomised 70/30 splits; the detector
  is fit on nominal windows only, so randomisation cannot leak a label it never
  uses. Reported uncertainties are mean ± standard deviation across the eight
  splits.
- The truncation sweep reports, per order `n`, the best ROC-AUC over bottleneck
  ranks `k ∈ {1, 2, 3, 5, 8, 12, 16}`; the kernel comparison fixes `k = 8`.
- `SOURCE_DATE_EPOCH` is pinned to `1700000000` before Matplotlib is imported,
  and `savefig` is called with `metadata={"CreationDate": None}`, so the PDF
  `CreationDate` and `ModDate` stamps stay fixed and the vector output is
  byte-identical across runs.
- `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` are set by the entry point
  to stabilise threaded numerics.

## Pinned dependencies

Dependency ranges are pinned in `pyproject.toml` to the following compatible
windows:

| Package | Range |
|---|---|
| numpy | `>=1.24,<3.0` |
| pandas | `>=2.0,<3.0` |
| scipy | `>=1.11,<2.0` |
| scikit-learn | `>=1.3,<2.0` |
| matplotlib | `>=3.7,<4.0` |
| torch (extra `ml`) | `>=2.1,<3.0` |

## Extend / tweak

Every experiment is driven by module-level configuration constants and by keyword
arguments to the generator/estimator functions. Nothing is hard-coded inside the
numerics.

### Tunable parameters and CLI flags

- **CLI.** `cst-reproduce` (alias `qdrift-reproduce`, or `python -m
  qdriftforecast.reproduce`) takes no positional arguments; it sets
  `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` and runs
  `make_paper_figures.py`. To change the experiment you edit the constants below
  (or call the library functions directly) rather than passing flags.

- **Global experiment constants** (top of `make_paper_figures.py`):

  | Constant | Meaning | Default |
  |---|---|---|
  | `N_QUBITS` | qubits in the telemetry generator | `5` |
  | `N_STEPS` | telemetry time steps | `200` |
  | `DT_HOURS` | sampling interval (hours) | `0.5` |
  | `SEQ_LEN` | forecasting window length | `32` |
  | `HORIZON` | forecasting horizon (steps) | `8` |
  | `T1_THRESHOLD` | T1 alarm threshold (µs) | `50.0` |
  | `FORECAST_SEEDS` | seeds averaged for the forecasting benchmark | `[0,1,2,3,4]` |
  | `DYN_SEED` | seed for the dynamics figure | `42` |
  | `DRIFT_N_EACH` | windows per class in the detection benchmark | `700` |
  | `DRIFT_SEED` | seed for the correlated-drift benchmark | `0` |
  | `N_GRID` | truncation orders `n` swept for detection | `[1,2,...,16]` |
  | `K_GRID` | bottleneck ranks `k` swept | `[1,2,3,5,8,12,16]` |
  | `N_SPLITS` | randomized splits for the detection AUC | `8` |
  | `N_CST` | truncation order used for the ROC / kernel comparison | `2` |
  | `N_PERIODIC` | order of the periodic (Hashimoto) baseline | `2` |

- **Data generators** (`qdriftforecast.data`): `generate_synthetic_dataset(n_qubits, n_steps, seed, ...)`
  and `generate_correlated_drift_windows(n_each, seq_len, seed, ...)`. The
  correlated-drift construction exposes the lag structure that carries the
  noncommutative signal; changing the lag/coupling arguments changes where the
  optimal order `n` lands.

- **Kernel** (`qdriftforecast.kernels`): `cst_feature_map(..., n=...)` and
  `cst_operator_kernel(..., n=...)` take the truncation order `n` directly;
  `n=1` is the commutative instantaneous covariance, `n>=2` adds ordered lagged
  cross-channel terms. `periodic_truncation_feature_map` is the Fourier
  (Hashimoto et al.) baseline.

### Adding new parameters or inputs

- **New telemetry channel / device model.** Add columns in
  `qdriftforecast.data` (extend `FEATURE_COLS` and the per-step generator loop)
  or replace `generate_synthetic_dataset` with a loader for archival calibration
  logs; the detector and forecaster consume any `(n_windows, seq_len, n_features)`
  array, so no downstream change is required.
- **New detector / baseline kernel.** Implement a feature map with the same
  `(windows) -> features` signature as `cst_feature_map` and register it in
  `qdriftforecast.detection.baseline_comparison`; it will appear in the
  comparison table automatically.
- **New sweep axis.** Add the axis to `N_GRID`/`K_GRID` (or to a new grid) and to
  the corresponding plotting/table block in `make_paper_figures.py`.

### Plugging into other projects

Import the library directly:

```python
from qdriftforecast.kernels import cst_feature_map
from qdriftforecast.detection import reconstruction_scores

feats = cst_feature_map(windows, n=2)          # ordered, lagged cross-channel features
scores = reconstruction_scores(nominal, test, n=2, k=8)  # unsupervised drift score
```

The kernel and detector are training-free and depend only on NumPy/SciPy, so they
drop into any monitoring or anomaly-detection pipeline that can provide windowed
multivariate telemetry.

## Cite this work

If you use this package or its results, please cite the paper:

```bibtex
@article{huynh2026cst,
  author  = {Huynh, Molena},
  title   = {Causal Spectral-Truncation Kernels: Noncommutativity Certifies
             Drift Detectability in Quantum Hardware},
  year    = {2026},
  note    = {Part of the spectral-truncation operators (SpecOps) program},
}
```

Software metadata is also provided in [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
