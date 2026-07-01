# specops-cst

Reproduction package for *Causal Spectral-Truncation Kernels: Noncommutativity
Certifies Drift Detectability in Quantum Hardware* (BibTeX key `huynh2026cst`).
The package deterministically regenerates the manuscript figures and result
tables from a seeded telemetry generator and a controlled correlated-drift
benchmark. Distribution name: `specops-cst` (version 2.0.0); the importable
package is `qdriftforecast`. Part of the **spectral-truncation operators
(SpecOps)** program, which treats spectral truncation of operators as a unifying
primitive for resource-accounted, CPU-reproducible quantum-AI methods.

The central contribution is the **causal spectral-truncation (CST) kernel**, a
C\*-algebra-valued positive-definite kernel on telemetry windows,
`K_n(X, Y) = Σ_{τ<n} w_τ · Xᵀ S^τ Y`, with `S` the causal lower shift. Its
truncation order `n` controls the noncommutativity of the cross-channel products
it represents: `n = 1` is the commutative instantaneous covariance, `n ≥ 2` adds
asymmetric lagged cross-channel terms. It is the causal counterpart of the
periodic spectral-truncation kernels of Hashimoto et al. (NeurIPS 2024,
arXiv:2405.17823).

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
