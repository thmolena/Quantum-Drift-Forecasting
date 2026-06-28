# qdriftforecast

Reproduction package for *Causal spectral-truncation kernels: noncommutativity
certifies drift detectability in quantum hardware*. The package deterministically
regenerates the manuscript figures and result tables from a seeded telemetry
generator and a controlled correlated-drift benchmark. Distribution name:
`qdriftforecast` (version 2.0.0).

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
pip install qdriftforecast
```

Installation from a local checkout of this directory is equivalent:

```bash
cd submission/code
pip install .
```

The figure-and-table pipeline runs CPU-only on NumPy, pandas, SciPy,
scikit-learn, and Matplotlib. PyTorch is not required for the figure/table
pipeline; it is available through the optional `ml` extra
(`pip install "qdriftforecast[ml]"`). The supported interpreter range is Python
`>=3.10`.

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

The package installs a single console entry point:

```bash
qdrift-reproduce
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

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
