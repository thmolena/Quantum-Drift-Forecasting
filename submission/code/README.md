# qdriftforecast

Reproduction package for *Objective-aware sequence modelling for drift
forecasting and anomaly detection in quantum hardware*. The package
deterministically regenerates the manuscript figures and result tables from the
seeded telemetry generator and the recorded notebook metrics. Distribution name:
`qdriftforecast`.

## Installation

```bash
pip install qdriftforecast
```

Installation from a local checkout of this directory is equivalent:

```bash
cd submission/code
pip install .
```

The core figure-and-table pipeline runs on CPU using NumPy, pandas, SciPy,
scikit-learn, and Matplotlib. PyTorch is required only for re-training the
sequence models in the notebooks and is available through the optional `ml`
extra (`pip install "qdriftforecast[ml]"`).

## Reproduction command

The package installs a single console entry point:

```bash
qdrift-reproduce
```

The entry point is equivalent to `python -m qdriftforecast.reproduce`. It pins
`KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` for deterministic numerics,
then executes `make_paper_figures.py`, which writes the figures, the table
bodies, and the diagnostic JSON summaries. Execution completes in seconds on a
laptop CPU.

## Regenerated figures and tables

`make_paper_figures.py` produces five vector PDF figures in
`submission/figures/`:

| Output | Content |
|---|---|
| `fig0_overview.pdf` | Method-overview schematic of the objective-aware benchmark |
| `fig1_dynamics.pdf` | Coherence ($T_1$) trajectories and detrended autocorrelation |
| `fig2_forecasting.pdf` | Forecasting MAE versus horizon and skill over persistence |
| `fig3_benchmark.pdf` | Sequence-model incident detection, cross-domain average, parameter counts |
| `fig4_anomaly.pdf` | Reconstruction-detector ROC curve and ROC-AUC versus bottleneck rank |

It writes three manuscript table bodies to `submission/tables/`:

| Output | Content |
|---|---|
| `forecasting_benchmark.tex` | Forecasting baselines (MAE at 1 and 8 steps, skill) |
| `thermal_benchmark.tex` | Thermal-failure incident detection by sequence model |
| `cross_domain_benchmark.tex` | Three-dataset average and periodic-regime ROC-AUC |

Diagnostic summaries are written to `submission/code/generated_data/` as
`coherence_dynamics_summary.json`, `forecasting_benchmark.json`,
`reconstruction_rank_sweep.json`, and `sequence_notebook_metrics.json`.

## Determinism and seeds

The pipeline is reproducible to the byte. Figures 1, 2, and 4 and the
forecasting table are computed live from the seeded telemetry generator in
`qdriftforecast.data`; identical inputs yield identical outputs on every run.

- Telemetry-generator seeds for the forecasting benchmark (Table 1, Fig. 2):
  `[0, 1, 2, 3, 4]`.
- Telemetry seed for the reconstruction study (Fig. 4): `42`, evaluated over
  eight randomised 70/30 splits drawn from `numpy.random.default_rng(0)`.
- Coherence-dynamics figure (Fig. 1) seed: `42`.
- `SOURCE_DATE_EPOCH` is pinned to `1700000000` before Matplotlib is imported,
  and `savefig` is called with `metadata={"CreationDate": None}`, so the PDF
  `CreationDate` and `ModDate` stamps stay fixed and the vector output is
  byte-identical across runs.
- `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` are set by the entry point
  to stabilise threaded numerics.

Figure 3 and the sequence-model tables render metrics transcribed verbatim from
the executed notebooks (`notebooks/rnn_drift_forecast.ipynb`,
`notebooks/quantum_drift_combined.ipynb`,
`notebooks/transformer_calibration.ipynb`); these values are fixed in the
package and are reproduced rather than re-trained.

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

The supported interpreter range is Python `>=3.10`.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
