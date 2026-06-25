# Objective-Aware Sequence Modelling for Drift Forecasting and Anomaly Detection in Quantum Hardware

Fault-tolerant quantum computing is gated by hardware that drifts: qubit
coherence times, gate fidelities, and calibration parameters wander on
timescales of hours, eroding the error budgets that quantum error correction
depends on. This repository recasts the reliability problem as a machine-learning
task—forecasting a multivariate, mean-reverting stochastic process and ranking
anomalous operating intervals—and studies it through an objective-aware benchmark
of sequence models. Evaluation spans a physically motivated multi-qubit telemetry
generator and three heterogeneous real-world regimes under one leakage-free
protocol. Coherence proves forecastable; model selection proves
objective-dependent; and drift is shown to occupy an off-manifold, low-rank
structure that a reconstruction detector exploits. The full pipeline is
hardware-agnostic, runs identically on CPU or GPU, and regenerates every figure
and metric deterministically in seconds. The accompanying manuscript is prepared
for submission to *Nature Machine Intelligence* (see [`submission/`](submission/)).

## Principal Contributions

1. **A formalised reliability benchmark.** Quantum-hardware reliability is posed
   as a joint forecasting-and-anomaly-detection problem over a physically
   motivated, reproducible telemetry generator, released as a single
   leakage-free benchmark that evaluates four sequence-model families on the
   synthetic task and on three heterogeneous real-world regimes.
2. **Objective-aware model selection.** Empirical evidence shows that
   architecture choice must follow the monitoring objective: the GRU is a
   parameter-efficient generalist and the only model to achieve non-trivial
   incident detection, whereas the Transformer is a specialist that leads on
   periodic calibration-like signals while generalising weakly.
3. **A mechanistic drift detector.** A reconstruction-based detector identifies
   the reconstruction-bottleneck rank as the control knob separating drift from
   nominal behaviour, yielding an interpretable monitor that requires training
   only on nominal windows.

## Main Results

The following figures are transcribed verbatim from the manuscript and the
project's generated tables. Uncertainties are mean ± standard deviation across
seeds or randomised splits.

**Forecasting (Table 1; five telemetry-generator seeds, leakage-free 80/20
split).** Mean absolute error (MAE) on the $T_1$ relaxation time, in microseconds
(µs):

| Forecaster | MAE @ 1 step (µs) | MAE @ 8 steps (µs) | Skill @ 8 steps |
|---|---|---|---|
| Persistence | 2.44 ± 0.16 | 6.79 ± 0.18 | — |
| Climatology | 13.01 ± 0.05 | 12.40 ± 0.05 | −83% |
| AR-ridge ($T_1$ history) | 1.85 ± 0.07 | 2.19 ± 0.06 | 68% |
| Multivariate ridge | **1.80 ± 0.09** | **1.88 ± 0.06** | **72%** |

The multivariate-ridge forecaster predicts $T_1$ eight steps ahead with 72%
lower error than persistence, and the advantage widens with horizon.

**Thermal-failure incident detection (Extended Data Table; machine-temperature
telemetry).** MAE and RMSE in microseconds (µs); ROC-AUC and F1 on the unit
scale; parameter counts as integers:

| Model | MAE (µs) | RMSE (µs) | F1 | ROC-AUC | Params |
|---|---|---|---|---|---|
| Elman RNN | 61.37 | 64.93 | 0.000 | 0.408 | **5,645** |
| LSTM | **51.48** | **55.00** | 0.000 | 0.423 | 116,845 |
| GRU | 51.79 | 55.35 | **0.257** | **0.718** | 87,949 |

The GRU is the only model to surface incidents, with F1 = 0.257 (0.2574),
ROC-AUC = 0.718 (0.7182), and recall = 1.0 (precision = 0.15)—a 75.9% relative
improvement in ranking quality over the Elman RNN—using 25% fewer parameters
than the LSTM (87,949 versus 116,845).

**Cross-domain forecasting and detection (Extended Data Table; three-dataset
average and periodic regime).** MAE and RMSE on the dataset's native scale;
ROC-AUC on the unit scale:

| Model | MAE | RMSE | ROC-AUC | Periodic regime | ROC-AUC |
|---|---|---|---|---|---|
| GRU | **1337.3** | **1628.8** | **0.660** | Transformer | **0.799** |
| LSTM | 1528.8 | 1895.0 | 0.628 | GRU (avg.) | 0.660 |
| Transformer | 1436.2 | 1791.6 | 0.196 | — | — |

The GRU attains the strongest three-dataset average (MAE = 1337.3,
RMSE = 1628.8, ROC-AUC = 0.660), whereas the Transformer leads on periodic
calibration-like telemetry (ROC-AUC = 0.799, i.e. 0.7987) while generalising
weakly across regimes (ROC-AUC = 0.196).

**Reconstruction detector (Fig. 4; telemetry seed 42, eight randomised 70/30
splits).** Detection ROC-AUC on the unit scale is strongest at a rank-one
bottleneck (0.870 ± 0.023) and collapses toward chance as the code becomes
over-complete. A rank-three bottleneck separates regimes with ROC-AUC = 0.72 on
a single held-out split (0.70 ± 0.05 across the eight splits). Drift is therefore
an off-manifold phenomenon, and the bottleneck rank governs false-alarm
sensitivity.

## Installation

```bash
pip install qdriftforecast
```

The packaged reproduction pipeline (`qdriftforecast`) resides at
[`submission/code`](submission/code) and depends on NumPy, pandas, SciPy,
scikit-learn, and Matplotlib. PyTorch is required only for re-training the
sequence models in the notebooks and is available through the optional `ml`
extra (`pip install "qdriftforecast[ml]"`).

## Reproduction

After installation, the figures and tables regenerate deterministically through
the console entry point:

```bash
qdrift-reproduce
```

The command pins `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1`, then
executes the figure-and-table pipeline, writing five vector PDF figures to
`submission/figures/`, three LaTeX table bodies to `submission/tables/`, and
diagnostic JSON summaries to `submission/code/generated_data/`. Determinism,
seeds, and the regenerated artifacts are documented in
[`submission/code/README.md`](submission/code/README.md).

The interactive notebooks and HTML reports regenerate with:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/rnn_drift_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/transformer_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_drift_combined.ipynb
jupyter nbconvert --to html --output-dir website/notebooks_html \
  notebooks/rnn_drift_forecast.ipynb \
  notebooks/transformer_calibration.ipynb \
  notebooks/quantum_drift_combined.ipynb
```

The optional local inference API used by the interactive demo runs with
`python -m src.server --port 5000`.

## Repository Layout

```text
Quantum-Drift-Forecasting/
├── README.md
├── LICENSE
├── index.html
├── data/                         # synthetic telemetry and real-world (NAB) regimes
│   ├── quantum_device_metrics.csv
│   └── nab/
├── notebooks/                    # executable experiment notebooks
│   ├── rnn_drift_forecast.ipynb
│   ├── transformer_calibration.ipynb
│   └── quantum_drift_combined.ipynb
├── outputs/                      # cached anomaly scores, forecasts, plots
├── src/                          # data generator, models, training, evaluation, server
│   ├── data.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── real_benchmark.py
│   └── server.py
├── submission/                   # manuscript and reproduction package
│   ├── main.tex
│   ├── main.pdf
│   ├── figures/                  # regenerated vector PDF figures
│   ├── tables/                   # regenerated LaTeX table bodies
│   └── code/                     # the qdriftforecast package
│       ├── pyproject.toml
│       ├── LICENSE
│       ├── README.md
│       ├── make_paper_figures.py
│       ├── generated_data/
│       └── qdriftforecast/
└── website/                      # static site and HTML notebook reports
    ├── index.html
    ├── style.css
    ├── demo.js
    └── notebooks_html/
```

## Manuscript

A self-contained research manuscript—*Objective-aware sequence modelling for
drift forecasting and anomaly detection in quantum hardware*—is prepared for
submission to *Nature Machine Intelligence* and lives in
[`submission/`](submission/):

- [`main.pdf`](submission/main.pdf): the compiled, journal-format article with
  embedded figures, three results tables, Methods, an inlined bibliography, and
  the data/code-availability, author-contributions, and competing-interests
  statements.
- [`main.tex`](submission/main.tex): the self-contained LaTeX source (standard
  `article` class, bibliography inlined).

Figures 1, 2, and 4 and the forecasting table are computed live from the seeded
telemetry generator in [`submission/code/qdriftforecast/data.py`](submission/code/qdriftforecast/data.py);
Figure 3 and the sequence-model tables render metrics recorded by the executed
notebooks.

## Citation

```bibtex
@article{huynh2026qdriftforecast,
  title   = {Objective-aware sequence modelling for drift forecasting and
             anomaly detection in quantum hardware},
  author  = {Huynh, Molena},
  year    = {2026},
  note    = {North Carolina State University. Correspondence:
             molena.huynh@jmp.com},
}
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
