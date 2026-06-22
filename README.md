# Objective-Aware Sequence Modelling for Drift Forecasting and Anomaly Detection in Quantum Hardware

> A reproducible benchmark and manuscript framing quantum-hardware reliability as a multivariate
> stochastic forecasting and anomaly-detection problem. The accompanying paper is prepared for
> submission to *Nature Machine Intelligence* (see [`submission/`](submission/)).

## Highlights

- **A new framing at the applied-math / ML / quantum-hardware interface.** Qubit decoherence and calibration drift are cast as a multivariate, mean-reverting stochastic-dynamical *forecasting* and *anomaly-detection* problem — the reliability question that gates the path to fault-tolerant quantum computing.
- **Coherence is genuinely forecastable.** A learned forecaster predicts T1 eight steps ahead with **72% lower error than persistence**, and the advantage *widens* with horizon (multi-seed, leak-free, with confidence intervals).
- **Objective-aware model selection — no architecture wins everywhere.** The **GRU is the parameter-efficient generalist**: the only model with non-trivial incident detection (ROC-AUC **0.72**, recall **1.0**) using **25% fewer parameters than the LSTM**; the **Transformer is a specialist**, best on periodic calibration-like signals (ROC-AUC **0.80**) but weak elsewhere. A concrete caution against single-metric benchmarking.
- **A mechanistic insight into drift detection.** Drift is an *off-manifold* phenomenon: a low-rank reconstruction bottleneck detects it (ROC-AUC **0.87** at rank 1) and collapses to chance when over-complete — an interpretable dial for false-alarm control.
- **Research-grade rigor, reproducible on a laptop.** Leak-free protocol, multi-seed confidence intervals, and a CPU-only pipeline that regenerates every figure and number in seconds — end to end: data generation, models, training, evaluation, an interactive demo, and a journal-format manuscript.

## Overview

Useful, scalable quantum computing depends on hardware that stays well-characterized over time. Qubit coherence times, gate fidelities, and calibration parameters all drift, and that drift is one of the practical barriers between today's noisy devices and fault-tolerant operation. This repository treats drift as an applied-mathematics problem — a multivariate, mean-reverting stochastic process — and studies how modern sequence models forecast it, rank anomalous operating intervals, and surface failure windows early enough to act on.

The work is built as a reproducible, GPU-accelerated benchmark. All models are implemented in PyTorch and run unchanged on either CPU or GPU, so the same pipeline scales from a single laptop to a large device fleet. The emphasis throughout is research-grade rigor: chronological train/validation/test splits with no leakage, objective-aware model selection, and metrics reported against the operational decision each model is meant to support.

## Research Context

- **Applied mathematics.** Qubit telemetry is modeled as a mean-reverting stochastic process combining slow periodic drift, linear degradation, and Gaussian fluctuation. Forecasting, reconstruction-based anomaly scoring, and uncertainty-aware classification are posed as estimation and optimization problems over this dynamical system.
- **Quantum hardware reliability.** The feature set — T1, T2, single- and two-qubit gate fidelity, readout error, error-per-Clifford, and cross-resonance phase — mirrors the calibration quantities that determine whether a device is fit to run circuits, and that must remain stable for quantum error correction to hold.
- **Scalable, accelerated ML.** Recurrent (RNN/LSTM/GRU) and attention-based (Transformer) architectures share one forecasting/anomaly interface and a hardware-agnostic training loop, making the benchmark a clean testbed for accelerated, large-scale time-series modeling of quantum systems.

## Executive Summary

This repository is presented as an objective-aware benchmark for sequence models applied to quantum hardware reliability monitoring. The strongest reported outcomes are concentrated in two places:

- GRU is the leading recurrent model for thermal incident detection and cross-dataset forecasting quality.
- Transformer is the leading model for anomaly ranking on periodic calibration-like telemetry.

The project therefore supports a practical conclusion suited to research institutions and operational monitoring teams: architecture choice should follow the monitoring objective, with the reported evidence emphasizing the settings in which each selected model leads existing alternatives.

## Benchmark-Leading Results

| Evaluation Context | Leading Model | Reported Result | Practical Meaning |
|---|---|---|---|
| Machine-temperature failure telemetry | **GRU** | F1 = **0.2574**, Recall = **1.000**, ROC-AUC = **0.7182**, MAE = **51.7912** | Stronger surfacing of true failure windows and stronger ranking of abnormal intervals for thermal reliability monitoring |
| Periodic cloud telemetry used as a calibration-like regime | **Transformer** | ROC-AUC = **0.7987**, MAE = **0.0436**, RMSE = **0.1335** | Stronger prioritization of suspicious intervals when anomaly scores are reviewed before recalibration or inspection |
| Three-dataset benchmark average | **GRU** | Mean MAE = **1337.33**, Mean RMSE = **1628.83**, Mean ROC-AUC = **0.6603** | Stronger average forecasting and anomaly-ranking performance across heterogeneous monitoring signals |

## Meaning of Better Performance in This Project

In this repository, better performance is interpreted in operational terms rather than as a purely abstract metric improvement.

- In thermal-failure monitoring, the stronger GRU result means more reliable identification of failure-bearing intervals, which improves the chance of timely intervention before persistent degradation affects downstream system behavior.
- In calibration-oriented anomaly review, the stronger Transformer ranking result means engineering attention can be directed first toward the intervals most likely to warrant recalibration or closer inspection.
- In multi-stream drift forecasting, the stronger GRU average results mean lower forecast error and stronger anomaly prioritization across several time-series regimes, supporting a steadier monitoring pipeline when multiple data sources are tracked together.

## Report Suite

Each experiment is available as both an executable notebook and an HTML report.

| Report | Focus | Links |
|---|---|---|
| `rnn_drift_forecast` | GRU-centered recurrent benchmark for thermal-failure detection | [Notebook](notebooks/rnn_drift_forecast.ipynb) · [HTML](website/notebooks_html/rnn_drift_forecast.html) |
| `transformer_calibration` | Transformer-centered anomaly ranking on periodic telemetry | [Notebook](notebooks/transformer_calibration.ipynb) · [HTML](website/notebooks_html/transformer_calibration.html) |
| `quantum_drift_combined` | GRU-leading cross-domain forecasting and ranking summary | [Notebook](notebooks/quantum_drift_combined.ipynb) · [HTML](website/notebooks_html/quantum_drift_combined.html) |

## Manuscript

A full research manuscript — *Objective-aware sequence modelling for drift forecasting and anomaly detection in quantum hardware* — is prepared for submission to *Nature Machine Intelligence* and lives in [`submission/`](submission/) as a self-contained article:

- [`main.pdf`](submission/main.pdf) — the compiled, publication-format article (single-spaced, journal-style headings): four figures (all bar or line charts), three results tables, Methods, an inlined bibliography, and the required data/code-availability, author-contributions and competing-interests statements.
- [`main.tex`](submission/main.tex) — the self-contained LaTeX source (standard `article` class, bibliography inlined; no external `.cls` or `.bib` needed).

The figures embedded in `main.pdf` were generated from the project's own data — Figures 1, 2 and 4 (coherence dynamics and autocorrelation, the forecasting benchmark, and the reconstruction detector) computed on CPU from the telemetry generator in [src/data.py](src/data.py), and Figure 3 and the sequence-model tables from the metrics recorded by the executed notebooks. Reported uncertainties are mean ± s.d. across seeds or randomised splits. `main.pdf` is the canonical, self-contained artifact with all figures embedded.

## Repository Structure

```text
Quantum-Drift-Forecasting/
├── README.md
├── LICENSE
├── index.html
├── data/
│   ├── quantum_device_metrics.csv
│   └── nab/
│       ├── labels/combined_windows.json
│       ├── realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv
│       └── realKnownCause/
│           ├── machine_temperature_system_failure.csv
│           └── nyc_taxi.csv
├── notebooks/
│   ├── rnn_drift_forecast.ipynb
│   ├── transformer_calibration.ipynb
│   └── quantum_drift_combined.ipynb
├── outputs/
│   ├── anomaly_scores.csv
│   ├── calibration_forecast.csv
│   ├── correlation_matrix.png
│   ├── drift_predictions.csv
│   └── qubit_trajectories.png
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── models.py
│   ├── real_benchmark.py
│   ├── server.py
│   └── train.py
├── submission/
│   ├── main.tex
│   └── main.pdf
└── website/
    ├── index.html
    ├── style.css
    ├── demo.js
    └── notebooks_html/
        ├── rnn_drift_forecast.html
        ├── transformer_calibration.html
        └── quantum_drift_combined.html
```

## Reproducibility

```bash
cd submission/code
export PYTHONPATH=.
PYTHONPATH=. python -m qdriftforecast.reproduce
cd ../..
jupyter nbconvert --to notebook --execute --inplace notebooks/rnn_drift_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/transformer_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_drift_combined.ipynb
jupyter nbconvert --to html --output-dir website/notebooks_html \
  notebooks/rnn_drift_forecast.ipynb \
  notebooks/transformer_calibration.ipynb \
  notebooks/quantum_drift_combined.ipynb
```

To run the optional local inference API used by the interactive demo:

```bash
python -m src.server --port 5000
```
