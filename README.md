# Quantum-Drift-Forecasting

## Overview

This repository presents a research-oriented time-series and Transformer-based modeling framework for **quantum hardware behavior, calibration drift, and noise analysis**. The project integrates:

- Recurrent Neural Networks (RNN, LSTM, GRU) for sequential calibration modeling
- Attention-based architectures and Transformers for long-range temporal structure
- Uncertainty-aware forecasting and anomaly detection on quantum device time series
- Drift-aware decision support for autonomous recalibration

The primary objective is to examine how **adaptive sequential models** can detect, predict, and quantify temporal variation in quantum-system measurement data, distinguishing transient fluctuation from meaningful degradation.

The workflow ingests hardware diagnostic streams—readout signals, gate fidelities, coherence times, and calibration histories—and applies learned temporal representations to support **early-warning detection of calibration failure**, performance degradation classification, and recalibration scheduling.

This work highlights how **statistical learning on quantum hardware telemetry** may contribute to high-impact objectives in quantum computing infrastructure, including:

- Autonomous hardware calibration management
- Noise characterization and drift quantification
- Reliable operation of real quantum devices over extended deployment periods

---

## Motivation

Quantum hardware produces **evolving, noisy, time-dependent signals**. Coherence times drift. Gate fidelities degrade. Readout errors shift with environmental perturbations and device aging. These are fundamentally **statistical problems** as much as physical ones.

Classical control systems rely on periodic recalibration schedules that ignore the temporal structure of degradation. This leads to either over-calibration—wasting computational resources—or under-calibration—allowing performance to degrade below acceptable thresholds before intervention.

Sequential learning models provide a principled alternative:

- **Recurrent networks (RNN/LSTM/GRU)** capture short- and medium-range temporal dependencies in calibration histories and device metric streams.
- **Transformer architectures** model long-range correlations across qubit channels and time, using self-attention to identify cross-qubit drift patterns.
- **Uncertainty-aware forecasting** methods distinguish confident trend estimates from high-variance transient noise, enabling more precise decision thresholds.
- **Anomaly detection pipelines** identify early-warning signatures before calibration failure becomes operationally significant.

By combining these components, this project provides an **experimental testbed for data-driven quantum hardware management**, with direct relevance to U.S. national interests in quantum computing infrastructure reliability.

---

## System Architecture

The pipeline contains four major stages.

### 1. Hardware Telemetry Collection and Preprocessing

Raw quantum device metrics are ingested as **multivariate time series** across qubit channels. Key signals include:

- **T1 relaxation time** (qubit energy decay, microseconds)
- **T2 coherence time** (qubit phase coherence, microseconds)
- **1-qubit gate fidelity** (average randomized benchmarking fidelity)
- **2-qubit gate fidelity** (cross-resonance and echoed cross-resonance gates)
- **Readout assignment error** (per-qubit measurement fidelity)
- **Gate error rates** (error per Clifford gate)
- **Cross-resonance phase** (tunable coupling parameter drift)

Preprocessing includes windowed normalization, missing value imputation, and sequence segmentation for supervised training.

---

### 2. Recurrent Sequence Models (RNN, LSTM, GRU)

Recurrent architectures serve as the baseline sequential modeling tier.

- **Vanilla RNN** establishes the minimum-complexity autoregressive baseline.
- **Long Short-Term Memory (LSTM)** networks address the vanishing gradient problem and capture multi-scale temporal dependencies critical for calibration drift modeling.
- **Gated Recurrent Units (GRU)** provide a computationally efficient alternative with comparable performance on medium-horizon forecasting tasks.

All recurrent models are trained on windowed calibration sequences and evaluated on next-step prediction accuracy, multi-step forecasting horizon, and drift-event recall.

---

### 3. Attention-Based and Transformer Architectures

Transformer models capture **long-range temporal structure** that recurrent architectures struggle to represent efficiently.

- **Positional encodings** inject time-step information into the Transformer input layer, enabling the model to reason about calibration periodicity and seasonal hardware behavior.
- **Multi-head self-attention** identifies cross-channel correlations between qubit pairs, uncovering shared drift dynamics across device topology.
- **Encoder-only Transformers** are used for anomaly scoring and classification of drift state (stable / slow drift / rapid degradation).
- **Encoder-Decoder Transformers** support multi-step sequence forecasting for predictive maintenance scheduling.

---

### 4. Uncertainty Quantification and Anomaly Detection

The final modeling tier applies **probabilistic forecasting** and **threshold-based detection** to support operational decision-making.

- **Monte Carlo Dropout** provides epistemic uncertainty estimates at inference time, distinguishing confident predictions from high-uncertainty regions.
- **Conformal prediction intervals** offer calibrated coverage guarantees on forecast error bounds.
- **Reconstruction-based anomaly scoring** uses Transformer encoder residuals to assign per-time-step anomaly scores.
- **Early-warning classifiers** are trained on labeled drift precursor windows to flag degradation signatures before threshold violations.

---

## Example Applications

### Autonomous Recalibration Scheduling

Predict when T1 or T2 coherence times will cross minimum operational thresholds, enabling **proactive recalibration** rather than reactive emergency intervention. Reduces calibration overhead while maintaining target gate fidelities.

---

### Real-Time Gate Fidelity Monitoring

Deploy streaming inference models on live hardware telemetry to detect **gate fidelity degradation events** with sub-minute latency, supporting in-circuit error monitoring pipelines.

---

### Cross-Qubit Drift Correlation Analysis

Identify topological drift patterns across multi-qubit processors using **attention weight visualization**, revealing common-mode environmental disturbances and correlated failure modes across qubit clusters.

---

### Readout Error Trend Forecasting

Model the temporal evolution of **readout assignment errors** across calibration cycles to predict when discriminator classifiers require retraining, improving measurement fidelity without full hardware recalibration.

---

## Repository Structure

The repository is organized to separate core modeling code, datasets, reproducible notebooks, generated outputs, and demonstration assets.

```text
Quantum-Drift-Forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── index.html                        ← GitHub Pages entry point (root)
├── data/
│   └── quantum_device_metrics.csv    ← synthetic multi-qubit telemetry dataset
├── notebooks/
│   ├── rnn_drift_forecast.ipynb      ← RNN / LSTM / GRU drift modeling
│   ├── transformer_calibration.ipynb ← Transformer calibration & anomaly detection
│   └── quantum_drift_combined.ipynb  ← unified pipeline and comparative analysis
├── outputs/
│   ├── drift_predictions.csv
│   ├── anomaly_scores.csv
│   └── calibration_forecast.csv
├── src/
│   ├── __init__.py
│   ├── data.py                       ← data generation and preprocessing
│   ├── models.py                     ← RNN, LSTM, GRU, Transformer definitions
│   ├── train.py                      ← training loop and CLI
│   ├── evaluate.py                   ← evaluation metrics and visualization
│   └── server.py                     ← Flask inference API
└── website/
    ├── README_SITE.md
    ├── demo.js
    ├── index.html                    ← local dev copy (served from website/)
    ├── style.css
    └── notebooks_html/
        ├── rnn_drift_forecast.html
        ├── transformer_calibration.html
        └── quantum_drift_combined.html
```

---

## Project Summary

This project serves as a reproducible research artifact for studying **sequential learning and Transformer-based modeling on quantum hardware telemetry**. The current implementation combines multivariate time-series preprocessing, recurrent and attention-based forecasting architectures, uncertainty quantification, and anomaly detection to evaluate how learned temporal representations can support drift-aware decision support in quantum computing systems.

In addition to model code, the repository includes fully executed notebooks, exported HTML reports, and an interactive web demo interface to support transparent communication of methods and outcomes. The overall design targets interdisciplinary audiences in **quantum computing, machine learning, and statistical signal processing**.

---

## Key Features

### End-to-End Temporal Modeling Pipeline

The codebase links synthetic hardware data generation, multivariate preprocessing, model training across RNN/LSTM/GRU/Transformer architectures, uncertainty estimation, and anomaly scoring in a unified pipeline. This structure enables controlled experiments on the relative sensitivity of model families to drift timescale, noise magnitude, and calibration signal complexity.

### Multi-Architecture Comparison

Training and evaluation are standardized across all model families, enabling direct performance comparison on identical train/test splits. Metrics include mean absolute error, root mean squared error, drift event recall, anomaly detection F1, and calibration reliability diagrams.

### Uncertainty-Aware Forecasting

Monte Carlo Dropout and conformal prediction intervals are implemented for all model families, providing principled coverage guarantees and epistemic uncertainty estimates that enable more reliable operational thresholds.

### Lightweight and Portable Implementation

The implementation targets standard Python environments with minimal dependencies (NumPy, PyTorch, scikit-learn). PyTorch is the primary deep learning backend; all models are implemented natively without specialized quantum simulation libraries, making the codebase accessible on standard CPU or GPU hardware.

### Interactive Web Demo

A static web demo provides a browser-side drift simulation and forecasting visualization. An optional Flask inference endpoint serves model predictions from locally trained checkpoints, with graceful fallback to browser-side computation when the API is unavailable.

---

## Quick Start

The following steps reproduce a baseline workflow, from data generation and model training to API-based inference.

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train the LSTM drift forecasting model.

```bash
python -m src.train \
  --model lstm \
  --sequence-length 32 \
  --horizon 8 \
  --epochs 30 \
  --model-path model_lstm.pt
```

3. Start the prediction API.

```bash
python -m src.server
```

4. Optionally serve the website demo locally in another terminal.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` and use the interactive demo to visualize drift predictions and anomaly scores.

> **GitHub Pages**: the live demo is published at
> `https://mohuyn.github.io/Quantum-Drift-Forecasting/`
> The root `index.html` is the GitHub Pages entry point; it loads assets from `website/`.

---

## Roadmap

### Near-Term

Expand benchmark coverage to additional qubit metric channels, with structured experiment tracking for model convergence, forecasting horizon sensitivity, and drift event recall across noise regimes.

### Mid-Term

Incorporate real hardware telemetry from open quantum device datasets (IBM Quantum, Rigetti) alongside synthetic benchmarks. Develop richer calibration graph representations that capture coupling map topology and cross-qubit drift dependencies.

### Longer-Term

Integrate hardware-aware inference pipelines that run directly within quantum control systems, and extend to online learning settings where model parameters adapt continuously as new calibration data arrives.

---

## Contributing

Contributions are welcome in the form of additional model architectures, improved uncertainty quantification methods, expanded benchmark datasets, ablation studies, and reproducibility upgrades. Pull requests with concise technical rationale, experimental assumptions, and validation evidence are especially valuable for maintaining a rigorous and extensible research codebase.

---

## License

This project is released under the terms of the LICENSE file in this repository.

