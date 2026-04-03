# Quantum-Drift-Forecasting

## Overview

This repository is a reproducible benchmark of adaptive sequence models for real operational time series. The core contribution is not a generic deep-learning demo; it is a comparative study of when recurrent gating and long-context self-attention provide technically defensible gains on CPU-runnable public datasets.

The three primary reports are fully executed Jupyter notebooks built on the Numenta Anomaly Benchmark (NAB):

- `machine_temperature_system_failure` for incident-aware thermal drift forecasting
- `ec2_cpu_utilization_24ae8d` for long-context cloud telemetry forecasting and anomaly scoring
- `nyc_taxi` for cross-domain robustness in the capstone comparison

Each notebook has been refreshed and exported to HTML for presentation use.

## Report Suite

- [notebooks/rnn_drift_forecast.ipynb](/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/notebooks/rnn_drift_forecast.ipynb)
- [notebooks/transformer_calibration.ipynb](/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/notebooks/transformer_calibration.ipynb)
- [notebooks/quantum_drift_combined.ipynb](/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/notebooks/quantum_drift_combined.ipynb)
- [website/notebooks_html/rnn_drift_forecast.html](/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/website/notebooks_html/rnn_drift_forecast.html)
- [website/notebooks_html/transformer_calibration.html](/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/website/notebooks_html/transformer_calibration.html)
- [website/notebooks_html/quantum_drift_combined.html](/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/website/notebooks_html/quantum_drift_combined.html)

## Best Results

### 1. Adaptive Recurrent Forecasting On Thermal Failure Data

The recurrent notebook tests whether gated recurrence materially improves over a plain Elman RNN on a real failure-bearing telemetry trace.

Observed results from the executed notebook:

- `GRU` achieved `MAE = 51.7912`, `RMSE = 55.3463`, `F1 = 0.2574`, `ROC-AUC = 0.7182`
- `LSTM` achieved `MAE = 51.4838`, `RMSE = 54.9950`, but `F1 = 0.0000`, `ROC-AUC = 0.4234`
- `VanillaRNN` trailed with `MAE = 61.3660`, `RMSE = 64.9298`, `F1 = 0.0000`, `ROC-AUC = 0.4083`
- Relative to `VanillaRNN`, the `GRU` reduced MAE by `15.60%` while also achieving the only non-zero incident F1 in the comparison

Figure map for the report:

- `Figure 1` shows the raw telemetry stream, anomaly windows, and engineered temporal features
- `Figure 2` shows training behavior and metric comparisons across VanillaRNN, LSTM, and GRU
- `Figure 3` shows uncertainty-aware forecast behavior and residual structure for the strongest recurrent model
- `Figure 4` shows the accuracy-efficiency frontier, residual separation, and calibration comparison used for final model selection

The most important visual argument is concentrated in `Figure 4`, with `Figure 2` and `Figure 3` supplying the supporting evidence for why adaptive gating is more useful than plain recurrence under incident supervision.

### 2. Transformer Calibration And Anomaly Concentration On Real Cloud Telemetry

The Transformer notebook asks a narrower question: whether long-context self-attention improves periodic alignment and anomaly concentration on real cloud telemetry.

Observed results from the executed notebook:

- `MAE = 0.043637`
- `RMSE = 0.133462`
- `ROC-AUC = 0.798664`
- `threshold = 0.100000`
- `parameter_count = 226765`

The test-set `F1` is `0.0000`, so this notebook should not be framed as a universal anomaly-detection win. Its technical value is instead the combination of low forecast error, strong ranking quality by ROC-AUC, and visualization-rich evidence that reconstruction scores concentrate differently in nominal and incident regimes.

Figure map for the report:

- `Figure 1` introduces the cloud telemetry trace, anomaly interval, and temporal covariates
- `Figure 2` shows optimization behavior for the Transformer forecaster and reconstruction model
- `Figure 3` shows forecast alignment, anomaly-score concentration, and representation structure
- `Figure 4` shows threshold response, regime separation, and hourly error diagnostics

The most presentation-ready evidence is `Figure 4`, with `Figure 3` providing the supporting anomaly-alignment and ranking-quality context. That is the correct framing for this notebook's strongest contribution.

### 3. Cross-Domain Benchmarking Across Three Real Datasets

The capstone notebook compares compact credible model families rather than weak strawman baselines.

Aggregate results from the executed notebook:

- `GRU` achieved the lowest mean forecast error with `mean_mae = 1337.3327` and the best mean ROC-AUC with `0.6603`
- `LSTM` achieved the best mean F1 with `0.1057`
- `Transformer` delivered the weakest aggregate cross-dataset performance in this compact comparison with `mean_mae = 1436.2461`, `mean_f1 = 0.0530`, `mean_auc = 0.1955`

Per-dataset winners:

- `EC2 CPU Utilization`: `LSTM` won MAE by a margin of `0.002698`
- `Machine Temperature System Failure`: `GRU` won MAE by `2.834884`
- `NYC Taxi Demand`: `GRU` won MAE by `293.907471`
- `LSTM` won F1 on all three datasets, including a margin of `0.034689` on `NYC Taxi Demand`

Figure map for the report:

- `Figure 1` shows the raw benchmark series across the three real datasets
- `Figure 2` shows per-dataset error and incident-sensitivity heatmaps
- `Figure 3` shows the aggregate summary by model family
- `Figure 4` shows mean ranks, cross-dataset frontiers, win counts, and leader margins

The decisive capstone evidence is `Figure 4`, with `Figure 2` and `Figure 3` supplying the per-dataset and aggregate context needed to defend an objective-aware model-selection claim.

## Contribution Claims

The repository supports three defensible contribution claims.

1. Adaptive recurrent gating improves materially over a plain RNN on real incident-bearing telemetry.
2. Long-context self-attention provides strong calibration and ranking behavior on real cloud metrics, even when a single threshold does not maximize test-set F1.
3. Cross-dataset model choice should be objective-aware: `GRU` is the most robust low-MAE model in the aggregate benchmark, while `LSTM` delivers the best mean incident F1.

## Best Visual Evidence

If the project needs to be presented quickly, these are the highest-value artifacts to show first.

1. The recurrent notebook's `Figure 4`, supported by `Figure 2` and `Figure 3`.
2. The Transformer notebook's `Figure 4`, supported by `Figure 3`.
3. The combined notebook's `Figure 4`, supported by `Figure 2` and `Figure 3`.

Those three `Figure 4` views communicate the full argument quickly, while the paired `Figure 2` and `Figure 3` panels in each report provide the technical evidence needed to survive deeper questioning.

## Repository Structure

```text
Quantum-Drift-Forecasting/
├── README.md
├── index.html
├── requirements.txt
├── data/
│   ├── quantum_device_metrics.csv
│   └── nab/
├── notebooks/
│   ├── quantum_drift_combined.ipynb
│   ├── rnn_drift_forecast.ipynb
│   └── transformer_calibration.ipynb
├── outputs/
│   ├── anomaly_scores.csv
│   ├── calibration_forecast.csv
│   └── drift_predictions.csv
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── models.py
│   ├── real_benchmark.py
│   ├── server.py
│   └── train.py
└── website/
    ├── demo.js
    ├── style.css
    └── notebooks_html/
```

## Environment And Reproduction

The notebooks were executed in the `qaoa` conda environment.

```bash
conda activate qaoa
pip install -r requirements.txt
jupyter lab
```

To regenerate the HTML reports:

```bash
conda activate qaoa
jupyter nbconvert --to notebook --execute --inplace notebooks/rnn_drift_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/transformer_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_drift_combined.ipynb
jupyter nbconvert --to html --output-dir website/notebooks_html notebooks/rnn_drift_forecast.ipynb notebooks/transformer_calibration.ipynb notebooks/quantum_drift_combined.ipynb
```

## Code Components

- `src/models.py` defines the recurrent, Transformer, and anomaly-detection architectures
- `src/real_benchmark.py` loads NAB datasets, generates features, and prepares temporal splits
- `src/evaluate.py` computes forecast and classification metrics and produces the main figures
- `src/train.py` provides a training entry point for the packaged models
- `src/server.py` exposes an inference endpoint for the demo

## Practical Reading Of The Results

This repository should not be presented as evidence that one architecture dominates every operational time-series task. The executed notebooks support a more precise conclusion:

- choose `GRU` when the requirement is stable low forecast error across heterogeneous datasets
- choose `LSTM` when event sensitivity is the main selection criterion in the cross-dataset benchmark
- choose the Transformer report when the audience wants to inspect calibration behavior, threshold response, and long-context anomaly ranking on real cloud telemetry

That narrower claim is more rigorous, and it is also the stronger interview presentation because it survives technical scrutiny.

## License

This project is released under the terms of the repository license.
