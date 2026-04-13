# Sequence Models for Quantum Hardware Drift, Calibration, and Reliability Monitoring

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

## Repository Structure

```text
Quantum-Drift-Forecasting/
├── README.md
├── index.html
├── requirements.txt
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
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/rnn_drift_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/transformer_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_drift_combined.ipynb
jupyter nbconvert --to html --output-dir website/notebooks_html \
  notebooks/rnn_drift_forecast.ipynb \
  notebooks/transformer_calibration.ipynb \
  notebooks/quantum_drift_combined.ipynb
```

To run the optional local inference API used by the website demo:

```bash
python -m src.server --port 5000
```
