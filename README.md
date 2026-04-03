# Quantum-Drift-Forecasting

## Abstract

Quantum-Drift-Forecasting is a reproducible benchmark for adaptive sequence models on real operational time series. The repository evaluates recurrent gating and long-context attention on public Numenta Anomaly Benchmark datasets, then packages the results as executed notebook reports suitable for technical review, presentation, and re-execution.

The central conclusion is deliberately narrow. The experiments do not support a claim that one architecture dominates every forecasting and anomaly-detection objective. Instead, they show that model selection should be objective-aware: `GRU` is the strongest low-error aggregate choice in this benchmark, `LSTM` is the strongest mean-F1 choice, and the Transformer report is most convincing when interpreted through calibration and ranking quality rather than a single thresholded F1 score.

## What The Repository Shows

1. Adaptive recurrent gating materially improves over a plain `VanillaRNN` on incident-bearing thermal telemetry.
2. The Transformer configuration achieves strong cloud-telemetry ranking quality with `ROC-AUC = 0.7987`, even though its chosen operating threshold does not maximize test-set F1.
3. Cross-dataset benchmarking favors different models depending on the metric being optimized, so claims of a universal winner are not technically defensible.

## Visuals

The repository includes two exported visual artifacts in `outputs/` and three full HTML notebook reports in `website/notebooks_html/`.

### Simulated Drift Trajectories

<p align="center">
  <img src="outputs/qubit_trajectories.png" alt="Simulated qubit coherence trajectories used by the drift forecasting demo" width="92%" />
</p>

This figure summarizes the synthetic drift behavior used by the interactive demo and server-facing forecasting pipeline.

### Feature Correlation Structure

<p align="center">
  <img src="outputs/correlation_matrix.png" alt="Feature correlation matrix for the drift forecasting data pipeline" width="72%" />
</p>

This plot provides a compact view of the engineered feature relationships used in the forecasting workflow.

## Executed Report Suite

| Report | Technical Question | Strongest Result | Artifacts |
| --- | --- | --- | --- |
| `rnn_drift_forecast` | Does adaptive recurrent memory improve forecasting and incident discrimination on thermal failure telemetry? | `GRU` achieved `MAE = 51.7912`, `RMSE = 55.3463`, `F1 = 0.2574`, and `ROC-AUC = 0.7182`, with a `15.60%` MAE reduction versus `VanillaRNN`. | [Notebook](notebooks/rnn_drift_forecast.ipynb) · [HTML](website/notebooks_html/rnn_drift_forecast.html) |
| `transformer_calibration` | Does long-context self-attention improve calibration and anomaly ranking on cloud telemetry? | The Transformer achieved `MAE = 0.043637`, `RMSE = 0.133462`, and `ROC-AUC = 0.798664`; the report is strongest as a calibration and ranking analysis. | [Notebook](notebooks/transformer_calibration.ipynb) · [HTML](website/notebooks_html/transformer_calibration.html) |
| `quantum_drift_combined` | Which compact architecture is most robust across three real datasets? | `GRU` achieved the best mean MAE (`1337.3327`) and mean ROC-AUC (`0.6603`), while `LSTM` achieved the best mean F1 (`0.1057`). | [Notebook](notebooks/quantum_drift_combined.ipynb) · [HTML](website/notebooks_html/quantum_drift_combined.html) |

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
```

## Environment And Reproduction

Use the `qaoa` conda environment for every notebook and script run in this repository.

```bash
conda activate qaoa
pip install -r requirements.txt
jupyter lab
```

To regenerate the executed notebook reports and their HTML exports:

```bash
conda activate qaoa
jupyter nbconvert --to notebook --execute --inplace notebooks/rnn_drift_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/transformer_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_drift_combined.ipynb
jupyter nbconvert --to html --output-dir website/notebooks_html notebooks/rnn_drift_forecast.ipynb notebooks/transformer_calibration.ipynb notebooks/quantum_drift_combined.ipynb
```

To run the optional demo-backed API locally:

```bash
conda activate qaoa
python -m src.server --port 5000
```

## Code Components

- `src/models.py` defines the recurrent, Transformer, and anomaly-detection architectures.
- `src/real_benchmark.py` loads NAB datasets, engineers features, and constructs time-respecting splits.
- `src/evaluate.py` computes forecast and classification metrics and produces the exported figures.
- `src/train.py` provides the packaged training entry point.
- `src/server.py` exposes the local inference endpoint used by the website demo.

## Interpretation

The most defensible way to present this project is as a benchmark about tradeoffs rather than architectural supremacy. Use the recurrent report when discussing incident-aware forecasting gains, the Transformer report when discussing calibration and anomaly ranking, and the combined report when the audience wants a cross-dataset model-selection argument.