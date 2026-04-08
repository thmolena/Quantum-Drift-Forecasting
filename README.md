# Time-Series and Transformer-Based Modeling for Quantum Hardware Calibration, Drift, and Noise Analysis

## Abstract

Quantum systems are highly sensitive to noise, time-dependent drift, and environmental variability, making reliable operation a fundamental challenge in quantum computing. We develop GPU-accelerated time-series pipelines and systematically benchmark recurrent neural networks (RNN, GRU, LSTM) and Transformer-based self-attention architectures for three core reliability objectives: forecasting drift, detecting anomalies, and quantifying uncertainty in operational hardware signals. Three self-contained experiments evaluate these architectures across real temporal regimes drawn from the Numenta Anomaly Benchmark — equipment thermal failure telemetry, cloud compute utilization, and high-volume demand data — under a CPU-feasible, fully reproducible protocol that spans the diversity of signal types encountered in quantum hardware characterization. The benchmark's central finding is that **no single architecture dominates all three objectives simultaneously**. GRU achieves the strongest aggregate forecast accuracy (mean MAE 1337.33, mean ROC-AUC 0.6603), LSTM maximizes mean incident-F1 (0.1057), and the Transformer delivers the most convincing calibration and anomaly-ranking result (ROC-AUC 0.7987 on periodic telemetry). These findings directly inform objective-aware architecture selection for quantum hardware monitoring and calibration pipelines. All results are produced by fully executed Jupyter notebooks with embedded figures and can be reproduced end-to-end on commodity hardware.

## 1. Introduction

Developing reliable quantum hardware requires continuous monitoring for drift, noise, and calibration degradation across multiple concurrent signal streams. Quantum devices are subject to decoherence, thermal noise, and time-dependent environmental perturbations that manifest as detectable patterns in sequential measurement data — including calibration histories, control signal traces, and readout error statistics. Effective monitoring requires architectures that can forecast drift, detect anomalies with calibrated sensitivity, and quantify uncertainty, all under computational constraints suitable for integration into hardware control and characterization pipelines.

This benchmark asks a deliberately scoped question: **among compact sequence architectures feasible for GPU-accelerated quantum hardware pipelines, which inductive bias best supports joint drift forecasting and incident localization, and does the answer depend on the evaluation objective and signal regime?** The three-experiment structure ensures each notebook delivers a distinct piece of evidence toward a unified conclusion: model selection for quantum hardware monitoring must be grounded in the target objective rather than in a blanket architectural prior.

**Contributions:**

1. We demonstrate that adaptive gated recurrence (GRU) reduces forecast MAE by 15.60% over a vanilla recurrent baseline while simultaneously sharpening residual concentration around documented failure events — directly relevant to early detection of thermal drift and calibration degradation in quantum hardware systems.
2. We show that long-context self-attention (Transformer) achieves ROC-AUC = 0.7987 on periodic operational telemetry and produces more convincing anomaly score separation than loss-minimizing threshold selection suggests, establishing it as the preferred architecture for score-ranked anomaly characterization in signals analogous to quantum calibration histories.
3. We provide a cross-domain evaluation across three heterogeneous temporal regimes demonstrating that no architecture dominates all three reliability objectives simultaneously — a critical finding for quantum hardware monitoring pipelines that must optimize for forecast fidelity, anomaly sensitivity, and calibration quality concurrently.

All figures are generated inline within the executed notebooks and can be regenerated deterministically.

## 2. Experimental Design and Data

Each experiment uses chronological train–validation–test splits that prevent temporal leakage. Input windows span 36 time steps; the forecast horizon spans 12 future steps. The training objective is a weighted combination of forecasting mean-squared error and binary cross-entropy for the auxiliary incident-scoring head, controlled by a weighting coefficient α = 0.75.

| Experiment | Dataset | Domain | Anomaly Type |
|---|---|---|---|
| 1 — Adaptive Gating | `machine_temperature_system_failure` | Equipment health | Thermal failure w/ precursor drift |
| 2 — Transformer Calibration | `ec2_cpu_utilization_24ae8d` | Cloud infrastructure | Utilization spikes on periodic background |
| 3 — Cross-Domain Benchmark | All three NAB datasets | Multi-domain | Heterogeneous regimes |

Feature engineering produces rolling statistics (mean, standard deviation, first difference) on each raw series. These covariates are visualized in the per-experiment dataset sections to make the forecasting context explicit before any architecture comparison is introduced.

## 3. Figures and Visual Evidence

Each notebook is structured around a shared nine-part presentation flow (objective, dataset, protocol, model construction, training, results, interpretation, limitations, takeaways) and produces four to six figures. The figures below constitute the primary visual evidence of the benchmark.

### 3.1 Figure Suite — Experiment 1: Adaptive Gated Recurrent Forecasting

**Figure 1.1 — Raw Telemetry and Engineered Feature View.**
A two-panel time-series figure. The upper panel plots the raw machine-temperature signal across time with the documented failure interval shaded directly on the timeline. The lower panel overlays rolling mean, rolling standard deviation, and first-difference features, making local trend and volatility changes visible as the incident window approaches. This figure establishes the forecasting target geometry before any model comparison.

**Figure 1.2 — Optimization Behavior and Headline Metrics.**
A four-subplot training summary consisting of per-epoch training loss, per-epoch validation loss, a comparative MAE bar chart, and a comparative incident-F1 bar chart across VanillaRNN, LSTM, and GRU. This figure demonstrates that gated models can be trained within a bounded CPU budget without unstable optimization.

**Figure 1.3 — Uncertainty-Aware Forecast and Incident-Aligned Residual Profile.**
A two-panel figure comprising an uncertainty-envelope forecast (MC-Dropout with conformal bound) for the best recurrent model alongside a residual-intensity view aligned with binary incident labels. The intended reading is whether model difficulty concentrates around the documented failure regime rather than remaining uniformly distributed across the test split.

**Figure 1.4 — Accuracy–Efficiency Frontier and Regime Separation.**
A four-subplot diagnostic panel including: (i) a relative MAE-reduction bar chart versus the vanilla baseline, (ii) a parameter-count versus incident-F1 scatter plot colored by MAE to expose the accuracy–efficiency trade-off, (iii) a boxplot comparing residual distributions between nominal and incident windows, and (iv) a first-step forecast calibration scatter plot. Together, these panels provide the strongest evidence for architectural selection in the recurrent family.

<p align="center">
  <img src="outputs/qubit_trajectories.png" alt="Simulated qubit coherence drift trajectories illustrating the forecasting target structure" width="92%" />
</p>

*Supplementary — Synthetic Drift Trajectories.* This figure summarizes the synthetic coherence-time behavior used by the interactive browser demo, providing an accessible entry point to the drift-detection framing before the real-data experiments.

<p align="center">
  <img src="outputs/correlation_matrix.png" alt="Feature correlation matrix for the drift forecasting pipeline, showing pairwise relationships among engineered inputs" width="72%" />
</p>

*Supplementary — Feature Correlation Structure.* The correlation matrix provides a compact view of pairwise relationships among the engineered inputs supplied to the sequence models throughout all three experiments.

### 3.2 Figure Suite — Experiment 2: Transformer Calibration on Cloud Telemetry

**Figure 2.1 — EC2 Signal Structure and Anomaly Context.**
A multi-panel view of the raw EC2 CPU utilization signal with periodic structure annotated and anomaly intervals shaded. Temporal covariates (rolling statistics, differenced series) are overlaid to motivate the use of long-context self-attention on a periodically structured operational workload.

**Figure 2.2 — Transformer Training Dynamics.**
Per-epoch training and validation loss curves for the Transformer forecaster, demonstrating stable convergence under CPU-feasible training settings.

**Figure 2.3 — Uncertainty-Aware Forecast and Reconstruction Anomaly Scores.**
A paired figure showing the Transformer's multi-step forecast with uncertainty bands alongside reconstruction-error-based anomaly scores aligned with the documented anomaly intervals. This figure is the primary evidence that long-context representation improves incident concentration.

**Figure 2.4 — Calibration and Threshold Sensitivity Analysis.**
A multi-panel calibration analysis including: (i) a forecast-versus-observed scatter plot to evaluate prediction alignment, (ii) a precision–recall curve and ROC curve to summarize detection quality across operating thresholds, and (iii) a threshold-sensitivity view showing how F1 and anomaly-count co-vary across the score range. ROC-AUC = 0.7987 is the headline result; the calibration and threshold panels motivate why ranking quality is a more informative summary than the single-threshold F1.

### 3.3 Figure Suite — Experiment 3: Cross-Domain Benchmark Comparison

**Figure 3.1 — Multi-Dataset Signal Comparison.**
A three-panel overview of the raw signals across machine temperature, EC2 CPU utilization, and NYC taxi demand, with documented anomaly intervals shaded in each panel. This figure makes regime differences visible before the cross-model performance comparison is introduced.

**Figure 3.2 — Cross-Dataset Performance Heatmap.**
A metric-by-model-by-dataset heatmap that compresses all cross-domain results into a single view. This is the central diagnostic figure of the benchmark: it shows which models rank first on MAE, F1, and ROC-AUC within each dataset and whether those rankings are consistent across domains.

**Figure 3.3 — Aggregate Performance Distributions.**
Box or violin plots of per-dataset metric distributions across model families, showing whether any architecture achieves consistently low spread across all three regimes.

**Figure 3.4 — Metric-Dependent Model Preference.**
A structured comparison of mean MAE, mean F1, and mean ROC-AUC across models, together with a visual encoding of within-dataset rank stability. This figure delivers the benchmark's main conclusion: GRU leads on mean MAE (1337.3327) and mean ROC-AUC (0.6603), LSTM leads on mean F1 (0.1057), and no architecture dominates all three objectives simultaneously.

## 4. Results Summary

| Experiment | Best Model | MAE | RMSE | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Adaptive Gating (thermal) | GRU | 51.7912 | 55.3463 | 0.2574 | 0.7182 |
| Transformer Calibration (cloud) | Transformer | 0.0436 | 0.1335 | — | 0.7987 |
| Cross-Domain Mean-MAE | GRU | 1337.33 (mean) | — | — | 0.6603 (mean) |
| Cross-Domain Mean-F1 | LSTM | — | — | 0.1057 (mean) | — |

**Recurrent experiment.** GRU reduces MAE by 15.60% over VanillaRNN and produces visibly stronger residual separation between nominal and incident windows. These gains hold under both a point-estimate MAE comparison and an uncertainty-aware conformal envelope assessment.

**Transformer experiment.** The Transformer achieves the strongest anomaly ranking (ROC-AUC = 0.7987) and tightest forecast calibration on the periodically structured EC2 signal. Its chosen operating threshold (0.1) does not maximize test-set F1, but the score distribution and precision–recall analysis confirm that the model's latent representation is genuinely informative.

**Cross-domain experiment.** Neither GRU nor LSTM nor Transformer achieves the best score on all three metrics simultaneously across all three datasets. The correct conclusion is that model selection should be made with reference to the deployment objective and dataset characteristics, not to an architecture-level prior.

## 5. Executed Report Suite

Each experiment is delivered as a fully executed Jupyter notebook with embedded figures and inline metric tables. HTML exports of all three notebooks are available for non-interactive review.

| Report | Technical Question | Primary Figures | Links |
|---|---|---|---|
| `rnn_drift_forecast` | Does adaptive gated recurrence improve joint forecasting and incident localization on real failure telemetry? | Figs. 1.1–1.4: telemetry view, training dynamics, uncertainty forecast, accuracy–efficiency frontier | [Notebook](notebooks/rnn_drift_forecast.ipynb) · [HTML](website/notebooks_html/rnn_drift_forecast.html) |
| `transformer_calibration` | Does long-context self-attention improve calibration and anomaly ranking on periodic cloud telemetry? | Figs. 2.1–2.4: signal structure, training dynamics, reconstruction anomaly scores, calibration and threshold analysis | [Notebook](notebooks/transformer_calibration.ipynb) · [HTML](website/notebooks_html/transformer_calibration.html) |
| `quantum_drift_combined` | Which compact architecture remains near the performance frontier across heterogeneous real-world regimes? | Figs. 3.1–3.4: multi-dataset signal view, cross-domain heatmap, aggregate distributions, metric-dependent model preference | [Notebook](notebooks/quantum_drift_combined.ipynb) · [HTML](website/notebooks_html/quantum_drift_combined.html) |

## 6. Repository Structure

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
│   ├── rnn_drift_forecast.ipynb          ← Experiment 1
│   ├── transformer_calibration.ipynb     ← Experiment 2
│   └── quantum_drift_combined.ipynb      ← Experiment 3
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

## 7. Reproducibility

All experiments run end-to-end on commodity CPU hardware. Stochastic operations are seeded at the entry point of each notebook.

```bash
conda activate qaoa
pip install -r requirements.txt
jupyter lab
```

To re-execute and re-export all three notebooks:

```bash
conda activate qaoa
jupyter nbconvert --to notebook --execute --inplace notebooks/rnn_drift_forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/transformer_calibration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_drift_combined.ipynb
jupyter nbconvert --to html --output-dir website/notebooks_html \
  notebooks/rnn_drift_forecast.ipynb \
  notebooks/transformer_calibration.ipynb \
  notebooks/quantum_drift_combined.ipynb
```

To start the optional local inference API used by the interactive demo:

```bash
conda activate qaoa
python -m src.server --port 5000
```

## 8. Code Components

- `src/models.py` — VanillaRNN, LSTMForecaster, GRUForecaster, TransformerForecaster, and reconstruction-based anomaly detection architectures.
- `src/real_benchmark.py` — NAB dataset loading, feature engineering, and time-respecting chronological splits.
- `src/evaluate.py` — Forecast and classification metric computation; figure generation utilities shared across all three notebooks.
- `src/train.py` — Packaged training entry point with reproducible seeding.
- `src/server.py` — Local Flask inference endpoint consumed by the browser demo.

## 9. Discussion and Limitations

**Why not a single dataset?** Single-dataset results risk selecting for the idiosyncrasies of one temporal regime. Quantum hardware monitoring pipelines must handle diverse signal types — thermal drift, calibration variability, and high-volume demand — so the cross-domain experiment directly tests whether architectural advantages generalize across the signal diversity of real hardware characterization settings.

**Why CPU-scale models?** Benchmark credibility depends on reproducibility. Training budgets constrained to consumer CPU hardware ensure that every result in this repository can be challenged and re-evaluated without specialized infrastructure. The methods are designed to scale to GPU-accelerated pipelines for production deployment.

**Known limitations.** Training budgets are intentionally small (6 epochs per model), which bounds the ceiling on reported metrics. The anomaly labels from NAB are used at face value without label-noise analysis. The Transformer report's headline F1 is suboptimal relative to its ROC-AUC, reflecting threshold selection sensitivity rather than a fundamental representational failure. These limitations are documented in the interpretation sections of each notebook.