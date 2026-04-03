from pathlib import Path
import re

BASE = Path("/Users/mohuyn/Library/CloudStorage/OneDrive-SAS/Documents/GitHub/Quantum-Drift-Forecasting/website/notebooks_html")
FILES = {
    "rnn_drift_forecast.html": [
        "Figure 1: raw machine-temperature telemetry with anomaly windows and engineered temporal features",
        "Figure 2a: training and validation loss trajectories for the recurrent forecasting models",
        "Figure 2b: grouped error and classification metrics comparing Vanilla RNN, LSTM, and GRU",
        "Figure 2c: rank-by-metric heatmap summarizing recurrent model ordering across forecasting and anomaly criteria",
        "Figure 3a: GRU forecast against the held-out machine-temperature trajectory with uncertainty-aware reconstruction",
        "Figure 3b: standardized residual profile highlighting separation between nominal and anomalous intervals",
        "Figure 4: accuracy-efficiency frontier, residual-separation summary, and calibration comparison for the recurrent models",
    ],
    "transformer_calibration.html": [
        "Figure 1: EC2 CPU telemetry with anomaly interval markers and engineered temporal covariates",
        "Figure 2: Transformer forecaster and autoencoder training curves over optimization epochs",
        "Figure 3a: Transformer forecast versus observed cloud telemetry over the evaluation horizon",
        "Figure 3b: anomaly score trajectory aligned with the labeled anomaly interval",
        "Figure 3c: representation affinity structure showing clustering behavior in the learned signal space",
        "Figure 4: threshold response, regime separation, and hourly error diagnostics for the Transformer pipeline",
    ],
    "quantum_drift_combined.html": [
        "Figure 1: cross dataset raw signal overview across the three benchmark series",
        "Figure 2: per dataset error and incident sensitivity heatmaps",
        "Figure 3: aggregate benchmark summary by model family",
        "Figure 4: mean ranks, frontiers, win counts, and leader margins across datasets",
    ],
}

PLACEHOLDER_PATTERN = re.compile(
    r'alt="No description has been provided for this image"',
    re.IGNORECASE,
)

for name, alts in FILES.items():
    path = BASE / name
    text = path.read_text(encoding="utf-8")
    state = {"count": 0}

    def replace(match: re.Match[str]) -> str:
        state["count"] += 1
        count = state["count"]
        alt = alts[count - 1] if count <= len(alts) else f"{Path(name).stem} image {count}"
        return f'alt="{alt}"'

    updated = PLACEHOLDER_PATTERN.sub(replace, text)
    path.write_text(updated, encoding="utf-8")
    print(f"{name}: added alt text to {state['count']} image tag(s)")
