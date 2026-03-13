"""
server.py — Flask inference API for Quantum Drift Forecasting.

Endpoints
---------
POST /forecast
    Body: { "sequence": [[f1, f2, ...], ...],  "model": "lstm" }
    Returns: { "forecast": [...], "drift_prob": float, "anomaly_score": float }

GET /health
    Returns: { "status": "ok", "models": [...] }

Run
---
    python -m src.server
    python -m src.server --port 5001 --model-path model_lstm.pt
"""

import argparse
import os
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.models import build_model, AnomalyDetector
from src.data import FEATURE_COLS

app = Flask(__name__)
CORS(app)

# Loaded model cache  {model_name → nn.Module}
_model_cache: dict = {}
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model loader ────────────────────────────────────────────────────────────
def _load_model(model_path: str) -> tuple:
    """Load a checkpoint and return (model, config)."""
    ckpt   = torch.load(model_path, map_location=_device)
    config = ckpt["config"]
    model  = build_model(
        config["model_name"],
        input_dim=config["input_dim"],
        horizon=config["horizon"],
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
    ).to(_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


def _get_model(model_name: str) -> tuple:
    """Return a cached model or attempt to load from common paths."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    candidate_paths = [
        f"model_{model_name}.pt",
        f"model.pt",
        f"outputs/model_{model_name}.pt",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            model, config = _load_model(path)
            _model_cache[model_name] = (model, config)
            return model, config

    raise FileNotFoundError(
        f"No checkpoint found for model '{model_name}'. "
        f"Run: python -m src.train --model {model_name}"
    )


# ── Anomaly detector (untrained, uses random weights for demo) ─────────────
_anomaly_detector = None

def _get_anomaly_detector(input_dim: int) -> AnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(input_dim=input_dim).to(_device)
        _anomaly_detector.eval()
    return _anomaly_detector


# ── /health ─────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    available = [
        name for name in ["rnn", "lstm", "gru", "transformer"]
        if any(
            os.path.exists(p) for p in [f"model_{name}.pt", "model.pt"]
        )
    ]
    return jsonify({"status": "ok", "available_models": available})


# ── /forecast ────────────────────────────────────────────────────────────────
@app.route("/forecast", methods=["POST"])
def forecast():
    payload = request.get_json(force=True)

    # Validate input
    if "sequence" not in payload:
        return jsonify({"error": "Missing 'sequence' field"}), 400

    sequence   = payload["sequence"]    # list of lists
    model_name = payload.get("model", "lstm")

    try:
        seq_arr = np.array(sequence, dtype=np.float32)  # (T, F)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid sequence format: {exc}"}), 400

    if seq_arr.ndim != 2:
        return jsonify({"error": "Sequence must be a 2-D list [[f1, f2, ...], ...]"}), 400

    x = torch.tensor(seq_arr, dtype=torch.float32, device=_device).unsqueeze(0)  # (1, T, F)

    # Forecasting model
    try:
        model, config = _get_model(model_name)
    except FileNotFoundError as exc:
        # Fallback: return browser-side hint
        return jsonify({
            "error": str(exc),
            "hint": "Use the browser-side simulation or train a model first.",
        }), 404

    with torch.no_grad():
        forecast_tensor, drift_logit = model(x)

    forecast_vals = forecast_tensor.squeeze(0).cpu().numpy().tolist()
    drift_prob    = float(torch.sigmoid(drift_logit).item())

    # Anomaly scoring
    anomaly_det   = _get_anomaly_detector(seq_arr.shape[-1])
    anomaly_score = float(anomaly_det.anomaly_scores(x).item())

    return jsonify({
        "forecast":      forecast_vals,
        "drift_prob":    round(drift_prob, 4),
        "anomaly_score": round(anomaly_score, 6),
        "model":         model_name,
        "horizon":       config["horizon"],
    })


# ── CLI ──────────────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser(description="Quantum Drift Forecasting API server")
    p.add_argument("--port",       type=int, default=5000)
    p.add_argument("--host",       default="127.0.0.1")
    p.add_argument("--model-path", default=None,
                   help="Pre-load a specific model checkpoint on startup")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    if args.model_path and os.path.exists(args.model_path):
        model, config = _load_model(args.model_path)
        _model_cache[config["model_name"]] = (model, config)
        print(f"[server] Loaded {config['model_name']} from {args.model_path}")
    print(f"[server] Listening on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
