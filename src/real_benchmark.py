from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .data import make_sequences, normalize, temporal_split


BASE_DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data"
BASE_LABEL_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_windows.json"

DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "machine_temperature_system_failure": {
        "path": "realKnownCause/machine_temperature_system_failure.csv",
        "display_name": "Machine Temperature System Failure",
        "application": "data-center thermal monitoring",
    },
    "ec2_cpu_utilization_24ae8d": {
        "path": "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
        "display_name": "EC2 CPU Utilization",
        "application": "cloud capacity monitoring",
    },
    "nyc_taxi": {
        "path": "realKnownCause/nyc_taxi.csv",
        "display_name": "NYC Taxi Demand",
        "application": "urban mobility demand planning",
    },
}

FEATURE_COLUMNS: List[str] = [
    "value",
    "value_diff",
    "rolling_mean_12",
    "rolling_std_12",
    "rolling_mean_36",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        urllib.request.urlretrieve(url, destination)
    return destination


def ensure_dataset(dataset_name: str, root_dir: str = "data/nab") -> Path:
    spec = DATASET_SPECS[dataset_name]
    destination = Path(root_dir) / spec["path"]
    return _download(f"{BASE_DATA_URL}/{spec['path']}", destination)


def ensure_labels(root_dir: str = "data/nab") -> Path:
    destination = Path(root_dir) / "labels" / "combined_windows.json"
    return _download(BASE_LABEL_URL, destination)


def load_dataset(dataset_name: str, root_dir: str = "data/nab") -> pd.DataFrame:
    csv_path = ensure_dataset(dataset_name, root_dir=root_dir)
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_labels(root_dir: str = "data/nab") -> Dict[str, list]:
    label_path = ensure_labels(root_dir=root_dir)
    with label_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_feature_frame(dataset_name: str, root_dir: str = "data/nab") -> pd.DataFrame:
    df = load_dataset(dataset_name, root_dir=root_dir).copy()
    labels = load_labels(root_dir=root_dir)
    label_key = DATASET_SPECS[dataset_name]["path"]

    df["label"] = 0
    for start_str, end_str in labels.get(label_key, []):
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)
        df.loc[df["timestamp"].between(start, end), "label"] = 1

    df["value_diff"] = df["value"].diff().fillna(0.0)
    df["rolling_mean_12"] = df["value"].rolling(12, min_periods=1).mean()
    df["rolling_std_12"] = df["value"].rolling(12, min_periods=1).std().fillna(0.0)
    df["rolling_mean_36"] = df["value"].rolling(36, min_periods=1).mean()

    fractional_hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    weekday = df["timestamp"].dt.dayofweek.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    return df


def prepare_sequence_dataset(
    dataset_name: str,
    seq_len: int,
    horizon: int,
    root_dir: str = "data/nab",
) -> dict:
    frame = build_feature_frame(dataset_name, root_dir=root_dir)
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.float32)
    target_col_idx = FEATURE_COLUMNS.index("value")
    X_seq, y_seq, lbl = make_sequences(
        X,
        y,
        seq_len=seq_len,
        horizon=horizon,
        target_col_idx=target_col_idx,
    )
    timestamps = frame["timestamp"].iloc[seq_len + horizon - 1 :].reset_index(drop=True)

    parts = temporal_split(X_seq, y_seq, lbl)
    Xtr, ytr, ltr, Xv, yv, lv, Xte, yte, lte = parts

    n_feat = Xtr.shape[-1]
    Xtr_n, Xv_n, Xte_n, x_min, x_max = normalize(
        Xtr.reshape(-1, n_feat),
        Xv.reshape(-1, n_feat),
        Xte.reshape(-1, n_feat),
    )

    n_total = len(X_seq)
    n_train = len(Xtr)
    n_val = len(Xv)

    return {
        "frame": frame,
        "features": FEATURE_COLUMNS,
        "train": (Xtr_n.reshape(Xtr.shape), ytr, ltr),
        "val": (Xv_n.reshape(Xv.shape), yv, lv),
        "test": (Xte_n.reshape(Xte.shape), yte, lte),
        "x_min": x_min,
        "x_max": x_max,
        "timestamps": {
            "all": timestamps,
            "train": timestamps.iloc[:n_train].reset_index(drop=True),
            "val": timestamps.iloc[n_train : n_train + n_val].reset_index(drop=True),
            "test": timestamps.iloc[n_train + n_val : n_total].reset_index(drop=True),
        },
        "dataset_name": dataset_name,
        "display_name": DATASET_SPECS[dataset_name]["display_name"],
        "application": DATASET_SPECS[dataset_name]["application"],
        "label_path": DATASET_SPECS[dataset_name]["path"],
    }
