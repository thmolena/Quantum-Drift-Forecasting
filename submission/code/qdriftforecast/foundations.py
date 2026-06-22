"""From-scratch foundations for objective-aware quantum drift forecasting."""

from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Computation model",
        "A telemetry run is a finite table indexed by time and qubit.  The code "
        "turns that table into arrays: windows X with shape (samples, time, "
        "features), future T1 targets y and horizon labels.  Every reported "
        "number is a deterministic function of these arrays and integer seeds.",
    ),
    (
        "Hardware drift",
        "Relaxation time, dephasing time, readout error and gate fidelity vary "
        "over time because a processor is not a stationary object.  The seeded "
        "generator in data.py combines a slow periodic component, a secular "
        "trend and bounded noise so the benchmark has predictable structure "
        "without using hidden external data.",
    ),
    (
        "Leak-free time-series learning",
        "make_sequences() forms recent-history windows and future targets.  "
        "Chronological splitting prevents a training window from overlapping a "
        "test future.  Scaling statistics are fit on training data only, then "
        "applied forward to validation and test data.",
    ),
    (
        "Ridge forecasting",
        "The CPU forecasters solve the closed-form ridge problem "
        "argmin_W ||XW-Y||^2 + alpha ||W||^2.  Persistence and climatology are "
        "non-learning baselines.  Forecast skill is reported as one minus the "
        "MAE ratio to persistence at the same horizon.",
    ),
    (
        "Why horizon skill can grow",
        "For a mean-reverting stationary process, persistence risk grows as "
        "autocorrelation decays.  A learned linear predictor can use the recent "
        "trajectory and correlated side channels, so its relative skill can "
        "increase at longer horizons where persistence becomes stale.",
    ),
    (
        "Unsupervised detection",
        "The reconstruction detector fits a low-rank subspace to nominal "
        "windows and scores residual energy off that subspace.  The SVD gives "
        "the optimal rank-k least-squares subspace; when k is too large, the "
        "detector reconstructs anomalous windows too well and ROC-AUC collapses "
        "toward chance.",
    ),
    (
        "Sequence-model comparison",
        "The RNN, LSTM, GRU and Transformer benchmark values in Fig. 3 are "
        "seeded notebook metrics rendered by make_paper_figures.py.  The paper "
        "uses them to compare objectives: general incident detection, cross-"
        "domain ranking and periodic-regime specialization.",
    ),
    (
        "Reproduction path",
        "Run python make_paper_figures.py or the qdrift-reproduce entry point "
        "from the code folder.  The script writes vector PDF figures, LaTeX "
        "table bodies and JSON source data from fixed seeds.  Dependencies are "
        "declared in pyproject.toml.",
    ),
)


def iter_foundations() -> tuple[tuple[str, str], ...]:
    return FOUNDATION_SECTIONS


def print_foundations() -> None:
    for index, (heading, body) in enumerate(FOUNDATION_SECTIONS, start=1):
        print(f"{index}. {heading}\n{body}\n")


if __name__ == "__main__":
    print_foundations()
