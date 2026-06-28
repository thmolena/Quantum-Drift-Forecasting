"""From-scratch foundations for causal spectral-truncation drift monitoring."""
from __future__ import annotations

FOUNDATION_SECTIONS: tuple[tuple[str, str], ...] = (
    (
        "Computation model",
        "A telemetry run is a finite table indexed by time and qubit.  The code "
        "turns it into windows X of shape (samples, time, channels), future T1 "
        "targets and drift labels.  Every reported number is a deterministic "
        "function of these arrays and integer seeds.",
    ),
    (
        "Two geometries of drift",
        "Marginal drift moves per-channel levels and is visible to ordinary "
        "monitors.  Correlated (crosstalk) drift couples channels with "
        "propagation delays while leaving marginals and the instantaneous "
        "covariance unchanged, so it lives only in ordered, lagged cross-channel "
        "correlations -- invisible to commutative monitors.",
    ),
    (
        "The causal spectral-truncation kernel",
        "A window is represented inside the channel C*-algebra M_C through the "
        "truncated causal shift S: K_n(X,Y) = sum_{tau<n} w_tau X^T S^tau Y.  The "
        "truncation order n controls noncommutativity; n=1 is the commutative "
        "instantaneous covariance, n>=2 adds asymmetric lagged cross-channel "
        "terms.  It is the causal counterpart of the periodic spectral-truncation "
        "kernel of Hashimoto et al. (NeurIPS 2024) and is positive definite for "
        "every n.",
    ),
    (
        "Leak-free time-series learning",
        "Windows pair recent history with future targets.  Chronological "
        "splitting prevents a training window from overlapping a test future; the "
        "unsupervised detector is fit on nominal windows only, so randomised "
        "splits cannot leak a label it never uses.",
    ),
    (
        "Ridge forecasting (the commutative member)",
        "The learned forecaster solves the closed-form ridge problem on the "
        "flattened window -- the n=1 kernel member.  Skill over persistence grows "
        "with horizon because persistence risk grows as autocorrelation decays.  "
        "Forecast skill is flat in n: noncommutativity is unnecessary for smooth "
        "marginal forecasting.",
    ),
    (
        "Noncommutative detection",
        "The reconstruction detector fits a low-rank nominal subspace in the "
        "kernel feature space and scores residual energy.  On correlated drift "
        "the commutative (n=1) detector is at chance and the noncommutative (n=2) "
        "detector reaches ROC-AUC ~0.90.  Detection is unimodal in n: too large "
        "an order dilutes the fixed-rank subspace with noise coordinates and "
        "collapses ROC-AUC toward chance.",
    ),
    (
        "Optimal-truncation certificate",
        "Increasing n captures more drift signal but adds C^2 noise coordinates "
        "per lag, so the separation-to-noise ratio peaks just past the "
        "informative lag.  This is the detection analogue of the "
        "representation-vs-complexity bound for spectral-truncation kernels, and "
        "it recovers the classical low-rank over-completeness collapse on the "
        "rank axis.",
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
