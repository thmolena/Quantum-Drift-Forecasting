"""From-principles guide for causal spectral-truncation drift monitoring.

Problem
-------
A quantum processor changes over time: relaxation time, dephasing time, gate
fidelities, readout error and calibration phases drift because the hardware and
environment are not static. The code represents that device history as a finite
multivariate time series (a window ``X`` of shape ``(L, C)``: ``L`` time steps,
``C`` channels). The scientific tasks are:

* forecast future coherence so recalibration can be scheduled earlier;
* detect anomalous drift windows without using labels at deployment time.

Two geometries of drift
-----------------------
Marginal drift moves per-channel levels (e.g. T1 crossing a threshold); it is
visible to ordinary level- and covariance-based monitors. Correlated
(crosstalk) drift instead couples several channels with propagation delays while
leaving every marginal and the instantaneous covariance unchanged: it lives
entirely in the ordered, *lagged* cross-channel correlations. The central point
of this work is that the second geometry is invisible to commutative monitors and
visible only to a noncommutative one.

The causal spectral-truncation kernel
--------------------------------------
``kernels.py`` represents a window inside the channel C*-algebra ``M_C`` through
the truncated causal shift ``S`` ((S v)_t = v_{t-1}):

    K_n(X, Y) = sum_{tau=0}^{n-1} w_tau X^T S^tau Y   in   M_C,

whose (c, c') entry is the order-``n`` causal lagged cross-channel correlation.
The truncation order ``n`` controls noncommutativity: ``n = 1`` is the
commutative instantaneous covariance ``X^T Y`` (symmetric), while for ``n >= 2``
the lagged terms are asymmetric in (c, c') because ``S^T != S``. Truncating the
shift algebra to ``n`` powers is the causal counterpart of truncating the
Fourier-multiplier algebra to ``n`` modes in the spectral-truncation kernels of
Hashimoto et al. (NeurIPS 2024); the difference is that ``S`` respects the arrow
of time. The scalar kernel ``k_n(X, Y) = <Phi_n(X), Phi_n(Y)>`` is positive
definite for every ``n`` (explicit feature map).

Forecasting
-----------
``forecasting.py`` predicts an H-step T1 horizon from a window. The learned
baseline is ridge regression on the flattened window -- the commutative (``n =
1``) member of the kernel family. Persistence and climatology are non-learning
controls. On smooth, mean-reverting marginal telemetry the commutative member is
already near-optimal, so forecast skill is flat in ``n``: noncommutativity is
unnecessary here. Skill over persistence grows with horizon because, for a weakly
stationary signal with autocorrelation rho(h), persistence risk is
2 gamma_0 (1 - rho(h)) while the best one-lag predictor's risk is
gamma_0 (1 - rho(h)^2); the gap widens as rho decays.

Detection
---------
``detection.py`` fits a low-rank nominal subspace in the kernel feature space and
scores a test window by its residual energy off that subspace. With ``n = 1`` it
is the classical reconstruction detector and is at chance on correlated drift
(matched marginals and instantaneous covariance). With ``n = 2`` it detects at
ROC-AUC ~0.90. Increasing ``n`` further adds noise coordinates that dilute the
fixed-rank subspace, so detection is unimodal in ``n`` with an interior optimum:
a representation-vs-complexity trade-off, the detection analogue of the
generalization bound that governs spectral-truncation kernels in regression.

Artifacts
---------
Run from submission/code:

    export PYTHONPATH=.
    python -m qdriftforecast.reproduce

The command writes manuscript figures to ../figures, table bodies to ../tables
and source JSON to generated_data. No manuscript number is hand-entered.
"""


GUIDE = __doc__


def main() -> None:
    print(GUIDE)


if __name__ == "__main__":
    main()
