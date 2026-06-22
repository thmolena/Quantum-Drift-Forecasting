"""From-principles guide for quantum-hardware drift forecasting.

Problem
-------
A quantum processor changes over time: relaxation time, dephasing time, gate
fidelities, readout error and calibration phases drift because the hardware and
environment are not static. The code represents that device history as a finite
multivariate time series. The scientific tasks are:

* forecast future coherence so recalibration can be scheduled earlier;
* detect anomalous drift windows without using labels at deployment time.

Computation model
-----------------
data.py creates deterministic telemetry arrays from a fixed random seed. A row
is a qubit-time observation and the feature vector contains physical quantities
such as T1, T2, one- and two-qubit fidelities and readout error. The generator
contains slow periodic structure, secular decay and Gaussian fluctuations.

Forecasting model
-----------------
make_paper_figures.py turns a recent window of length L into a design matrix and
predicts an H-step future target. The main learned forecaster is ridge
regression:

    W = (X.T X + alpha I)^(-1) X.T Y.

This is the unique minimizer of squared prediction error plus an L2 penalty.
Persistence and climatology are non-learning controls.

Why forecasting can beat persistence
------------------------------------
For a weakly stationary scalar signal with autocorrelation rho(h), persistence
risk is 2 gamma_0 (1 - rho(h)), while the best one-lag linear predictor has risk
gamma_0 (1 - rho(h)**2). As rho(h) decays with horizon, the room for skill over
persistence grows. The multivariate ridge model uses the whole seven-channel
history, which is why its horizon skill is larger than this one-lag baseline.

Anomaly detection
-----------------
The unsupervised detector learns a low-rank nominal subspace from non-drift
windows. A test window is scored by the squared reconstruction residual outside
that subspace. Singular value decomposition gives the optimal rank-k linear
subspace for nominal reconstruction, so the detector is a controlled geometric
baseline rather than an arbitrary heuristic. If k is too large, drift is also
reconstructed and detection collapses toward chance.

Sequence-model benchmark
------------------------
The manuscript compares RNN, LSTM, GRU and Transformer records from executed
notebooks. These are not retrained during figure generation; the code renders the
stored seeded metrics into figures and tables so the artifact is deterministic.

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
