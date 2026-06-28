"""Quantum drift forecasting reproduction package.

Public API
----------
The central contribution is the *causal spectral-truncation C*-algebraic kernel*
(:mod:`qdriftforecast.kernels`), whose truncation order ``n`` controls the
noncommutativity of the products appearing in the kernel.  It powers an
unsupervised drift detector (:mod:`qdriftforecast.detection`) and a coherence
forecaster (:mod:`qdriftforecast.forecasting`) over a seeded telemetry generator
and a controlled correlated-drift benchmark (:mod:`qdriftforecast.data`).
"""

__version__ = "2.0.0"

from .data import (  # noqa: F401
    FEATURE_COLS,
    generate_synthetic_dataset,
    generate_correlated_drift_windows,
)
from .kernels import (  # noqa: F401
    cst_feature_map,
    cst_operator_kernel,
    cst_scalar_gram,
    lower_shift,
    periodic_truncation_feature_map,
)
from .detection import (  # noqa: F401
    baseline_comparison,
    rank_truncation_grid,
    truncation_sweep,
)
from .forecasting import (  # noqa: F401
    cstk_forecast_skill,
    forecast_baselines,
)

__all__ = [
    "__version__",
    "FEATURE_COLS",
    "generate_synthetic_dataset",
    "generate_correlated_drift_windows",
    "cst_feature_map",
    "cst_operator_kernel",
    "cst_scalar_gram",
    "lower_shift",
    "periodic_truncation_feature_map",
    "baseline_comparison",
    "rank_truncation_grid",
    "truncation_sweep",
    "cstk_forecast_skill",
    "forecast_baselines",
]
