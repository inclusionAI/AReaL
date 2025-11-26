import datetime

# For large models, generation may consume more than 7200s.
# We set a large value to avoid timeout issues during generation.
DIST_GROUP_DEFAULT_TIMEOUT = datetime.timedelta(seconds=7200)

# Proximal log-probability computation methods for decoupled PPO
# These control how the proximal policy log-probabilities are computed
PROX_LOGP_METHOD_RECOMPUTE = (
    "recompute"  # Standard decoupled PPO: recompute via forward pass
)
PROX_LOGP_METHOD_LOGLINEAR = (
    "loglinear"  # Use log-linear approximation (skip forward pass)
)
PROX_LOGP_METHOD_METRICS = "metrics"  # Recompute + compute approximation metrics

# List of all valid prox_logp_method values for configuration
PROX_LOGP_METHODS_ALL = [
    PROX_LOGP_METHOD_RECOMPUTE,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
]

# Methods that skip the forward pass (optimization enabled)
PROX_LOGP_METHODS_SKIP_FORWARD = [
    PROX_LOGP_METHOD_LOGLINEAR,
]

# Approximation method names used in compute_prox_logp_approximations()
PROX_APPROX_METHOD_LOGLINEAR = "loglinear"  # Log-linear interpolation
PROX_APPROX_METHOD_LINEAR = "linear"  # Linear interpolation in probability space
PROX_APPROX_METHOD_ROLLOUT = "rollout"  # Use behavior policy directly

# List of all approximation methods computed for metrics comparison
PROX_APPROX_METHODS_ALL = [
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
]
