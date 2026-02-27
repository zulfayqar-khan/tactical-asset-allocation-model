"""
baselines/mean_variance.py — Mean-Variance Optimization (Markowitz) baseline.
Phase 4: Baselines.

Uses 252-day rolling window for expected returns (sample mean) and covariance.
Solves for max Sharpe ratio portfolio with long-only, max 30% constraints.
Rebalances monthly.

Runnable standalone: python baselines/mean_variance.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict
from scipy.optimize import minimize


def solve_max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float,
    n_assets: int,
) -> np.ndarray:
    """Solve for the maximum Sharpe ratio portfolio (long-only, bounded).

    Args:
        mu: Expected return vector (n_assets,).
        cov: Covariance matrix (n_assets, n_assets).
        max_weight: Maximum single-asset weight.
        n_assets: Number of assets.

    Returns:
        Optimal weight vector.
    """
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret / port_vol)

    bounds = [(0.0, max_weight)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n_assets) / n_assets

    result = minimize(
        neg_sharpe, x0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if result.success:
        return result.x
    else:
        # Fallback to equal weight if optimization fails
        return np.ones(n_assets) / n_assets


def mean_variance_func(
    date: pd.Timestamp,
    returns_history: pd.DataFrame,
    current_weights: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Mean-variance weight function. Monthly rebalance with 252-day lookback."""
    n_assets = len(current_weights)
    lookback = config["baselines"]["mean_variance"]["lookback_days"]
    max_weight = config["portfolio"]["max_single_weight"]

    # Only rebalance at month start
    if date.day > 5:
        return current_weights

    # Need enough history
    if len(returns_history) < lookback:
        return np.ones(n_assets) / n_assets

    recent = returns_history.iloc[-lookback:]
    mu = recent.mean().values * 252  # Annualize
    cov = recent.cov().values * 252  # Annualize

    # Regularize covariance to avoid singularity
    cov += np.eye(n_assets) * 1e-6

    return solve_max_sharpe(mu, cov, max_weight, n_assets)
