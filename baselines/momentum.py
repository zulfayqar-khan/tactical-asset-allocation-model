"""
baselines/momentum.py — Momentum strategy baseline.
Phase 4: Baselines.

Each month, rank assets by trailing 12-month return (excluding most recent month).
Top quartile: 2/N weight, bottom quartile: 0.5/N weight, middle: 1/N.
Normalize to sum to 1. Rebalance monthly.

Runnable standalone: python baselines/momentum.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict


def momentum_func(
    date: pd.Timestamp,
    returns_history: pd.DataFrame,
    current_weights: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Momentum weight function with monthly rebalancing."""
    n_assets = len(current_weights)
    mom_cfg = config["baselines"]["momentum"]
    lookback_months = mom_cfg["lookback_months"]
    skip_months = mom_cfg["skip_months"]
    top_mult = mom_cfg["top_quartile_multiplier"]
    bottom_mult = mom_cfg["bottom_quartile_multiplier"]

    # Only rebalance at month start
    if date.day > 5:
        return current_weights

    # Need ~252 days of history (12 months)
    min_days = lookback_months * 21
    if len(returns_history) < min_days:
        return np.ones(n_assets) / n_assets

    # Trailing 12-month return, excluding most recent month
    skip_days = skip_months * 21
    lookback_days = lookback_months * 21

    if len(returns_history) < lookback_days + skip_days:
        return np.ones(n_assets) / n_assets

    # Returns from [t-12mo, t-1mo]
    period_returns = returns_history.iloc[-(lookback_days + skip_days):-skip_days]
    cumulative = (1 + period_returns).prod() - 1  # Cumulative return per asset

    # Rank and assign weights
    ranks = cumulative.rank(ascending=True)
    n = len(ranks)
    q1 = n * 0.25  # Bottom quartile threshold
    q3 = n * 0.75  # Top quartile threshold

    weights = np.ones(n_assets) / n_assets  # Default 1/N

    for i, rank in enumerate(ranks.values):
        if rank <= q1:
            weights[i] = bottom_mult / n_assets
        elif rank > q3:
            weights[i] = top_mult / n_assets

    # Normalize to sum to 1
    weights = weights / weights.sum()
    return weights
