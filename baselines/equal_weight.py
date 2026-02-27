"""
baselines/equal_weight.py — Equal-weight (1/N) baseline strategy.
Phase 4: Baselines. Rebalances to 1/N monthly.

This is the hardest naive baseline to beat and the primary benchmark.

Runnable standalone: python baselines/equal_weight.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict


def equal_weight_func(
    date: pd.Timestamp,
    returns_history: pd.DataFrame,
    current_weights: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Return equal weights. Rebalancing is handled by the caller checking
    if a new month has started."""
    n = len(current_weights)
    return np.ones(n) / n


def get_rebalance_dates(dates: pd.DatetimeIndex) -> set:
    """Get the first trading day of each month as rebalance dates."""
    monthly = pd.Series(dates, index=dates).resample("MS").first()
    return set(monthly.values)


def equal_weight_with_monthly_rebal(
    date: pd.Timestamp,
    returns_history: pd.DataFrame,
    current_weights: np.ndarray,
    config: Dict[str, Any],
    _rebal_dates: set = None,
) -> np.ndarray:
    """Equal weight, only rebalance on month-start dates.

    On non-rebalance dates, returns current (drifted) weights.
    """
    n = len(current_weights)

    # Check if it's a rebalance date (first trading day of month)
    if _rebal_dates is not None:
        if date not in _rebal_dates:
            return current_weights
    else:
        # Fallback: rebalance if day <= 5 (first few days of month)
        if date.day > 5 and not np.allclose(current_weights, np.ones(n) / n, atol=0.001):
            return current_weights

    return np.ones(n) / n
