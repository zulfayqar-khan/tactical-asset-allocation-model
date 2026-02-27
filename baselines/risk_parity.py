"""
baselines/risk_parity.py — Risk parity baseline strategy.
Phase 4: Baselines.

Allocates inversely proportional to each asset's 63-day rolling volatility.
Normalize to sum to 1. Rebalance monthly.

Runnable standalone: python baselines/risk_parity.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict


def risk_parity_func(
    date: pd.Timestamp,
    returns_history: pd.DataFrame,
    current_weights: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Risk parity weight function with monthly rebalancing."""
    n_assets = len(current_weights)
    vol_lookback = config["baselines"]["risk_parity"]["vol_lookback"]

    # Only rebalance at month start
    if date.day > 5:
        return current_weights

    if len(returns_history) < vol_lookback:
        return np.ones(n_assets) / n_assets

    recent = returns_history.iloc[-vol_lookback:]
    vols = recent.std().values

    # Inverse volatility weights
    inv_vol = np.where(vols > 1e-10, 1.0 / vols, 0.0)
    total = inv_vol.sum()

    if total < 1e-10:
        return np.ones(n_assets) / n_assets

    weights = inv_vol / total
    return weights
