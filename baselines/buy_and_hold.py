"""
baselines/buy_and_hold.py — Buy-and-hold baseline strategy.
Phase 4: Baselines.

Invests equal weight at the start and never rebalances.
Weights drift naturally with returns.

Runnable standalone: python baselines/buy_and_hold.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict


def buy_and_hold_func(
    date: pd.Timestamp,
    returns_history: pd.DataFrame,
    current_weights: np.ndarray,
    config: Dict[str, Any],
) -> np.ndarray:
    """Buy-and-hold: always return current (drifted) weights.

    The backtest engine handles drift automatically. By returning
    current_weights, we signal no rebalancing desired.
    """
    return current_weights
