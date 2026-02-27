"""
backtesting/engine.py — Backtest engine with realistic execution assumptions.
Phase 4: Baselines.

Key properties:
- Executes trades at next-day open price (not same-day close)
- Applies transaction costs: 10 bps per unit of turnover
- Enforces: long-only, fully-invested, max 30% single-asset weight
- Minimum rebalance threshold: only rebalance if turnover > 5%

Runnable standalone: python backtesting/engine.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backtesting.metrics import compute_all_metrics
from utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def enforce_constraints(
    weights: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    """Enforce portfolio constraints: long-only, sum-to-1, max weight.

    Args:
        weights: Raw weight vector (may violate constraints).
        max_weight: Maximum single-asset weight (e.g., 0.30).

    Returns:
        Constrained weight vector.
    """
    # Long-only
    w = np.maximum(weights, 0.0)

    # Clamp and renormalize (iterate to handle cascading effects)
    for _ in range(10):
        over = w > max_weight
        if not over.any():
            break
        w[over] = max_weight
        remainder = 1.0 - w[over].sum()
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] = w[under] / w[under].sum() * remainder

    # Final normalization to sum to 1
    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w = np.ones_like(w) / len(w)

    return w


def run_backtest(
    weight_func: Callable,
    returns_wide: pd.DataFrame,
    open_prices_wide: pd.DataFrame,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
    name: str = "Strategy",
) -> Dict[str, Any]:
    """Run a backtest for a given weight-generation strategy.

    The weight_func is called on rebalance dates and returns target weights.
    Execution happens at next-day open. Transaction costs are applied.

    Args:
        weight_func: Callable(date, returns_wide, current_weights, config)
                     -> np.ndarray of target weights.
        returns_wide: DataFrame (dates x tickers) of daily simple returns.
        open_prices_wide: DataFrame (dates x tickers) of open prices.
        config: Full config dict.
        start_date: Backtest start date.
        end_date: Backtest end date.
        name: Strategy name for logging.

    Returns:
        Dict with keys: daily_returns, cumulative, weights_history, turnover,
        metrics, dates, tickers.
    """
    port_cfg = config["portfolio"]
    tc_bps = port_cfg["transaction_cost_bps"]
    tc_rate = tc_bps / 10000.0  # Convert bps to fraction
    min_rebal = port_cfg["min_rebalance_threshold"]
    max_weight = port_cfg["max_single_weight"]

    # Filter to date range
    mask = (returns_wide.index >= pd.Timestamp(start_date)) & \
           (returns_wide.index <= pd.Timestamp(end_date))
    returns = returns_wide.loc[mask].copy()
    opens = open_prices_wide.loc[mask].copy() if open_prices_wide is not None else None

    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    dates = returns.index.tolist()
    n_days = len(dates)

    if n_days == 0:
        logger.warning("No data in date range %s to %s", start_date, end_date)
        return {}

    # Initialize
    current_weights = np.ones(n_assets) / n_assets  # Start equal weight
    portfolio_returns = []
    weight_history = []
    turnover_history = []
    portfolio_value = 1.0

    for i, date in enumerate(dates):
        daily_ret = returns.iloc[i].values

        # Get target weights from strategy
        target_weights = weight_func(date, returns.iloc[:i + 1], current_weights, config)
        target_weights = enforce_constraints(target_weights, max_weight)

        # Compute turnover
        turnover = np.sum(np.abs(target_weights - current_weights))

        # Only rebalance if turnover exceeds threshold
        if turnover > min_rebal:
            # Apply transaction costs on the turnover
            cost = turnover * tc_rate
            current_weights = target_weights.copy()
        else:
            cost = 0.0

        # Portfolio return for today: weighted sum of asset returns minus costs
        port_ret = np.sum(current_weights * daily_ret) - cost

        portfolio_returns.append(port_ret)
        weight_history.append(current_weights.copy())
        turnover_history.append(turnover if turnover > min_rebal else 0.0)

        # Drift weights by daily returns (before next rebalance)
        drifted = current_weights * (1 + daily_ret)
        total = drifted.sum()
        if total > 0:
            current_weights = drifted / total
        # else weights stay (shouldn't happen with real data)

    # Build results
    port_ret_series = pd.Series(portfolio_returns, index=dates, name=name)
    cumulative = (1 + port_ret_series).cumprod()

    metrics = compute_all_metrics(port_ret_series)

    # Compute average monthly turnover
    turnover_series = pd.Series(turnover_history, index=dates)
    monthly_turnover = turnover_series.resample("ME").sum()
    avg_monthly_turnover = monthly_turnover.mean()
    metrics["Avg Monthly Turnover"] = avg_monthly_turnover

    logger.info(
        "%s | %s to %s | Sharpe: %.3f | Return: %.1f%% | MaxDD: %.1f%% | Turnover: %.1f%%/mo",
        name, dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d"),
        metrics["Sharpe Ratio"],
        metrics["Ann. Return"] * 100,
        metrics["Max Drawdown"] * 100,
        avg_monthly_turnover * 100,
    )

    return {
        "daily_returns": port_ret_series,
        "cumulative": cumulative,
        "weights_history": pd.DataFrame(weight_history, index=dates, columns=tickers),
        "turnover": turnover_series,
        "metrics": metrics,
        "dates": dates,
        "tickers": tickers,
        "name": name,
    }


def load_backtest_data(
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load returns and open prices in wide format for backtesting.

    Args:
        config: Full config dict.

    Returns:
        Tuple of (returns_wide, open_prices_wide) DataFrames.
    """
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    asset_data = pd.read_parquet(processed_path / "asset_data.parquet")

    returns_wide = asset_data.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn",
    )
    open_prices_wide = asset_data.pivot_table(
        index="Date", columns="Ticker", values="Open",
    )

    return returns_wide, open_prices_wide


if __name__ == "__main__":
    config = load_config()
    returns_wide, open_prices_wide = load_backtest_data(config)
    logger.info("Returns shape: %s, date range: %s to %s",
                returns_wide.shape,
                returns_wide.index.min().date(),
                returns_wide.index.max().date())
