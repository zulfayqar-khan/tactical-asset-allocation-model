"""
backtesting/metrics.py — Performance metric computations.
Phase 4: Baselines. Computes all primary and secondary metrics from Section 4.2/4.3.

Runnable standalone: python backtesting/metrics.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict

import numpy as np
import pandas as pd


def annualized_return(daily_returns: pd.Series) -> float:
    """Annualized return from daily returns."""
    mean_daily = daily_returns.mean()
    return (1 + mean_daily) ** 252 - 1


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized volatility from daily returns."""
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(daily_returns: pd.Series) -> float:
    """Annualized Sharpe ratio (assuming risk-free rate ~ 0)."""
    vol = annualized_volatility(daily_returns)
    if vol < 1e-10:
        return 0.0
    return annualized_return(daily_returns) / vol


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown (as a positive fraction, e.g., 0.25 = 25% drawdown)."""
    cumulative = (1 + daily_returns).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / rolling_peak
    return abs(drawdown.min())


def calmar_ratio(daily_returns: pd.Series) -> float:
    """Calmar ratio: annualized return / max drawdown."""
    mdd = max_drawdown(daily_returns)
    if mdd < 1e-10:
        return 0.0
    return annualized_return(daily_returns) / mdd


def sortino_ratio(daily_returns: pd.Series) -> float:
    """Sortino ratio: annualized return / downside deviation."""
    downside = daily_returns[daily_returns < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = downside.std() * np.sqrt(252)
    if downside_std < 1e-10:
        return 0.0
    return annualized_return(daily_returns) / downside_std


def hit_rate(daily_returns: pd.Series) -> float:
    """Monthly hit rate: % of months with positive return."""
    monthly = daily_returns.resample("ME").sum()
    if len(monthly) == 0:
        return 0.0
    return (monthly > 0).mean()


def profit_factor(daily_returns: pd.Series) -> float:
    """Profit factor: gross profit / gross loss (monthly)."""
    monthly = daily_returns.resample("ME").sum()
    gross_profit = monthly[monthly > 0].sum()
    gross_loss = abs(monthly[monthly < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def cvar_95(daily_returns: pd.Series) -> float:
    """95% CVaR (Conditional Value at Risk) on monthly returns."""
    monthly = daily_returns.resample("ME").sum()
    if len(monthly) < 5:
        return 0.0
    cutoff = monthly.quantile(0.05)
    tail = monthly[monthly <= cutoff]
    if len(tail) == 0:
        return 0.0
    return tail.mean()


def compute_all_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """Compute all primary and secondary metrics.

    Args:
        daily_returns: Series of daily portfolio returns indexed by Date.

    Returns:
        Dict of metric_name -> value.
    """
    return {
        "Ann. Return": annualized_return(daily_returns),
        "Ann. Volatility": annualized_volatility(daily_returns),
        "Sharpe Ratio": sharpe_ratio(daily_returns),
        "Max Drawdown": max_drawdown(daily_returns),
        "Calmar Ratio": calmar_ratio(daily_returns),
        "Sortino Ratio": sortino_ratio(daily_returns),
        "Hit Rate": hit_rate(daily_returns),
        "Profit Factor": profit_factor(daily_returns),
        "95% CVaR": cvar_95(daily_returns),
    }


if __name__ == "__main__":
    # Quick self-test with random returns
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=500)
    fake_returns = pd.Series(np.random.randn(500) * 0.01 + 0.0003, index=dates)

    metrics = compute_all_metrics(fake_returns)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:+.4f}")
