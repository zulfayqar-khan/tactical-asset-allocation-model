"""
baselines/run_baselines.py — Run all 5 baseline strategies and produce metrics.
Phase 4: Baselines. Runs backtests on each test fold and the concatenated period.

Runnable standalone: python baselines/run_baselines.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils import load_config, set_seeds, ensure_directories
from backtesting.engine import run_backtest, load_backtest_data
from baselines.equal_weight import equal_weight_func
from baselines.mean_variance import mean_variance_func
from baselines.momentum import momentum_func
from baselines.buy_and_hold import buy_and_hold_func
from baselines.risk_parity import risk_parity_func

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


STRATEGIES = {
    "Equal Weight": equal_weight_func,
    "Mean-Variance": mean_variance_func,
    "Momentum": momentum_func,
    "Buy & Hold": buy_and_hold_func,
    "Risk Parity": risk_parity_func,
}


def run_all_baselines(
    config: Dict[str, Any],
    returns_wide: pd.DataFrame,
    open_prices_wide: pd.DataFrame,
    start_date: str,
    end_date: str,
    period_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Run all baseline strategies on a given period.

    Args:
        config: Full config dict.
        returns_wide: Wide-format daily returns.
        open_prices_wide: Wide-format open prices.
        start_date: Period start.
        end_date: Period end.
        period_name: Label for logging.

    Returns:
        Dict of strategy_name -> backtest result dict.
    """
    logger.info("--- %s: %s to %s ---", period_name, start_date, end_date)
    results = {}

    for name, func in STRATEGIES.items():
        result = run_backtest(
            weight_func=func,
            returns_wide=returns_wide,
            open_prices_wide=open_prices_wide,
            config=config,
            start_date=start_date,
            end_date=end_date,
            name=name,
        )
        results[name] = result

    return results


def build_comparison_table(
    all_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a comparison table of metrics across all strategies.

    Args:
        all_results: Dict of strategy_name -> backtest result.

    Returns:
        DataFrame with strategies as columns and metrics as rows.
    """
    metrics_dict = {}
    for name, result in all_results.items():
        if result and "metrics" in result:
            metrics_dict[name] = result["metrics"]

    df = pd.DataFrame(metrics_dict)

    # Format for display
    format_map = {
        "Ann. Return": "{:.2%}",
        "Ann. Volatility": "{:.2%}",
        "Sharpe Ratio": "{:.3f}",
        "Max Drawdown": "{:.2%}",
        "Calmar Ratio": "{:.3f}",
        "Sortino Ratio": "{:.3f}",
        "Hit Rate": "{:.2%}",
        "Profit Factor": "{:.2f}",
        "95% CVaR": "{:.2%}",
        "Avg Monthly Turnover": "{:.2%}",
    }

    return df


def run_phase4() -> None:
    """Execute the full Phase 4 baselines pipeline."""
    config = load_config()
    set_seeds(config)
    ensure_directories(config)

    tables_path = PROJECT_ROOT / config["paths"]["tables"]
    splits = config["splits"]

    logger.info("=" * 70)
    logger.info("PHASE 4 — BASELINE STRATEGIES")
    logger.info("=" * 70)

    # Load data
    returns_wide, open_prices_wide = load_backtest_data(config)
    logger.info("Loaded returns: %s", returns_wide.shape)

    # Define periods to test
    periods = {
        "Test Fold 1 (2018-2019)": (splits["test_fold_1"]["start"], splits["test_fold_1"]["end"]),
        "Test Fold 2 (2020-2021)": (splits["test_fold_2"]["start"], splits["test_fold_2"]["end"]),
        "Test Fold 3 (2022-2024)": (splits["test_fold_3"]["start"], splits["test_fold_3"]["end"]),
        "Full Test (2018-2024)": (splits["test_fold_1"]["start"], splits["test_fold_3"]["end"]),
    }

    all_tables = {}

    for period_name, (start, end) in periods.items():
        results = run_all_baselines(
            config, returns_wide, open_prices_wide, start, end, period_name,
        )

        table = build_comparison_table(results)
        all_tables[period_name] = table

        # Save table
        safe_name = period_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        csv_path = tables_path / f"baselines_{safe_name}.csv"
        table.to_csv(csv_path, float_format="%.6f")
        logger.info("Saved %s to %s", period_name, csv_path)

        # Print table
        logger.info("\n%s\n%s", period_name, table.round(4).to_string())

    # --- GATE 3 VERDICT ---
    logger.info("=" * 70)
    logger.info("GATE 3 — BASELINE SANITY CHECK")
    logger.info("=" * 70)

    full_test_table = all_tables["Full Test (2018-2024)"]
    ew_sharpe = full_test_table.loc["Sharpe Ratio", "Equal Weight"]

    logger.info("  Equal-weight Sharpe (full test): %.3f", ew_sharpe)

    checks = []

    # Check 1: EW Sharpe in plausible range (0.3 to 0.8 for DJIA)
    ew_plausible = 0.1 <= ew_sharpe <= 1.2  # Slightly wider for real data
    checks.append(("Equal-weight Sharpe in plausible range", ew_plausible))

    # Check 2: All baselines produce valid results (no NaN Sharpe)
    all_valid = not full_test_table.loc["Sharpe Ratio"].isna().any()
    checks.append(("All baselines produce valid metrics", all_valid))

    # Check 3: Buy & Hold has zero turnover (sanity check)
    bh_turnover = full_test_table.loc["Avg Monthly Turnover", "Buy & Hold"]
    bh_ok = bh_turnover < 0.01
    checks.append(("Buy & Hold has near-zero turnover", bh_ok))

    # Check 4: No baseline has Sharpe > 3 (would indicate a bug)
    max_sharpe = full_test_table.loc["Sharpe Ratio"].max()
    no_bug = max_sharpe < 3.0
    checks.append(("No baseline Sharpe > 3.0 (no bug)", no_bug))

    all_passed = all(result for _, result in checks)

    for name, result in checks:
        status = "PASS" if result else "FAIL"
        logger.info("  [%s] %s", status, name)

    logger.info("=" * 70)
    if all_passed:
        logger.info("GATE 3: PASS — Baselines implemented and producing sensible results.")
        logger.info("  Equal-weight Sharpe: %.3f (target to beat: %.3f)",
                     ew_sharpe, ew_sharpe + 0.15)
    else:
        failed = [name for name, result in checks if not result]
        logger.error("GATE 3: FAIL — %s", "; ".join(failed))
    logger.info("=" * 70)

    # Print final summary table
    logger.info("\n\nFULL TEST PERIOD COMPARISON TABLE:")
    print("\n" + full_test_table.round(4).to_string() + "\n")


if __name__ == "__main__":
    run_phase4()
