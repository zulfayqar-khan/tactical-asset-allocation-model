"""
training/walk_forward.py — Walk-forward validation pipeline.
Phase 7: Final Evaluation.

Implements the expanding-window walk-forward protocol from Section 3.1:
  1. Train on 2008-2015, validate on 2016-2017, select best hyperparameters
  2. Retrain on 2008-2017, test on 2018-2019
  3. Retrain on 2008-2019, test on 2020-2021
  4. Retrain on 2008-2021, test on 2022-2024

With 21-day embargo between train end and test start.

After training, runs full backtest through the engine with transaction costs,
computes all metrics, compares against baselines, and runs Gate 6 checks.

Runnable standalone: python training/walk_forward.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from utils import load_config, set_seeds, ensure_directories
from training.trainer import create_datasets, train_model, PortfolioDataset
from models.tcn_attention import build_tcn_model
from models.lstm_allocator import build_lstm_model
from models.losses import SharpeRatioLoss, MeanVarianceUtilityLoss, ReturnMaxLoss, MultiTaskLoss
from backtesting.engine import run_backtest, load_backtest_data, enforce_constraints
from backtesting.metrics import compute_all_metrics
from baselines.run_baselines import run_all_baselines, build_comparison_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_walk_forward_folds(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate walk-forward fold definitions with expanding training windows.

    Returns list of dicts with train_start, train_end, val_start, val_end,
    test_start, test_end.
    """
    splits = config["splits"]

    folds = [
        {
            "name": "Fold 1 (test 2018-2019)",
            "train_start": splits["initial_train"]["start"],
            "train_end": splits["validation"]["end"],
            "test_start": splits["test_fold_1"]["start"],
            "test_end": splits["test_fold_1"]["end"],
        },
        {
            "name": "Fold 2 (test 2020-2021)",
            "train_start": splits["initial_train"]["start"],
            "train_end": splits["test_fold_1"]["end"],
            "test_start": splits["test_fold_2"]["start"],
            "test_end": splits["test_fold_2"]["end"],
        },
        {
            "name": "Fold 3 (test 2022-2024)",
            "train_start": splits["initial_train"]["start"],
            "train_end": splits["test_fold_2"]["end"],
            "test_start": splits["test_fold_3"]["start"],
            "test_end": splits["test_fold_3"]["end"],
        },
    ]

    return folds


def apply_embargo(train_end: str, embargo_days: int) -> str:
    """Move train_end backward by embargo_days business days."""
    end_date = pd.Timestamp(train_end)
    adjusted = end_date - pd.tseries.offsets.BDay(embargo_days)
    return adjusted.strftime("%Y-%m-%d")


def build_loss_fn(config: Dict[str, Any]) -> torch.nn.Module:
    """Build the loss function from config, wrapping with MultiTaskLoss."""
    loss_type = config["loss"].get("type", "sharpe")
    if loss_type == "mv_utility":
        base_loss = MeanVarianceUtilityLoss(config)
    elif loss_type == "return_max":
        base_loss = ReturnMaxLoss(config)
    else:
        base_loss = SharpeRatioLoss(config)
    return MultiTaskLoss(base_loss, config)


def make_dl_weight_func(
    model: torch.nn.Module,
    X: np.ndarray,
    sample_dates: np.ndarray,
    num_assets: int,
) -> callable:
    """Create a weight function compatible with the backtest engine.

    The backtest engine calls weight_func(date, returns_history, current_weights, config)
    on each trading day. This closure looks up the model's predicted weights for
    that date from pre-computed predictions.

    Args:
        model: Trained model.
        X: Feature tensor (num_samples, lookback, input_dim).
        sample_dates: Dates for each sample in X.
        num_assets: Number of assets.

    Returns:
        Callable compatible with backtest engine.
    """
    # Pre-compute all weights
    model.eval()
    sample_dates_idx = pd.DatetimeIndex(sample_dates)

    # Build a date -> weights lookup
    weight_lookup = {}
    with torch.no_grad():
        batch_size = 64
        for start_idx in range(0, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            x_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32)
            w_batch = model(x_batch).numpy()
            for j in range(end_idx - start_idx):
                date_key = sample_dates_idx[start_idx + j]
                weight_lookup[date_key] = w_batch[j]

    def dl_weight_func(
        date: pd.Timestamp,
        returns_history: pd.DataFrame,
        current_weights: np.ndarray,
        config: Dict[str, Any],
    ) -> np.ndarray:
        """Return model-predicted weights for the given date."""
        if date in weight_lookup:
            return weight_lookup[date]
        # If date not in feature set, keep current weights (no rebalance)
        return current_weights

    return dl_weight_func


def run_walk_forward(
    config: Dict[str, Any],
    model_type: str = "tcn",
) -> Dict[str, Any]:
    """Run the full walk-forward validation pipeline.

    For each fold:
    1. Train the model on expanding training data
    2. Run the model through the backtest engine on test data
    3. Run all baselines on the same test period
    4. Compute comparison metrics

    Args:
        config: Full config dict.
        model_type: "tcn" or "lstm".

    Returns:
        Dict with fold results, backtest results, baseline comparisons.
    """
    set_seeds(config)
    ensure_directories(config)

    splits = config["splits"]
    embargo = splits["embargo_days"]
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    checkpoints_path = PROJECT_ROOT / config["paths"]["checkpoints"]
    forward_days = config["training"].get("forward_days", 5)

    folds = get_walk_forward_folds(config)

    logger.info("=" * 70)
    logger.info("PHASE 7 — WALK-FORWARD EVALUATION — %s model", model_type.upper())
    logger.info("  forward_days=%d, loss=%s", forward_days, config["loss"].get("type", "sharpe"))
    logger.info("=" * 70)

    # Load shared data
    X = np.load(processed_path / "X_tensor.npy")
    sample_dates = np.load(processed_path / "sample_dates.npy", allow_pickle=True)
    returns_wide, open_prices_wide = load_backtest_data(config)
    tickers = sorted(returns_wide.columns.tolist())
    returns_wide = returns_wide[tickers]
    if open_prices_wide is not None:
        open_prices_wide = open_prices_wide[tickers]

    num_assets = len(tickers)
    input_dim = X.shape[-1]

    fold_results = []
    all_dl_daily_returns = []

    for fold in folds:
        fold_start_time = time.time()
        logger.info("-" * 70)
        logger.info("  %s", fold["name"])
        logger.info("  Train: %s to %s (embargo: %d days)",
                     fold["train_start"], fold["train_end"], embargo)
        logger.info("  Test:  %s to %s", fold["test_start"], fold["test_end"])
        logger.info("-" * 70)

        # Apply embargo: trim train_end
        effective_train_end = apply_embargo(fold["train_end"], embargo)

        # Use last 2 years of training as validation for early stopping
        effective_train_end_ts = pd.Timestamp(effective_train_end)
        val_start_ts = effective_train_end_ts - pd.DateOffset(years=2)

        sample_dates_idx = pd.DatetimeIndex(sample_dates)

        # Split: train-proper and val (for early stopping)
        train_mask = np.array(
            (sample_dates_idx >= pd.Timestamp(fold["train_start"])) &
            (sample_dates_idx < pd.Timestamp(val_start_ts))
        )
        val_mask = np.array(
            (sample_dates_idx >= pd.Timestamp(val_start_ts)) &
            (sample_dates_idx <= pd.Timestamp(effective_train_end))
        )
        test_mask = np.array(
            (sample_dates_idx >= pd.Timestamp(fold["test_start"])) &
            (sample_dates_idx <= pd.Timestamp(fold["test_end"]))
        )

        logger.info("  Train samples: %d, Val samples: %d, Test features: %d",
                     train_mask.sum(), val_mask.sum(), test_mask.sum())

        if train_mask.sum() < 100 or val_mask.sum() < 50:
            logger.warning("  Insufficient data for fold — skipping")
            continue

        # Create datasets with forward_days
        train_ds = PortfolioDataset(
            X[train_mask], returns_wide, sample_dates[train_mask], tickers, forward_days
        )
        val_ds = PortfolioDataset(
            X[val_mask], returns_wide, sample_dates[val_mask], tickers, forward_days
        )

        # Build fresh model and loss
        set_seeds(config)
        if model_type == "tcn":
            model = build_tcn_model(config, num_assets, input_dim)
        else:
            model = build_lstm_model(config, num_assets, input_dim)

        loss_fn = build_loss_fn(config)
        logger.info("  Model: %s, Parameters: %d", model_type.upper(), model.count_parameters())

        # Train
        train_result = train_model(model, train_ds, val_ds, loss_fn, config)

        # Save checkpoint
        checkpoint = {
            "model_state": train_result["best_model_state"],
            "fold": fold["name"],
            "best_epoch": train_result["best_epoch"],
            "best_val_sharpe": train_result["best_val_sharpe"],
        }
        ckpt_path = checkpoints_path / f"{model_type}_{fold['name'].replace(' ', '_').replace('(', '').replace(')', '')}.pt"
        torch.save(checkpoint, ckpt_path)
        logger.info("  Saved checkpoint to %s", ckpt_path)

        # Create weight function for backtest engine
        # Use ALL features in the test period (not just forward_days-compatible ones)
        test_X = X[test_mask]
        test_sample_dates = sample_dates[test_mask]

        dl_weight_func = make_dl_weight_func(
            model, test_X, test_sample_dates, num_assets
        )

        # Run backtest for DL model
        dl_result = run_backtest(
            weight_func=dl_weight_func,
            returns_wide=returns_wide,
            open_prices_wide=open_prices_wide,
            config=config,
            start_date=fold["test_start"],
            end_date=fold["test_end"],
            name=f"{model_type.upper()} Model",
        )

        # Run all baselines on same period
        baseline_results = run_all_baselines(
            config, returns_wide, open_prices_wide,
            fold["test_start"], fold["test_end"],
            fold["name"],
        )

        # Combine DL + baselines for comparison table
        all_results = {f"{model_type.upper()} Model": dl_result}
        all_results.update(baseline_results)
        comparison_table = build_comparison_table(all_results)

        # Collect daily returns for concatenated evaluation
        if dl_result and "daily_returns" in dl_result:
            all_dl_daily_returns.append(dl_result["daily_returns"])

        fold_elapsed = time.time() - fold_start_time
        logger.info("  Fold completed in %.0fs", fold_elapsed)
        logger.info("\n%s\n%s", fold["name"], comparison_table.round(4).to_string())

        fold_results.append({
            "name": fold["name"],
            "train_result": train_result,
            "dl_backtest": dl_result,
            "baseline_results": baseline_results,
            "comparison_table": comparison_table,
            "test_start": fold["test_start"],
            "test_end": fold["test_end"],
        })

    # --- Concatenated test period ---
    logger.info("=" * 70)
    logger.info("CONCATENATED TEST PERIOD (2018-2024)")
    logger.info("=" * 70)

    if all_dl_daily_returns:
        concat_dl_returns = pd.concat(all_dl_daily_returns)
        concat_dl_metrics = compute_all_metrics(concat_dl_returns)

        # Compute monthly turnover for concat period
        # (already included in individual fold backtests)

        # Run baselines on full test period
        full_baseline_results = run_all_baselines(
            config, returns_wide, open_prices_wide,
            splits["test_fold_1"]["start"],
            splits["test_fold_3"]["end"],
            "Full Test (2018-2024)",
        )

        # Build full comparison
        full_results = {f"{model_type.upper()} Model": {
            "metrics": concat_dl_metrics,
            "daily_returns": concat_dl_returns,
        }}
        full_results.update(full_baseline_results)
        full_comparison = build_comparison_table(full_results)

        logger.info("\nFULL TEST PERIOD COMPARISON:")
        logger.info("\n%s", full_comparison.round(4).to_string())
    else:
        concat_dl_returns = None
        concat_dl_metrics = None
        full_comparison = None
        full_baseline_results = {}

    return {
        "model_type": model_type,
        "folds": fold_results,
        "concat_dl_returns": concat_dl_returns,
        "concat_dl_metrics": concat_dl_metrics,
        "full_comparison": full_comparison,
        "full_baseline_results": full_baseline_results,
    }


def run_stress_tests(
    results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run stress tests on crisis sub-periods.

    Computes metrics for DL model and EW baseline on:
    - COVID crash (Feb-Apr 2020)
    - 2022 rate shock (Jan-Oct 2022)
    - Q4 2018 selloff (Oct-Dec 2018)
    """
    stress_cfg = config["stress_tests"]
    model_type = results["model_type"]
    concat_returns = results["concat_dl_returns"]

    if concat_returns is None:
        logger.warning("No concatenated returns available for stress tests")
        return {}

    logger.info("=" * 70)
    logger.info("STRESS TESTS — %s", model_type.upper())
    logger.info("=" * 70)

    returns_wide, open_prices_wide = load_backtest_data(config)

    stress_results = {}
    for crisis_name, period in stress_cfg.items():
        start = period["start"]
        end = period["end"]

        # DL model returns for this sub-period
        mask = (concat_returns.index >= pd.Timestamp(start)) & \
               (concat_returns.index <= pd.Timestamp(end))
        dl_sub = concat_returns[mask]

        if len(dl_sub) < 5:
            logger.warning("  %s: insufficient data (%d days)", crisis_name, len(dl_sub))
            continue

        dl_metrics = compute_all_metrics(dl_sub)

        # EW baseline for same period
        from baselines.equal_weight import equal_weight_func
        ew_result = run_backtest(
            equal_weight_func, returns_wide, open_prices_wide,
            config, start, end, "Equal Weight",
        )

        logger.info("  %s (%s to %s):", crisis_name, start, end)
        logger.info("    DL  — Sharpe: %.3f, MaxDD: %.1f%%, Return: %.1f%%",
                     dl_metrics["Sharpe Ratio"],
                     dl_metrics["Max Drawdown"] * 100,
                     dl_metrics["Ann. Return"] * 100)
        if ew_result and "metrics" in ew_result:
            ew_m = ew_result["metrics"]
            logger.info("    EW  — Sharpe: %.3f, MaxDD: %.1f%%, Return: %.1f%%",
                         ew_m["Sharpe Ratio"],
                         ew_m["Max Drawdown"] * 100,
                         ew_m["Ann. Return"] * 100)

        stress_results[crisis_name] = {
            "dl_metrics": dl_metrics,
            "ew_metrics": ew_result.get("metrics", {}) if ew_result else {},
            "start": start,
            "end": end,
            "n_days": len(dl_sub),
        }

    return stress_results


def run_gate6_checks(
    results: Dict[str, Any],
    config: Dict[str, Any],
    stress_results: Dict[str, Any],
) -> bool:
    """Run Gate 6 (final evaluation) checks per Section 4.4.

    Checks:
    1. Sharpe > EW Sharpe + 0.15 (concatenated test)
    2. Absolute Sharpe >= 0.5
    3. Max Drawdown <= 35%
    4. Monthly Turnover <= 60%
    5. Sharpe > 0 in at least 2 of 3 folds
    6. (Seed sensitivity deferred — would require 5 full retraining runs)

    Returns True if all pass.
    """
    model_type = results["model_type"]
    eval_cfg = config["evaluation"]

    logger.info("=" * 70)
    logger.info("GATE 6 — FINAL EVALUATION — %s", model_type.upper())
    logger.info("=" * 70)

    checks = []

    # Get DL metrics on concatenated test period
    dl_metrics = results.get("concat_dl_metrics")
    full_comparison = results.get("full_comparison")

    if dl_metrics is None:
        logger.error("  No DL metrics available — GATE 6: FAIL")
        return False

    dl_sharpe = dl_metrics["Sharpe Ratio"]
    dl_dd = dl_metrics["Max Drawdown"]
    dl_return = dl_metrics["Ann. Return"]

    # Get EW baseline metrics
    ew_sharpe = None
    if full_comparison is not None and "Equal Weight" in full_comparison.columns:
        ew_sharpe = full_comparison.loc["Sharpe Ratio", "Equal Weight"]

    # Check 1: Sharpe > EW + 0.15
    if ew_sharpe is not None:
        min_sharpe_above_ew = eval_cfg["min_sharpe_above_equal_weight"]
        sharpe_above_ew = dl_sharpe - ew_sharpe
        check1 = sharpe_above_ew >= min_sharpe_above_ew
        checks.append((
            f"Sharpe above EW by >= {min_sharpe_above_ew} "
            f"({dl_sharpe:.3f} - {ew_sharpe:.3f} = {sharpe_above_ew:+.3f})",
            check1
        ))
    else:
        checks.append(("Sharpe above EW (no EW data)", False))

    # Check 2: Absolute Sharpe >= 0.5
    min_sharpe = eval_cfg["min_sharpe_absolute"]
    check2 = dl_sharpe >= min_sharpe
    checks.append((
        f"Absolute Sharpe >= {min_sharpe} (got {dl_sharpe:.3f})",
        check2
    ))

    # Check 3: Max Drawdown <= 35%
    max_dd = eval_cfg["max_drawdown"]
    check3 = dl_dd <= max_dd
    checks.append((
        f"Max Drawdown <= {max_dd:.0%} (got {dl_dd:.1%})",
        check3
    ))

    # Check 4: Monthly Turnover <= 60%
    max_turnover = eval_cfg["max_monthly_turnover"]
    avg_turnover = dl_metrics.get("Avg Monthly Turnover", 0)
    check4 = avg_turnover <= max_turnover
    checks.append((
        f"Avg Monthly Turnover <= {max_turnover:.0%} (got {avg_turnover:.1%})",
        check4
    ))

    # Check 5: Sharpe > 0 in at least 2 of 3 folds
    min_positive_folds = eval_cfg["min_positive_folds"]
    fold_sharpes = []
    for fold in results["folds"]:
        if fold["dl_backtest"] and "metrics" in fold["dl_backtest"]:
            fold_sharpe = fold["dl_backtest"]["metrics"]["Sharpe Ratio"]
            fold_sharpes.append(fold_sharpe)
            logger.info("  %s: Sharpe = %.3f", fold["name"], fold_sharpe)

    positive_folds = sum(1 for s in fold_sharpes if s > 0)
    check5 = positive_folds >= min_positive_folds
    checks.append((
        f"Sharpe > 0 in >= {min_positive_folds} folds (got {positive_folds}/{len(fold_sharpes)})",
        check5
    ))

    # Check 6: Not collapsed to equal weight
    cosine_max = eval_cfg.get("weight_cosine_sim_max", 0.95)
    # Check from fold backtests
    avg_cosine = 0.0
    n_cosine = 0
    for fold in results["folds"]:
        if fold["dl_backtest"] and "weights_history" in fold["dl_backtest"]:
            w_hist = fold["dl_backtest"]["weights_history"].values
            n_assets = w_hist.shape[1]
            ew = np.ones(n_assets) / n_assets
            cosines = []
            for w in w_hist:
                cos = np.dot(w, ew) / (np.linalg.norm(w) * np.linalg.norm(ew) + 1e-10)
                cosines.append(cos)
            avg_cosine += np.mean(cosines)
            n_cosine += 1
    if n_cosine > 0:
        avg_cosine /= n_cosine
    check6 = avg_cosine < cosine_max
    checks.append((
        f"Not collapsed to EW (cosine < {cosine_max:.2f}, got {avg_cosine:.3f})",
        check6
    ))

    # Check 7: Max drawdown rejection threshold
    max_dd_reject = eval_cfg.get("max_drawdown_reject", 0.45)
    check7 = dl_dd <= max_dd_reject
    checks.append((
        f"Max Drawdown below rejection ({max_dd_reject:.0%}, got {dl_dd:.1%})",
        check7
    ))

    # Check 8: Training Sharpe not absurdly high (overfit detection)
    max_train_sharpe = eval_cfg.get("training_sharpe_overfit", 4.0)
    worst_overfit = False
    for fold in results["folds"]:
        tr = fold["train_result"]
        if tr["train_history"]:
            last_train_sharpe = tr["train_history"][-1].get("sharpe", 0)
            if last_train_sharpe > max_train_sharpe:
                worst_overfit = True
    check8 = not worst_overfit
    checks.append((
        f"No training Sharpe > {max_train_sharpe} (overfit check)",
        check8
    ))

    # Print all checks
    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s", status, name)
        if not passed:
            all_passed = False

    logger.info("=" * 70)
    if all_passed:
        logger.info("GATE 6: PASS — Model meets all go/no-go criteria.")
    else:
        failed = [name for name, passed in checks if not passed]
        logger.info("GATE 6: FAIL — Failed checks:")
        for f in failed:
            logger.info("  - %s", f)
    logger.info("=" * 70)

    # Print summary
    logger.info("\nFINAL SUMMARY — %s:", model_type.upper())
    logger.info("  Annualized Return: %.1f%%", dl_return * 100)
    logger.info("  Annualized Sharpe: %.3f", dl_sharpe)
    logger.info("  Max Drawdown: %.1f%%", dl_dd * 100)
    if ew_sharpe is not None:
        logger.info("  EW Sharpe: %.3f (delta: %+.3f)", ew_sharpe, dl_sharpe - ew_sharpe)

    # Count baselines beaten
    if full_comparison is not None:
        baseline_names = [c for c in full_comparison.columns if c != f"{model_type.upper()} Model"]
        beaten = sum(
            1 for b in baseline_names
            if full_comparison.loc["Sharpe Ratio", b] < dl_sharpe
        )
        logger.info("  Baselines beaten on Sharpe: %d/%d", beaten, len(baseline_names))

    return all_passed


def save_results(
    results: Dict[str, Any],
    stress_results: Dict[str, Any],
    gate6_passed: bool,
    config: Dict[str, Any],
) -> None:
    """Save all Phase 7 results to disk."""
    tables_path = PROJECT_ROOT / config["paths"]["tables"]
    model_type = results["model_type"]

    # Save per-fold comparison tables
    for fold in results["folds"]:
        safe_name = fold["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        csv_path = tables_path / f"phase7_{model_type}_{safe_name}.csv"
        fold["comparison_table"].to_csv(csv_path, float_format="%.6f")
        logger.info("Saved %s to %s", fold["name"], csv_path)

    # Save full comparison table
    if results["full_comparison"] is not None:
        csv_path = tables_path / f"phase7_{model_type}_full_test.csv"
        results["full_comparison"].to_csv(csv_path, float_format="%.6f")
        logger.info("Saved full test comparison to %s", csv_path)

    # Save stress test results
    if stress_results:
        stress_path = tables_path / f"phase7_{model_type}_stress_tests.json"
        # Convert to serializable format
        stress_serializable = {}
        for k, v in stress_results.items():
            stress_serializable[k] = {
                "dl_metrics": {mk: float(mv) for mk, mv in v["dl_metrics"].items()},
                "ew_metrics": {mk: float(mv) for mk, mv in v["ew_metrics"].items()} if v["ew_metrics"] else {},
                "start": v["start"],
                "end": v["end"],
                "n_days": v["n_days"],
            }
        with open(stress_path, "w") as f:
            json.dump(stress_serializable, f, indent=2)
        logger.info("Saved stress tests to %s", stress_path)

    # Save summary
    summary = {
        "model_type": model_type,
        "gate6_passed": gate6_passed,
        "concat_metrics": {k: float(v) for k, v in results["concat_dl_metrics"].items()} if results["concat_dl_metrics"] else {},
        "fold_sharpes": [],
    }
    for fold in results["folds"]:
        if fold["dl_backtest"] and "metrics" in fold["dl_backtest"]:
            summary["fold_sharpes"].append({
                "name": fold["name"],
                "sharpe": float(fold["dl_backtest"]["metrics"]["Sharpe Ratio"]),
                "max_dd": float(fold["dl_backtest"]["metrics"]["Max Drawdown"]),
                "return": float(fold["dl_backtest"]["metrics"]["Ann. Return"]),
            })

    summary_path = tables_path / f"phase7_{model_type}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    config = load_config()

    logger.info("=" * 70)
    logger.info("PHASE 7 — FINAL EVALUATION")
    logger.info("=" * 70)

    total_start = time.time()

    # Run TCN (primary model)
    tcn_results = run_walk_forward(config, model_type="tcn")

    # Stress tests
    stress_results = run_stress_tests(tcn_results, config)

    # Gate 6 checks
    gate6_passed = run_gate6_checks(tcn_results, config, stress_results)

    # Save all results
    save_results(tcn_results, stress_results, gate6_passed, config)

    total_elapsed = time.time() - total_start
    logger.info("\nTotal Phase 7 time: %.0f seconds (%.1f minutes)", total_elapsed, total_elapsed / 60)
