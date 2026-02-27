"""
training/hyperparameter_search.py — Hyperparameter tuning via random search.
Phase 6: Hyperparameter Tuning.

Searches over the hyperparameter space defined in config.yaml using random search.
Budget: 15 configurations max (CPU constraint).
Uses validation set only (2016-2017). Train on 2008-2015.
21-day embargo between train end and validation start.

Gate 5: Best validation Sharpe must be positive and exceed equal-weight baseline.

Runnable standalone: python training/hyperparameter_search.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import load_config, set_seeds, ensure_directories
from training.trainer import create_datasets, train_model, PortfolioDataset
from models.tcn_attention import build_tcn_model, TCNAttentionModel
from models.lstm_allocator import build_lstm_model, LSTMAllocator
from models.losses import SharpeRatioLoss, MeanVarianceUtilityLoss, ReturnMaxLoss, MultiTaskLoss
from backtesting.engine import load_backtest_data, run_backtest
from baselines.equal_weight import equal_weight_func
from backtesting.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def sample_hyperparameters(config: Dict[str, Any], rng: np.random.RandomState) -> Dict[str, Any]:
    """Sample a random hyperparameter configuration from the search space.

    Args:
        config: Full config dict with search_space defined.
        rng: Random state for reproducibility.

    Returns:
        Dict of sampled hyperparameters.
    """
    space = config["hyperparameter_search"]["search_space"]

    # Learning rate: log-uniform
    lr_low = space["learning_rate"]["low"]
    lr_high = space["learning_rate"]["high"]
    lr = float(np.exp(rng.uniform(np.log(lr_low), np.log(lr_high))))

    # Categorical/grid choices
    lookback = int(rng.choice(space["lookback_window"]))
    hidden_channels = int(rng.choice(space["tcn_hidden_channels"]))
    dropout = float(rng.choice(space["dropout"]))
    lambda_turnover = float(rng.choice(space["lambda_turnover"]))
    lambda_concentration = float(rng.choice(space["lambda_concentration"]))
    lambda_entropy = float(rng.choice(space["lambda_entropy"]))
    softmax_temperature = float(rng.choice(space["softmax_temperature"]))
    batch_size = int(rng.choice(space["batch_size"]))

    # Loss function type and associated parameters
    loss_type = str(rng.choice(space["loss_type"]))
    gamma_risk_aversion = float(rng.choice(space["gamma_risk_aversion"]))
    alpha_return_pred = float(rng.choice(space["alpha_return_pred"]))
    alpha_vol_pred = float(rng.choice(space["alpha_vol_pred"]))

    return {
        "learning_rate": lr,
        "lookback_window": lookback,
        "tcn_hidden_channels": hidden_channels,
        "dropout": dropout,
        "lambda_turnover": lambda_turnover,
        "lambda_concentration": lambda_concentration,
        "lambda_entropy": lambda_entropy,
        "softmax_temperature": softmax_temperature,
        "batch_size": batch_size,
        "loss_type": loss_type,
        "gamma_risk_aversion": gamma_risk_aversion,
        "alpha_return_pred": alpha_return_pred,
        "alpha_vol_pred": alpha_vol_pred,
    }


def apply_hyperparameters(base_config: Dict[str, Any], hp: Dict[str, Any]) -> Dict[str, Any]:
    """Apply sampled hyperparameters to a config dict (deep copy).

    Args:
        base_config: Original config dict.
        hp: Sampled hyperparameters.

    Returns:
        Modified config dict.
    """
    config = copy.deepcopy(base_config)

    config["training"]["learning_rate"] = hp["learning_rate"]
    config["training"]["batch_size"] = hp["batch_size"]
    config["features"]["lookback_window"] = hp["lookback_window"]
    config["model"]["tcn"]["hidden_channels"] = hp["tcn_hidden_channels"]
    config["model"]["tcn"]["dropout"] = hp["dropout"]
    config["model"]["portfolio_head"]["dropout"] = min(hp["dropout"] + 0.1, 0.5)
    config["loss"]["lambda_turnover"] = hp["lambda_turnover"]
    config["loss"]["lambda_concentration"] = hp["lambda_concentration"]
    config["loss"]["lambda_entropy"] = hp["lambda_entropy"]
    config["model"]["softmax_temperature"] = hp["softmax_temperature"]

    # Loss function parameters
    config["loss"]["type"] = hp["loss_type"]
    config["loss"]["gamma_risk_aversion"] = hp["gamma_risk_aversion"]
    config["loss"]["alpha_return_pred"] = hp["alpha_return_pred"]
    config["loss"]["alpha_vol_pred"] = hp["alpha_vol_pred"]

    return config


def rebuild_features_for_lookback(
    config: Dict[str, Any],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or adapt feature tensors for a given lookback window.

    If lookback <= default (63), slices the last `lookback` timesteps from
    the pre-built tensor. If lookback > 63, rebuilds from the normalized
    features using vectorized operations.

    Args:
        config: Full config dict.
        lookback: Lookback window size.

    Returns:
        Tuple of (X_tensor, sample_dates).
    """
    default_lookback = 63
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]

    X = np.load(processed_path / "X_tensor.npy")
    sample_dates = np.load(processed_path / "sample_dates.npy", allow_pickle=True)

    if lookback == default_lookback:
        return X, sample_dates

    if lookback < default_lookback:
        # Slice the last `lookback` timesteps — correct since sample_dates
        # correspond to the end of each window
        X_sliced = X[:, -lookback:, :]
        logger.info("  Sliced lookback %d -> %d: shape %s",
                     default_lookback, lookback, X_sliced.shape)
        return X_sliced, sample_dates

    # lookback > 63: rebuild from normalized features (vectorized)
    logger.info("  Rebuilding features with lookback=%d (vectorized)...", lookback)
    features_df = pd.read_parquet(processed_path / "features_normalized.parquet")

    tickers = sorted(features_df["Ticker"].unique())
    non_feature_cols = ["Date", "Ticker"]
    feature_cols = [c for c in features_df.columns if c not in non_feature_cols]
    num_features = len(feature_cols)
    num_assets = len(tickers)

    # Pivot each feature to wide format and stack
    dates_all = sorted(features_df["Date"].unique())
    n_dates = len(dates_all)

    # Build a (n_dates, num_assets * num_features) matrix using pivot
    wide_frames = []
    for ticker in tickers:
        ticker_data = features_df[features_df["Ticker"] == ticker].set_index("Date")
        ticker_data = ticker_data.reindex(dates_all)[feature_cols]
        wide_frames.append(ticker_data.values)

    # Stack: (n_dates, num_assets, num_features) -> (n_dates, num_assets * num_features)
    full_matrix = np.concatenate(wide_frames, axis=1).astype(np.float32)

    # Build sliding windows
    X_list = []
    valid_dates = []
    for i in range(lookback, n_dates):
        window = full_matrix[i - lookback:i, :]
        if not np.any(np.isnan(window)):
            X_list.append(window)
            valid_dates.append(dates_all[i])

    X_new = np.array(X_list, dtype=np.float32)
    sample_dates_new = np.array(valid_dates)

    logger.info("  Rebuilt tensor: shape %s, dates %s to %s",
                X_new.shape, sample_dates_new[0], sample_dates_new[-1])

    return X_new, sample_dates_new


def create_datasets_for_hp(
    config: Dict[str, Any],
    X: np.ndarray,
    sample_dates: np.ndarray,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
) -> Tuple[PortfolioDataset, PortfolioDataset]:
    """Create train and validation datasets from pre-loaded data.

    Args:
        config: Config dict.
        X: Feature tensor.
        sample_dates: Corresponding dates.
        train_start, train_end, val_start, val_end: Date range strings.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]

    asset_data = pd.read_parquet(processed_path / "asset_data.parquet")
    returns_wide = asset_data.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn",
    )
    tickers = sorted(returns_wide.columns.tolist())
    returns_wide = returns_wide[tickers]

    sample_dates_idx = pd.DatetimeIndex(sample_dates)
    train_mask = np.array(
        (sample_dates_idx >= pd.Timestamp(train_start)) &
        (sample_dates_idx <= pd.Timestamp(train_end))
    )
    val_mask = np.array(
        (sample_dates_idx >= pd.Timestamp(val_start)) &
        (sample_dates_idx <= pd.Timestamp(val_end))
    )

    forward_days = config["training"].get("forward_days", 5)
    train_ds = PortfolioDataset(X[train_mask], returns_wide, sample_dates[train_mask], tickers, forward_days)
    val_ds = PortfolioDataset(X[val_mask], returns_wide, sample_dates[val_mask], tickers, forward_days)

    return train_ds, val_ds


def run_single_trial(
    trial_num: int,
    hp: Dict[str, Any],
    base_config: Dict[str, Any],
    model_type: str = "tcn",
) -> Dict[str, Any]:
    """Run a single hyperparameter trial.

    Args:
        trial_num: Trial number for logging.
        hp: Sampled hyperparameters.
        base_config: Base config dict.
        model_type: "tcn" or "lstm".

    Returns:
        Dict with trial results.
    """
    config = apply_hyperparameters(base_config, hp)
    splits = config["splits"]
    embargo_days = splits["embargo_days"]

    # Apply embargo to train end
    train_end = pd.Timestamp(splits["initial_train"]["end"])
    effective_train_end = (train_end - pd.tseries.offsets.BDay(embargo_days)).strftime("%Y-%m-%d")

    logger.info("=" * 60)
    logger.info("Trial %d/%d — %s", trial_num, base_config["hyperparameter_search"]["max_configurations"], model_type.upper())
    logger.info("  lr=%.2e, lookback=%d, hidden=%d, dropout=%.1f, "
                "λ_to=%.2f, λ_co=%.2f, temp=%.1f, batch=%d, "
                "loss=%s, γ=%.1f, α_ret=%.1f, α_vol=%.1f",
                hp["learning_rate"], hp["lookback_window"],
                hp["tcn_hidden_channels"], hp["dropout"],
                hp["lambda_turnover"], hp["lambda_concentration"],
                hp["softmax_temperature"], hp["batch_size"],
                hp["loss_type"], hp["gamma_risk_aversion"],
                hp["alpha_return_pred"], hp["alpha_vol_pred"])
    logger.info("=" * 60)

    start_time = time.time()

    try:
        # Set seeds for this trial
        set_seeds(config)

        # Get features for this lookback
        X, sample_dates = rebuild_features_for_lookback(config, hp["lookback_window"])

        # Create datasets
        train_ds, val_ds = create_datasets_for_hp(
            config, X, sample_dates,
            train_start=splits["initial_train"]["start"],
            train_end=effective_train_end,
            val_start=splits["validation"]["start"],
            val_end=splits["validation"]["end"],
        )

        if len(train_ds) < 100 or len(val_ds) < 50:
            logger.warning("  Insufficient data (train=%d, val=%d) — skipping", len(train_ds), len(val_ds))
            return {"trial": trial_num, "hp": hp, "val_sharpe": -999, "status": "skipped"}

        # Build model
        num_assets = 30
        input_dim = X.shape[-1]

        if model_type == "tcn":
            model = build_tcn_model(config, num_assets, input_dim)
        else:
            model = build_lstm_model(config, num_assets, input_dim)

        # Build loss function based on config
        loss_type = config["loss"].get("type", "sharpe")
        if loss_type == "mv_utility":
            base_loss = MeanVarianceUtilityLoss(config)
        elif loss_type == "return_max":
            base_loss = ReturnMaxLoss(config)
        else:
            base_loss = SharpeRatioLoss(config)
        loss_fn = MultiTaskLoss(base_loss, config)

        logger.info("  Model params: %d, train samples: %d, val samples: %d, loss=%s",
                     model.count_parameters(), len(train_ds), len(val_ds), loss_type)

        # Train with reduced epochs for search
        search_config = copy.deepcopy(config)
        search_config["training"]["max_epochs"] = 100
        search_config["training"]["early_stopping_patience"] = 15

        result = train_model(model, train_ds, val_ds, loss_fn, search_config)

        elapsed = time.time() - start_time

        # Extract validation metrics
        val_sharpe = result["best_val_sharpe"]

        # Check weight behavior on validation set
        model.eval()
        val_loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False)
        with torch.no_grad():
            X_val, _ = next(iter(val_loader))
            weights = model(X_val)
            weight_entropy = -(weights * torch.log(weights + 1e-10)).sum(dim=-1).mean().item()
            max_entropy = np.log(num_assets)
            entropy_ratio = weight_entropy / max_entropy
            mean_max_weight = weights.max(dim=-1).values.mean().item()

        logger.info("  Trial %d result: val_sharpe=%.4f, entropy=%.3f, "
                     "max_weight=%.3f, epochs=%d, time=%.0fs",
                     trial_num, val_sharpe, entropy_ratio,
                     mean_max_weight, result["best_epoch"], elapsed)

        return {
            "trial": trial_num,
            "hp": hp,
            "val_sharpe": val_sharpe,
            "best_epoch": result["best_epoch"],
            "entropy_ratio": entropy_ratio,
            "mean_max_weight": mean_max_weight,
            "elapsed_seconds": elapsed,
            "model_params": model.count_parameters(),
            "train_history": result["train_history"],
            "val_history": result["val_history"],
            "status": "completed",
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("  Trial %d FAILED: %s (%.0fs)", trial_num, str(e), elapsed)
        return {
            "trial": trial_num,
            "hp": hp,
            "val_sharpe": -999,
            "elapsed_seconds": elapsed,
            "status": f"error: {str(e)}",
        }


def compute_equal_weight_val_sharpe(config: Dict[str, Any]) -> float:
    """Compute the equal-weight baseline Sharpe on the validation period.

    Args:
        config: Full config dict.

    Returns:
        Equal-weight Sharpe ratio on 2016-2017.
    """
    returns_wide, open_prices_wide = load_backtest_data(config)

    result = run_backtest(
        weight_func=equal_weight_func,
        returns_wide=returns_wide,
        open_prices_wide=open_prices_wide,
        config=config,
        start_date=config["splits"]["validation"]["start"],
        end_date=config["splits"]["validation"]["end"],
        name="Equal Weight (Validation)",
    )

    return result["metrics"]["Sharpe Ratio"]


def run_hyperparameter_search(
    config: Dict[str, Any],
    model_type: str = "tcn",
) -> Dict[str, Any]:
    """Run the full hyperparameter search.

    Args:
        config: Full config dict.
        model_type: "tcn" or "lstm".

    Returns:
        Dict with all trial results and best configuration.
    """
    max_configs = config["hyperparameter_search"]["max_configurations"]
    rng = np.random.RandomState(config["seeds"]["numpy"])

    logger.info("=" * 70)
    logger.info("PHASE 6 — HYPERPARAMETER SEARCH (%s)", model_type.upper())
    logger.info("  Max configurations: %d", max_configs)
    logger.info("  Method: random search")
    logger.info("  Validation period: %s to %s",
                config["splits"]["validation"]["start"],
                config["splits"]["validation"]["end"])
    logger.info("=" * 70)

    # Compute equal-weight baseline on validation period
    ew_val_sharpe = compute_equal_weight_val_sharpe(config)
    logger.info("Equal Weight validation Sharpe: %.4f", ew_val_sharpe)

    # Generate all configurations upfront for reproducibility
    all_hp = [sample_hyperparameters(config, rng) for _ in range(max_configs)]

    # Run trials
    results = []
    total_start = time.time()

    for i, hp in enumerate(all_hp):
        trial_result = run_single_trial(i + 1, hp, config, model_type)
        results.append(trial_result)

        # Log running best
        completed = [r for r in results if r["status"] == "completed"]
        if completed:
            best_so_far = max(completed, key=lambda r: r["val_sharpe"])
            logger.info("  >> Best so far: Trial %d with val_sharpe=%.4f",
                         best_so_far["trial"], best_so_far["val_sharpe"])

    total_time = time.time() - total_start

    # Find best
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        logger.error("No trials completed successfully!")
        return {"results": results, "best": None, "ew_val_sharpe": ew_val_sharpe}

    best = max(completed, key=lambda r: r["val_sharpe"])

    logger.info("=" * 70)
    logger.info("HYPERPARAMETER SEARCH COMPLETE — %s", model_type.upper())
    logger.info("  Total time: %.0f seconds (%.1f minutes)", total_time, total_time / 60)
    logger.info("  Completed trials: %d/%d", len(completed), max_configs)
    logger.info("  Best trial: %d", best["trial"])
    logger.info("  Best val Sharpe: %.4f", best["val_sharpe"])
    logger.info("  Equal Weight val Sharpe: %.4f", ew_val_sharpe)
    logger.info("  Best hyperparameters:")
    for k, v in best["hp"].items():
        if isinstance(v, float):
            logger.info("    %s: %.6f", k, v)
        else:
            logger.info("    %s: %s", k, v)
    logger.info("=" * 70)

    return {
        "model_type": model_type,
        "results": results,
        "best": best,
        "ew_val_sharpe": ew_val_sharpe,
        "total_time": total_time,
        "forward_days": config["training"].get("forward_days", 21),
    }


def run_gate5_check(
    search_results: Dict[str, Any],
) -> bool:
    """Run Gate 5 checks.

    Gate 5: Best validation Sharpe is positive and exceeds equal-weight.

    Args:
        search_results: Output from run_hyperparameter_search.

    Returns:
        True if Gate 5 passes.
    """
    best = search_results["best"]
    ew_sharpe = search_results["ew_val_sharpe"]
    model_type = search_results["model_type"]

    logger.info("=" * 70)
    logger.info("GATE 5 CHECKS — %s", model_type.upper())
    logger.info("=" * 70)

    if best is None:
        logger.error("  [FAIL] No completed trials")
        return False

    val_sharpe_raw = best["val_sharpe"]
    # Annualize: returns are computed over forward_days trading days,
    # so there are 252/forward_days periods per year.
    import math
    forward_days = search_results.get("forward_days", 21)
    periods_per_year = 252 / forward_days
    val_sharpe_annual = val_sharpe_raw * math.sqrt(periods_per_year)
    checks = []

    # Check 1: Validation Sharpe is positive
    check1 = val_sharpe_raw > 0
    checks.append(("Validation Sharpe > 0", check1,
                    f"val_sharpe={val_sharpe_raw:.4f} (annualized: {val_sharpe_annual:.4f})"))

    # Check 2: Exceeds equal-weight (both annualized)
    check2 = val_sharpe_annual > ew_sharpe
    checks.append(("Annualized Sharpe > Equal Weight", check2,
                    f"model={val_sharpe_annual:.4f} vs ew={ew_sharpe:.4f}"))

    # Check 3: Weight entropy is reasonable (not fully collapsed to 1 asset)
    # Allow concentrated bets — entropy > 0.3 is fine (>30% of max entropy)
    check3 = best.get("entropy_ratio", 0) > 0.3
    checks.append(("Weight entropy > 0.3*ln(N)", check3,
                    f"entropy_ratio={best.get('entropy_ratio', 0):.3f}"))

    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        logger.info("  [%s] %s — %s", status, name, detail)
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("GATE 5: PASS")
    else:
        logger.warning("GATE 5: FAIL — see above for failing checks")
        logger.info("If val Sharpe does not exceed equal-weight, this may indicate:")
        logger.info("  1. The model needs architectural changes")
        logger.info("  2. Feature engineering may need improvement")
        logger.info("  3. The validation period (2016-2017) may be particularly favorable for equal-weight")
        logger.info("Proceeding with best available configuration for Phase 7 evaluation.")

    return all_passed


def save_search_results(
    search_results: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Save search results to disk for reference.

    Args:
        search_results: Output from run_hyperparameter_search.
        config: Full config dict.
    """
    tables_path = PROJECT_ROOT / config["paths"]["tables"]
    tables_path.mkdir(parents=True, exist_ok=True)

    model_type = search_results["model_type"]

    # Save trial summary as CSV
    rows = []
    for r in search_results["results"]:
        row = {
            "trial": r["trial"],
            "status": r["status"],
            "val_sharpe": r["val_sharpe"],
        }
        row.update(r["hp"])
        if "best_epoch" in r:
            row["best_epoch"] = r["best_epoch"]
            row["entropy_ratio"] = r.get("entropy_ratio", "")
            row["mean_max_weight"] = r.get("mean_max_weight", "")
            row["elapsed_seconds"] = r.get("elapsed_seconds", "")
            row["model_params"] = r.get("model_params", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = tables_path / f"hp_search_{model_type}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved trial summary to %s", csv_path)

    # Save best config as JSON
    if search_results["best"] is not None:
        best_hp = search_results["best"]["hp"]
        import math
        raw_sharpe = search_results["best"]["val_sharpe"]
        forward_days = search_results.get("forward_days", 21)
        periods_per_year = 252 / forward_days
        best_info = {
            "model_type": model_type,
            "best_trial": search_results["best"]["trial"],
            "best_val_sharpe_raw": raw_sharpe,
            "best_val_sharpe_annualized": raw_sharpe * math.sqrt(periods_per_year),
            "forward_days": forward_days,
            "ew_val_sharpe_annualized": search_results["ew_val_sharpe"],
            "hyperparameters": best_hp,
        }
        json_path = tables_path / f"best_hp_{model_type}.json"
        with open(json_path, "w") as f:
            json.dump(best_info, f, indent=2)
        logger.info("Saved best config to %s", json_path)


if __name__ == "__main__":
    config = load_config()
    set_seeds(config)
    ensure_directories(config)

    logger.info("=" * 70)
    logger.info("PHASE 6 — HYPERPARAMETER TUNING")
    logger.info("=" * 70)

    all_passed = True

    # Run search for TCN
    tcn_results = run_hyperparameter_search(config, model_type="tcn")
    save_search_results(tcn_results, config)
    tcn_gate5 = run_gate5_check(tcn_results)

    # Run search for LSTM
    lstm_results = run_hyperparameter_search(config, model_type="lstm")
    save_search_results(lstm_results, config)
    lstm_gate5 = run_gate5_check(lstm_results)

    all_passed = tcn_gate5 or lstm_gate5  # At least one model must pass

    logger.info("=" * 70)
    if all_passed:
        logger.info("GATE 5: PASS — At least one model's validation Sharpe is positive and exceeds equal-weight")
    else:
        logger.warning("GATE 5: FAIL — Neither model exceeded equal-weight on validation")
        logger.info("Corrective actions:")
        logger.info("  1. Revisit feature engineering (add/remove features)")
        logger.info("  2. Adjust model architecture (layers, attention)")
        logger.info("  3. Re-examine loss function balance (entropy, turnover penalties)")
    logger.info("=" * 70)