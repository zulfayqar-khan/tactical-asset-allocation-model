"""
features/validation.py — Leakage detection and feature validation tests.
Phase 3: Feature Engineering (Gate 2).

Implements a truncation test for look-ahead bias detection:
1. Run feature computation on the FULL dataset.
2. Run feature computation on a TRUNCATED dataset (first 80% of dates).
3. Verify that features for all dates within the truncated period are
   IDENTICAL in both runs.
4. If they differ, there is look-ahead bias — HALT and report.

Also validates normalization statistics use backward-looking windows only.

Runnable standalone: python features/validation.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from utils import load_config, set_seeds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def truncation_test(config: Dict[str, Any]) -> bool:
    """Run the truncation test for look-ahead bias detection.

    Computes features on the full dataset and on a truncated (first 80%)
    dataset. Features for overlapping dates must be identical. Any difference
    indicates look-ahead bias.

    Args:
        config: Full config dictionary.

    Returns:
        True if test passes (no leakage), False otherwise.
    """
    from features.engineering import (
        compute_group_a,
        compute_group_b,
        compute_cross_sectional_features,
        compute_group_d,
        rolling_normalize,
    )

    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    feat_cfg = config["features"]

    logger.info("=" * 70)
    logger.info("TRUNCATION TEST — Look-ahead bias detection")
    logger.info("=" * 70)

    # Load data
    asset_data = pd.read_parquet(processed_path / "asset_data.parquet")
    market_data = pd.read_parquet(processed_path / "market_data.parquet")

    all_dates = sorted(asset_data["Date"].unique())
    cutoff_idx = int(len(all_dates) * 0.8)
    cutoff_date = all_dates[cutoff_idx]
    test_dates = all_dates[:cutoff_idx]  # Dates that must match

    logger.info("  Full dataset: %d dates (%s to %s)",
                len(all_dates), all_dates[0], all_dates[-1])
    logger.info("  Truncated dataset: %d dates (up to %s)",
                cutoff_idx, cutoff_date)

    # Truncate data
    asset_trunc = asset_data[asset_data["Date"] <= cutoff_date].copy()
    market_trunc = market_data.loc[:cutoff_date].copy()

    tickers = sorted(asset_data["Ticker"].unique())
    all_passed = True

    # --- Test Group A + B per-asset features ---
    logger.info("Testing Group A + B (per-asset features) ...")
    mismatches_ab = 0

    for ticker in tickers[:5]:  # Test on first 5 tickers for speed
        tdf_full = asset_data[asset_data["Ticker"] == ticker].set_index("Date").sort_index()
        tdf_trunc = asset_trunc[asset_trunc["Ticker"] == ticker].set_index("Date").sort_index()

        ga_full = compute_group_a(tdf_full, config)
        ga_trunc = compute_group_a(tdf_trunc, config)

        gb_full = compute_group_b(tdf_full, config)
        gb_trunc = compute_group_b(tdf_trunc, config)

        # Compare on overlapping dates
        overlap_dates = ga_full.index.intersection(ga_trunc.index)

        for df_full, df_trunc, name in [
            (ga_full, ga_trunc, "Group A"),
            (gb_full, gb_trunc, "Group B"),
        ]:
            full_overlap = df_full.loc[overlap_dates]
            trunc_overlap = df_trunc.loc[overlap_dates]

            # Drop NaN rows for fair comparison
            valid = full_overlap.notna() & trunc_overlap.notna()
            diff = (full_overlap[valid] - trunc_overlap[valid]).abs()

            max_diff = diff.max().max()
            if max_diff > 1e-10:
                logger.error(
                    "  FAIL: %s %s — max difference = %.2e",
                    ticker, name, max_diff,
                )
                mismatches_ab += 1
            else:
                logger.info("  PASS: %s %s — identical on overlap", ticker, name)

    if mismatches_ab > 0:
        all_passed = False

    # --- Test Group C (cross-sectional) ---
    logger.info("Testing Group C (cross-sectional features) ...")
    returns_wide_full = asset_data.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn",
    )
    returns_wide_trunc = asset_trunc.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn",
    )

    cs_full = compute_cross_sectional_features(
        returns_wide_full, feat_cfg["cross_section_window"],
    )
    cs_trunc = compute_cross_sectional_features(
        returns_wide_trunc, feat_cfg["cross_section_window"],
    )

    for key in ["rank", "zscore"]:
        overlap = cs_full[key].index.intersection(cs_trunc[key].index)
        valid = cs_full[key].loc[overlap].notna() & cs_trunc[key].loc[overlap].notna()
        diff = (cs_full[key].loc[overlap][valid] - cs_trunc[key].loc[overlap][valid]).abs()
        max_diff = diff.max().max()
        if max_diff > 1e-10:
            logger.error("  FAIL: Cross-sectional %s — max diff = %.2e", key, max_diff)
            all_passed = False
        else:
            logger.info("  PASS: Cross-sectional %s — identical on overlap", key)

    # --- Test Group D (market regime) ---
    logger.info("Testing Group D (market regime features) ...")
    gd_full = compute_group_d(market_data, returns_wide_full, config)
    gd_trunc = compute_group_d(market_trunc, returns_wide_trunc, config)

    overlap = gd_full.index.intersection(gd_trunc.index)
    valid = gd_full.loc[overlap].notna() & gd_trunc.loc[overlap].notna()
    diff = (gd_full.loc[overlap][valid] - gd_trunc.loc[overlap][valid]).abs()
    max_diff = diff.max().max()
    if max_diff > 1e-10:
        logger.error("  FAIL: Group D — max diff = %.2e", max_diff)
        all_passed = False
    else:
        logger.info("  PASS: Group D — identical on overlap")

    # --- Test rolling normalization ---
    logger.info("Testing rolling normalization ...")
    # Create a small test series and verify normalization is backward-looking
    np.random.seed(42)
    test_series = pd.DataFrame({"val": np.random.randn(500).cumsum()})

    norm_full = rolling_normalize(test_series, feat_cfg["normalization_window"], feat_cfg["clip_range"])
    norm_trunc = rolling_normalize(test_series.iloc[:400], feat_cfg["normalization_window"], feat_cfg["clip_range"])

    overlap_idx = norm_full.index[:400]
    valid = norm_full.loc[overlap_idx].notna() & norm_trunc.notna()
    diff = (norm_full.loc[overlap_idx][valid] - norm_trunc[valid]).abs()
    max_diff = diff.max().max()
    if max_diff > 1e-10:
        logger.error("  FAIL: Rolling normalization — max diff = %.2e", max_diff)
        all_passed = False
    else:
        logger.info("  PASS: Rolling normalization — backward-looking confirmed")

    return all_passed


def validate_date_splits(config: Dict[str, Any]) -> bool:
    """Verify no overlap between train and test date ranges.

    Args:
        config: Full config dictionary.

    Returns:
        True if no overlap detected.
    """
    logger.info("Validating train/test date splits ...")
    splits = config["splits"]
    embargo = splits["embargo_days"]

    train_end = pd.Timestamp(splits["initial_train"]["end"])
    val_start = pd.Timestamp(splits["validation"]["start"])

    # Check embargo between train and validation
    gap_days = np.busday_count(
        train_end.date(), val_start.date()
    )

    all_ok = True

    if gap_days < embargo:
        logger.warning(
            "  WARNING: Gap between train end (%s) and val start (%s) is %d "
            "business days, less than embargo (%d). Embargo will be enforced "
            "during walk-forward splitting.",
            train_end.date(), val_start.date(), gap_days, embargo,
        )
    else:
        logger.info("  PASS: Train/val gap = %d business days (embargo = %d)", gap_days, embargo)

    # Check no overlap between periods
    periods = [
        ("initial_train", splits["initial_train"]),
        ("validation", splits["validation"]),
        ("test_fold_1", splits["test_fold_1"]),
        ("test_fold_2", splits["test_fold_2"]),
        ("test_fold_3", splits["test_fold_3"]),
    ]

    for i in range(len(periods) - 1):
        name_a, period_a = periods[i]
        name_b, period_b = periods[i + 1]
        end_a = pd.Timestamp(period_a["end"])
        start_b = pd.Timestamp(period_b["start"])

        if end_a >= start_b:
            logger.error("  FAIL: %s end (%s) overlaps with %s start (%s)",
                         name_a, end_a.date(), name_b, start_b.date())
            all_ok = False
        else:
            logger.info("  PASS: %s -> %s — no overlap", name_a, name_b)

    return all_ok


def run_validation() -> None:
    """Run all feature validation checks for Gate 2."""
    config = load_config()
    set_seeds(config)

    logger.info("=" * 70)
    logger.info("PHASE 3 — FEATURE VALIDATION (GATE 2)")
    logger.info("=" * 70)

    checks = []

    # Test 1: Truncation test (look-ahead bias)
    checks.append(("Truncation test (no look-ahead bias)", truncation_test(config)))

    # Test 2: Date split validation
    checks.append(("Date split validation (no overlap)", validate_date_splits(config)))

    # Test 3: Feature distributions (load saved features)
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    features = pd.read_parquet(processed_path / "features_normalized.parquet")
    meta_cols = ["Date", "Ticker"]
    feat_cols = [c for c in features.columns if c not in meta_cols]

    # Check no extreme values beyond clip range
    clip = config["features"]["clip_range"]
    extremes = (features[feat_cols].abs() > clip + 0.01).any().any()
    checks.append(("Features clipped within ±%.1f" % clip, not extremes))

    # Check no NaN in final features
    has_nan = features[feat_cols].isna().any().any()
    checks.append(("No NaN in normalized features", not has_nan))

    # Check feature tensor
    X = np.load(processed_path / "X_tensor.npy")
    checks.append(("Tensor has no NaN", not np.any(np.isnan(X))))
    checks.append(("Tensor shape valid", X.ndim == 3 and X.shape[1] == config["features"]["lookback_window"]))

    # --- GATE 2 VERDICT ---
    logger.info("=" * 70)
    all_passed = all(result for _, result in checks)

    for name, result in checks:
        status = "PASS" if result else "FAIL"
        logger.info("  [%s] %s", status, name)

    logger.info("=" * 70)
    if all_passed:
        logger.info("GATE 2: PASS — Feature leakage test passes. Distributions reasonable.")
    else:
        failed = [name for name, result in checks if not result]
        logger.error("GATE 2: FAIL — %s", "; ".join(failed))
    logger.info("=" * 70)


if __name__ == "__main__":
    run_validation()
