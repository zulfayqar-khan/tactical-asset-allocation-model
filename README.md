# Tactical Asset Allocation Model

A deep learning framework for tactical asset allocation across 30 DJIA stocks, built with strict time-series evaluation rules and direct benchmark comparison.

## Highlights

- End to end pipeline from data download to final walk-forward evaluation
- TCN plus attention and LSTM portfolio models
- Five benchmark strategies: equal weight, mean variance, momentum, buy and hold, risk parity
- Transaction costs integrated in training and backtesting
- Walk-forward protocol with embargo between train and test windows
- Reproducible configuration driven workflow via `config.yaml`

## Repository Structure

```text
.
|-- backtesting/
|   |-- engine.py
|   |-- metrics.py
|-- baselines/
|   |-- run_baselines.py
|-- data/
|   |-- download.py
|-- features/
|   |-- engineering.py
|   |-- validation.py
|-- models/
|   |-- tcn_attention.py
|   |-- lstm_allocator.py
|   |-- losses.py
|-- training/
|   |-- trainer.py
|   |-- hyperparameter_search.py
|   |-- walk_forward.py
|   |-- phase7_diagnostics.py
|-- results/
|   |-- figures/
|   |-- tables/
|-- config.yaml
|-- requirements.txt
```

## Methodology

1. Download and clean market data
2. Build multi-group feature tensors
3. Validate feature pipeline for leakage safety
4. Run benchmark strategies
5. Tune hyperparameters on validation period
6. Run walk-forward test folds (2018-2024)
7. Generate diagnostics and summary tables

## Reported Out of Sample Results (2018-2024)

| Strategy | Sharpe | Ann. Return | Max Drawdown |
| :-- | --: | --: | --: |
| Equal Weight | 0.883 | 17.0% | 32.5% |
| Mean Variance | 1.133 | 24.0% | 30.7% |
| Momentum | 0.882 | 16.9% | 31.5% |
| Buy and Hold | 1.013 | 20.6% | 31.5% |
| Risk Parity | 0.857 | 15.4% | 31.7% |
| TCN plus Attention | 1.169 | 33.7% | 44.3% |

Notes:
- The TCN model leads on Sharpe and annual return in the full period.
- Max drawdown is above the configured acceptance threshold, documented in `results/tables/phase7_tcn_summary.json`.

## Setup

### 1) Create environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run Pipeline

Run each phase directly from repository root.

```bash
python data/download.py
python features/engineering.py
python features/validation.py
python baselines/run_baselines.py
python training/hyperparameter_search.py
python training/walk_forward.py
python training/phase7_diagnostics.py
```

## Key Outputs

- `results/tables/baselines_Full_Test_2018_2024.csv`
- `results/tables/best_hp_tcn.json`
- `results/tables/phase7_tcn_full_test.csv`
- `results/tables/phase7_tcn_summary.json`
- `results/figures/chart1_cumulative_returns.png`
- `results/figures/chart2_drawdown.png`
- `results/figures/chart3_weight_allocation.png`

## Reproducibility

- Global seeds are set in `config.yaml`
- Deterministic PyTorch mode is enabled in utilities
- Paths and thresholds are centralized in config

## Limitations

- Data source depends on Yahoo Finance availability
- Results can vary slightly by platform and BLAS backend

## Author

Zulfiqar Khan
UFCEKP-30-3 Final Project
UWE Bristol
