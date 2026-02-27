"""
utils.py — Core utilities for the Tactical Asset Allocation system.
Phase 1: Infrastructure. Provides config loading and reproducibility setup.
"""

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load the master config.yaml file.

    Args:
        config_path: Optional override path. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        Dictionary of all configuration parameters.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seeds(config: Dict[str, Any] = None) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python, NumPy, and PyTorch. Enables deterministic
    algorithms in PyTorch. Must be called at the top of every script.

    Args:
        config: Config dict. If None, loads from default config.yaml.
    """
    if config is None:
        config = load_config()

    seed_cfg = config["seeds"]
    python_seed = seed_cfg["python"]
    numpy_seed = seed_cfg["numpy"]
    torch_seed = seed_cfg["torch"]

    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def ensure_directories(config: Dict[str, Any] = None) -> None:
    """Create all required output directories if they don't exist.

    Args:
        config: Config dict. If None, loads from default config.yaml.
    """
    if config is None:
        config = load_config()

    for key in ["raw_data", "processed_data", "figures", "tables", "checkpoints"]:
        path = PROJECT_ROOT / config["paths"][key]
        path.mkdir(parents=True, exist_ok=True)
