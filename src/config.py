"""This module contains the general configuration of the project."""
from __future__ import annotations

from pathlib import Path

# Use `resolve` to deal with symlinks and such
SRC = Path(__file__).parent.resolve()
ROOT = SRC.parent
BLD = ROOT / "bld"

BLD_DATA = BLD / "data"
BLD_PLOT_DATA = BLD / "plot_data"
BLD_MODELS = BLD / "models"
BLD_PLOTS = BLD / "plots"

EXPERIMENT_CONFIGS = ROOT / "experiment_configs"
EXPERIMENTS_PATH = SRC / "experiments"

__all__ = ["BLD", "SRC"]
