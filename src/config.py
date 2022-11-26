"""This module contains the general configuration of the project."""
from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parent.resolve()
BLD = (ROOT / "bld").resolve()

BLD_DATA = BLD / "data"
BLD_PLOT_DATA = BLD / "plot_data"
BLD_MODELS = BLD / "models"
BLD_PLOTS = BLD / "plots"

__all__ = ["BLD", "SRC"]
