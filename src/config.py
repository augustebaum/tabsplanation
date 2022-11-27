"""This module contains the general configuration of the project."""
from __future__ import annotations

from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parent.resolve()
BLD = (ROOT / "bld").resolve()

BLD_DATA = BLD / "data"
BLD_PLOT_DATA = BLD / "plot_data"
BLD_MODELS = BLD / "models"
BLD_PLOTS = BLD / "plots"


def get_configs() -> List[DictConfig]:
    """Read experiment configuration files."""
    cfg_path = BLD / "config.yaml"
    cfgs = [OmegaConf.load(cfg_path)]
    return cfgs


__all__ = ["BLD", "SRC", "get_configs"]
