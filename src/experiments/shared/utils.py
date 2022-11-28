import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypeAlias

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch

from omegaconf import DictConfig, OmegaConf

from config import ROOT, SRC


def setup(seed: int) -> Tuple[torch.device, Path, Path]:
    """Set up an experiment.

    Set the seed, the plot style and get the pytorch device.
    """
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_matplotlib_style()
    return device


def get_map_img():
    return mpimg.imread(SRC / "experiments" / "shared" / "data_map.png")


def set_matplotlib_style():
    latex_preamble = r"""
\usepackage{libertine}
"""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": latex_preamble,
            "font.size": 12,
            "mathtext.fontset": "stix",
        }
    )


def get_time() -> str:
    return datetime.now().isoformat()


def hash_(cfg: DictConfig):
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_str = json.dumps(cfg_dict, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(cfg_str.encode("ascii")).hexdigest()


ExperimentName: TypeAlias = Literal[
    "classification", "latent_shift", "ae_reconstruction"
]


def get_configs(experiment_name: Optional[ExperimentName] = None) -> List[DictConfig]:
    """Read experiment configuration files."""
    cfgs_dir = ROOT / "experiment_configs"
    if experiment_name is None:
        cfg_names = os.listdir(cfgs_dir)
    else:
        cfg_names = [f"{experiment_name}.yaml"]
    cfgs = [OmegaConf.load(cfgs_dir / name) for name in cfg_names]
    return cfgs
