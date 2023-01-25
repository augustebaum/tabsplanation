import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypeAlias

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from config import ROOT, SRC
from tabsplanation.data import CakeOnSeaDataModule, SyntheticDataset


def load_mpl_style():
    plt.style.use(Path(__file__).parent.resolve() / "default.mplstyle")


def setup(seed: Optional[int] = None) -> Tuple[torch.device, Path, Path]:
    """Set up an experiment.

    Set the seed, the plot style and get the pytorch device.
    """
    load_mpl_style()
    if seed is not None:
        pl.seed_everything(seed, workers=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def get_map_img():
    return mpimg.imread(SRC / "experiments" / "shared" / "data_map.png")


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
        cfg_names = [file for file in os.listdir(cfgs_dir) if file.endswith(".yaml")]
    else:
        cfg_names = [f"{experiment_name}.yaml"]
    cfgs = [OmegaConf.load(cfgs_dir / name) for name in cfg_names]
    return cfgs


def save_full_config(cfg: DictConfig, path: Path) -> None:
    OmegaConf.save(cfg, path, resolve=True)


def save_config(cfg: DictConfig, path: Path) -> None:
    OmegaConf.save(cfg, path, resolve=False)


def get_module_object(module_path: str, object_name: str):
    """Import module given by `module_path` and return a function
    or class defined in that module with the name `object_name`."""
    exec(f"import {module_path}")
    return getattr(sys.modules[module_path], object_name)


# This is to generate task functions dynamically from a task class
# But this doesn't work yet: <https://github.com/pytask-dev/pytask/issues/324>
def camel_to_snake(str):
    """
    <https://www.geeksforgeeks.org/python-program-to-convert-camel-case-string-to-snake-case/>
    No shame!
    """

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def define_task(cfg_name, _task_class):
    """Prepare a definition of a task, which will be `exec`ed in the
    task file.

    The task definition must be run in the task file; `pytask` won't
    collect it if it's run in this file.

    I don't like this but I don't know how to do it otherwise...
    """

    cfgs = get_configs(cfg_name)
    for cfg in cfgs:
        task = _task_class(cfg)

        task_name = camel_to_snake(_task_class.__name__)
        task_class_name = _task_class.__name__

        task_definition = f"""
import pytask
from experiments.shared.utils import save_config, save_full_config

@pytask.mark.task(id=task.id_)
@pytask.mark.depends_on(task.depends_on)
@pytask.mark.produces(task.produces)
def {task_name}(depends_on, produces, cfg=task.cfg):
    {task_class_name}.task_function(depends_on, produces, cfg)
    save_full_config(cfg, produces["full_config"])
    save_config(cfg, produces["config"])
"""
        return task, task_definition


# TODO: Extract output_dir from name of subclass
class Task:
    def __init__(self, cfg: OmegaConf, output_dir: Path):
        self.cfg = cfg
        self.id_ = hash_(self.cfg)

        self.produces_dir = output_dir / self.id_
        self.produces = {
            "config": self.produces_dir / "config.yaml",
            "full_config": self.produces_dir / "full_config.yaml",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        raise NotImplementedError()


def get_data_module(depends_on, cfg, device):
    """Load a dataset and instantiate the corresponding `DataModule`."""

    dataset = SyntheticDataset(
        depends_on["xs"],
        depends_on["ys"],
        depends_on["coefs"],
        cfg.data.nb_dims,
        device,
    )
    data_module_kwargs = {"dataset": dataset} | OmegaConf.to_object(cfg.data_module)
    data_module = CakeOnSeaDataModule(**data_module_kwargs)
    return data_module

def write(obj, file_path: Path) -> None:
    
