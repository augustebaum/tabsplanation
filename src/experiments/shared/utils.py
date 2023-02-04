import hashlib
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypeAlias

import lightning as pl

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from config import ROOT, SRC

from tabsplanation.data import CakeOnSeaDataModule, CakeOnSeaDataset


def load_mpl_style():
    plt.style.use(Path(__file__).parent.resolve() / "default.mplstyle")


def setup(seed: Optional[int] = None) -> torch.device:
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


def get_module_object(module_path: str, object_name: str):
    """Import module given by `module_path` and return a function
    or class defined in that module with the name `object_name`."""
    exec(f"import {module_path}")
    return getattr(sys.modules[module_path], object_name)


# --- Task boilerplate

ExperimentName: TypeAlias = str


def get_configs(experiment_name: Optional[ExperimentName] = None) -> List[DictConfig]:
    """Read experiment configuration files."""
    cfgs_dir = ROOT / "experiment_configs"
    if experiment_name is None:
        cfg_names = [file for file in os.listdir(cfgs_dir) if file.endswith(".yaml")]
    else:
        cfg_names = [f"{experiment_name}.yaml"]

    cfgs = []
    for name in cfg_names:
        cfg = cfgs_dir / name
        try:
            cfgs.append(OmegaConf.load(cfg))
        except FileNotFoundError:
            warnings.warn(f"{cfg} not found; skipping.")

    return cfgs


def save_full_config(cfg: DictConfig, path: Path) -> None:
    OmegaConf.save(cfg, path, resolve=True)


def save_config(cfg: DictConfig, path: Path) -> None:
    OmegaConf.save(cfg, path, resolve=False)


# This is to generate task functions dynamically from a task class
# But this doesn't work yet: <https://github.com/pytask-dev/pytask/issues/324>
def camel_to_snake(str):
    """
    <https://www.geeksforgeeks.org/python-program-to-convert-camel-case-string-to-snake-case/>
    No shame!
    """

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def get_module(o):
    klass = o.__class__
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__  # avoid outputs like 'builtins.str'
    return module


# TODO: Extract output_dir from name of subclass
class Task:
    def __init__(self, cfg: OmegaConf, output_dir: Path):
        self.task_deps = []
        self.depends_on = {}
        self.produces = {}
        self.cfg = {}
        self.id_ = "0"

        if cfg is not None:
            OmegaConf.resolve(cfg)
            # Clone config
            self.cfg = OmegaConf.create(OmegaConf.to_container(cfg))
            self.id_ = hash_(self.cfg)

            self.produces_dir = output_dir / self.id_
            self.produces = {
                "config": self.produces_dir / "config.yaml",
                "full_config": self.produces_dir / "full_config.yaml",
            }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        raise NotImplementedError()

    def _define_task(self, imports=False, module_import=True):
        cls_name = self.__class__.__name__
        task = self

        if imports:
            imports = """
from omegaconf import OmegaConf
import pytask
from experiments.shared.utils import save_config, save_full_config
"""
        else:
            imports = ""

        if module_import:
            module_import = f"""
from {get_module(self)} import {cls_name}
"""
        else:
            module_import = ""

        task_definition = f"""
task = {cls_name}(OmegaConf.create({self.cfg}))
@pytask.mark.task(id=task.id_)
@pytask.mark.depends_on(task.depends_on)
@pytask.mark.produces(task.produces)
def {camel_to_snake(cls_name)}(depends_on, produces, cfg=task.cfg):
    {cls_name}.task_function(depends_on, produces, cfg)
    if isinstance(produces, dict):
        if produces.get("full_config") is not None:
            save_full_config(cfg, produces["full_config"])
        if produces.get("config") is not None:
            save_config(cfg, produces["config"])

"""

        task_definition = imports + module_import + task_definition

        return task, task_definition

    def define_task(self, result=""):
        _, task_def = self._define_task(
            imports=(result == ""), module_import=(result != "")
        )
        result += task_def
        if self.task_deps == []:
            return None, result
        else:
            for task_dep in self.task_deps:
                _, result = task_dep.define_task(result)
            return None, result

    def all_task_deps(self, result=[]) -> List:
        for task in self.task_deps:
            result.append(task)
            result = task.all_task_deps(result=result)
        return result


# ---


def get_data_module(depends_on, cfg, device):
    """Load a dataset and instantiate the corresponding `DataModule`."""

    dataset = CakeOnSeaDataset(
        depends_on["xs"],
        depends_on["ys"],
        depends_on["coefs"],
        cfg.data.nb_dims,
        device,
    )
    data_module_kwargs = {"dataset": dataset} | OmegaConf.to_object(cfg.data_module)
    data_module = CakeOnSeaDataModule(**data_module_kwargs)
    return data_module


# --- More task boilerplate: Read/write


def write_pkl(obj, file_path):
    import pickle

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def write_svg(obj, file_path):
    obj.savefig(file_path)


def write(obj, file_path: Path) -> None:
    write_variants = {".pkl": write_pkl, ".svg": write_svg}

    write_fn = write_variants.get(file_path.suffix)
    if write_fn is None:
        raise NotImplementedError(
            f"No write function implemented for extension '{file_path.suffix}' yet."
        )
    write_fn(obj, file_path)


def read_pkl(file_path):
    import pickle

    with open(file_path, "rb") as f:
        result = pickle.load(f)
    return result


def read_pt(file_path, device):
    model = torch.load(file_path).to(device)
    model.eval()
    return model


def read(file_path: Path, *args) -> object:
    read_variants = {".pkl": read_pkl, ".pt": read_pt}

    read_fn = read_variants.get(file_path.suffix)
    if read_fn is None:
        raise NotImplementedError(
            f"No read function implemented for extension '{file_path.suffix}' yet."
        )
    return read_fn(file_path, *args)
