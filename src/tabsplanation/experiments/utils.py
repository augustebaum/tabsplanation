from datetime import datetime
from os.path import dirname
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import tomli
import torch
from hydra.core.hydra_config import HydraConfig

from .style import set_matplotlib_style


def file_to_string(filename: Path) -> str:
    with open(filename, "r") as f:
        text = f.read().rstrip()
    return text


def string_to_file(string: str, filename: Path) -> None:
    with open(filename, "w") as f:
        f.write(string)


def load_config(file_path: Path) -> dict:
    file_content = file_path.read_text(encoding="utf-8")
    config = tomli.loads(file_content)
    return config


def get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def setup(seed: int) -> Tuple[torch.device, Path, Path]:
    """Set up an experiment.

    Set the seed, get the pytorch device and get the output
    directories.
    """
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_matplotlib_style()
    run_dir = Path(HydraConfig.get().run.dir)
    outputs_dir = Path(dirname(dirname(run_dir)))
    return device, run_dir, outputs_dir
