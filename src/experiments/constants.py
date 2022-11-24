import os
from pathlib import Path

from .utils import load_config

this_file = Path(os.path.abspath(__file__))
ROOT_DIR = this_file.parent.parent
DATA_DIR = ROOT_DIR / "data"
CONFIG_DIR = ROOT_DIR / "config"

CONFIG = load_config(CONFIG_DIR / "config.toml")
