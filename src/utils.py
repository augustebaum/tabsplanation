import hashlib
import json
from datetime import datetime
from typing import List

from omegaconf import DictConfig, OmegaConf

from config import BLD


def hash_(cfg: DictConfig):
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_str = json.dumps(cfg_dict, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(cfg_str.encode("ascii")).hexdigest()


def get_configs() -> List[DictConfig]:
    """Read experiment configuration files."""
    cfg_path = BLD / "config.yaml"
    cfgs = [OmegaConf.load(cfg_path)]
    return cfgs


def get_time() -> str:
    return datetime.now().isoformat()
