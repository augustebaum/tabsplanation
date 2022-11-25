import hashlib
import json

import numpy as np
import pytask
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from config import BLD


def hash_(cfg: DictConfig):
    cfg_dict = OmegaConf.to_object(cfg)
    cfg_str = json.dumps(cfg_dict, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(cfg_str.encode("ascii")).hexdigest()


cfg = OmegaConf.create(
    dict(
        seed=42,
        gaussian=False,
        nb_dims=250,
        nb_uncorrelated_dims=2,
        nb_points_initial=100_001,
    )
)

for cfg in [cfg]:
    depends_on = {}

    cfg_code = hash_(cfg)
    produces = {
        "db": BLD / "data" / "test_db" / cfg_code / "db.npy",
        "config": BLD / "data" / "test_db" / cfg_code / "config.yaml",
    }

    @pytask.mark.task(id=cfg_code)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_make_db(depends_on, produces, cfg=cfg):
        np.save(produces["db"], np.eye(2))
        OmegaConf.save(config=cfg, f=produces["config"])


### New task that depends on the db


class MyDataset(Dataset):
    def __init__(self, x, name):
        self.x = x
        self.name = name

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

    def hello(self):
        print(f"Hello, my name is {self.name}")


cfg = OmegaConf.create(
    {
        "dataset": {"name": "Jean-Jean"},
        "data": {
            "seed": 42,
            "gaussian": False,
            "nb_dims": 250,
            "nb_uncorrelated_dims": 2,
            "nb_points_initial": 3,
        },
    }
)

for cfg in [cfg]:
    db_cfg = cfg.get("data")
    db_cfg_code = hash_(db_cfg)
    depends_on = {"data": BLD / "data" / "test_db" / db_cfg_code / "db.npy"}

    cfg_code = hash_(cfg)
    produces = {
        "dataset": BLD / "data" / "test_db_torch" / cfg_code / "db_torch.pt",
        "config": BLD / "data" / "test_db_torch" / cfg_code / "config.yaml",
    }

    @pytask.mark.task(id=cfg_code)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_make_torch_dataset(depends_on, produces, cfg=cfg):
        x = np.load(depends_on["data"])
        # dataset = MyDataset(x, cfg.dataset.name)
        print(f"Save data {x} to `MyDataset` class with name {cfg.dataset.name}")

        # torch.save(dataset, produces["dataset"])
        OmegaConf.save(config=cfg, f=produces["config"])
