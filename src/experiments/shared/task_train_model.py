"""
Train a model based on a model config, which contains information on
the data to use for training, the training parameters (batch size, number
of epochs...) and the model to use (including its hyperparameters).

Assuming the shape of a model config is unlikely to change,
in order for this module to be versatile we need to parse all config
files for model configs, where the model configs are identified by
the key names (either "autoencoder" or "classifier").
"""
from typing import Dict, List

import lightning as pl

import pytask
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from omegaconf import OmegaConf

from config import BLD_MODELS
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.utils import (
    get_configs,
    get_data_module,
    get_module_object,
    get_time,
    hash_,
    save_config,
    setup,
)


def _get_class(class_name: str):
    return get_module_object("tabsplanation.models", class_name)


class TaskTrainModel:
    def __init__(self, cfg):
        self.cfg = cfg
        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)
        self.depends_on = task_create_cake_on_sea.produces

        self.id_ = hash_(cfg)
        produces_dir = BLD_MODELS / self.id_
        self.produces = {
            "model": produces_dir / "model.pt",
            "config": produces_dir / "config.yaml",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        data_module = get_data_module(depends_on, cfg, device)

        model_class = _get_class(cfg.model.class_name)
        model = model_class(**cfg.model.args).to(device)

        model = TaskTrainModel.train_model(data_module, model, cfg)

        torch.save(model, produces["model"])

    @classmethod
    def train_model(cls, data_module, model, cfg):
        early_stopping_cb = EarlyStopping(
            monitor="val_loss", mode="min", patience=cfg.training.patience
        )

        version = f"{model.__class__.__name__}_{hash_(cfg)}_{get_time()}"
        tb_logger = TensorBoardLogger(save_dir=BLD_MODELS, version=version)

        if torch.cuda.is_available():
            gpu_kwargs = {"accelerator": "gpu", "devices": -1}
        else:
            gpu_kwargs = {}

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            # val_check_interval=cfg.training.val_check_interval,
            logger=tb_logger,
            callbacks=[early_stopping_cb],
            enable_model_summary=False,
            **gpu_kwargs,
        )

        # Run a dummy forward pass to initialize Lazy layers
        # Two rows so that batch norm doesn't complain
        model.forward(data_module.train_set[0:2][0].reshape(2, -1))

        trainer.fit(model=model, datamodule=data_module)

        return model


def find_model_cfgs(cfg: Dict) -> List:
    """Given a config (nested) dict, recover all the values where the key
    is `"autoencoder"` or `"classifier"`."""

    # Alter the Depth-First-Search (DFS) algorithm slightly
    def modified_dfs(dict_: Dict, result: List) -> List:
        for k, v in dict_.items():
            # If the key fits, we don't need to look inside
            if k in ["autoencoder", "classifier"]:
                result.append(v)
            elif isinstance(v, dict):
                result = modified_dfs(v, result)
        return result

    return modified_dfs(cfg, [])


cfgs = get_configs()

_task_class = TaskTrainModel

for cfg in cfgs:
    model_cfgs: List[Dict] = find_model_cfgs(OmegaConf.to_object(cfg))

    for model_cfg in model_cfgs:
        # Task classes take an omegaconf, not a dict
        task = _task_class(OmegaConf.create(model_cfg))

        @pytask.mark.task(id=task.id_)
        @pytask.mark.depends_on(task.depends_on)
        @pytask.mark.produces(task.produces)
        def task_train_model(depends_on, produces, cfg=task.cfg):
            _task_class.task_function(depends_on, produces, cfg)
            save_config(cfg, produces["config"])
