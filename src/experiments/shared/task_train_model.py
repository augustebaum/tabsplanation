import sys

import pytask
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from config import BLD_MODELS
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.utils import get_configs, get_time, hash_, save_config, setup
from tabsplanation.data import split_dataset, SyntheticDataset

cfgs = get_configs("latent_shift")
cfg = cfgs[0]


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


for model_name, model_cfg in cfg.models.items():
    # for cfg in cfgs:
    task = TaskTrainModel(model_cfg)

    @pytask.mark.task(id=task.id_)
    @pytask.mark.depends_on(task.depends_on)
    @pytask.mark.produces(task.produces)
    def task_train_model(depends_on, produces, cfg=task.cfg):
        device = setup(cfg.seed)

        # TODO: Replace with DataModule
        # Note that I no longer need to specify nb_dims here
        # I can just ask to generate a dataset with the right
        # number of dims (granted, this is wasted space since
        # I can just take a dataset with 250 columns and ask
        # to remove the last ones)
        dataset = SyntheticDataset(
            depends_on["xs"],
            depends_on["ys"],
            depends_on["coefs"],
            cfg.data.nb_dims,
            device,
        )
        subsets, loaders = split_dataset(
            dataset,
            cfg.data_module.validation_data_proportion,
            cfg.data_module.test_data_proportion,
            cfg.data_module.batch_size,
            weighted_sampler=False,
        )

        model_class = _get_class(cfg.model.class_name)
        model = model_class(**cfg.model.args)

        early_stopping_cb = EarlyStopping(
            monitor="val_loss", mode="min", patience=cfg.training.patience
        )

        version = f"{cfg.model.class_name}_{hash_(cfg)}_{get_time()}"
        tb_logger = TensorBoardLogger(save_dir=BLD_MODELS, version=version)

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            logger=tb_logger,
            callbacks=[early_stopping_cb],
            enable_model_summary=False,
        )

        trainer.fit(
            model=model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["validation"],
        )

        torch.save(model, produces["model"])
        save_config(cfg, produces["config"])


def _get_class(class_name: str):
    import tabsplanation.models  # ignore

    return getattr(sys.modules["tabsplanation.models"], class_name)
