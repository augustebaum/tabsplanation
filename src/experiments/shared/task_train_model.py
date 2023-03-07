"""
Train a model based on a model config, which contains information on
the data to use for training, the training parameters (batch size, number
of epochs...) and the model to use (including its hyperparameters).
"""
from typing import Dict, TypeAlias, TypedDict

import lightning as pl

import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from config import BLD_MODELS
from experiments.shared.data.task_get_data_module import (
    DataModuleCfg,
    TaskGetDataModule,
)
from experiments.shared.utils import (
    get_module_object,
    get_time,
    hash_,
    setup,
    Task,
    write,
)
from tabsplanation.explanations.nice_path_regularized import PathRegularizedNICE
from tabsplanation.models.classifier.classifier import Classifier, plot_confusion_matrix


def _get_class(class_name: str):
    return get_module_object("tabsplanation.models", class_name)


ClassName: TypeAlias = str
ModelArgs: TypeAlias = Dict


class ModelCfg:
    class_name: ClassName
    args: ModelArgs


class TrainingCfg(TypedDict):
    max_epochs: int
    patience: int


class TrainModelCfg(TypedDict):
    seed: int
    data_module: DataModuleCfg
    training: TrainingCfg
    model: ModelCfg


class TaskTrainModel(Task):
    def __init__(self, cfg: TrainModelCfg):
        output_dir = BLD_MODELS
        super(TaskTrainModel, self).__init__(cfg, output_dir)

        task_dataset = TaskGetDataModule.task_dataset(self.cfg.data_module)

        self.task_deps = [task_dataset]
        self.depends_on = task_dataset.produces
        self.produces |= {"model": self.produces_dir / "model.pt"}
        if self.cfg.model.class_name == "Classifier":
            self.produces |= {
                "model_confusion_matrix": self.produces_dir / "confusion_matrix.svg"
            }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        data_module = TaskGetDataModule.read_data_module(
            depends_on, cfg.data_module, device
        )

        model_class = _get_class(cfg.model.class_name)
        model = model_class(
            input_dim=data_module.input_dim,
            output_dim=data_module.output_dim,
            **cfg.model.args,
        ).to(device)

        model = TaskTrainModel.train_model(data_module, model, cfg)
        if isinstance(model, Classifier):
            plot = plot_confusion_matrix(model.to(device), data_module)
            write(plot, produces["model_confusion_matrix"])

        write(model, produces["model"])

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
        # Run with two rows so that batch norm doesn't complain
        model.forward(data_module.train_set[0:2][0].reshape(2, -1))

        trainer.fit(model=model, datamodule=data_module)
        if not isinstance(model, PathRegularizedNICE):
            trainer = pl.Trainer(
                devices=1,
                logger=tb_logger,
                enable_model_summary=False,
            )
            trainer.test(model=model, datamodule=data_module)

        return model
