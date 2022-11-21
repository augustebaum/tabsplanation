"""Experiment functions."""

import json
import os
from logging import Logger
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from matplotlib.figure import Figure
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn

from tabsplanation.data import (
    split_dataset,
    SyntheticDataset,
    TrainLoader,
    ValidationLoader,
)
from .models import AE_MODELS, CLF_MODELS, get_models
from .utils import get_time


def get_data(cfg, device):
    """Read the data from disk and create subsets along with
    the corresponding `DataLoader`s."""

    dataset_dir = Path(cfg.dataset.dir)
    xs_path = dataset_dir / "xs.npy"
    ys_path = dataset_dir / "ys.npy"

    dataset = SyntheticDataset(
        xs_path,
        ys_path,
        cfg.dataset.input_dim,
        device,
    )

    ae_cfg = cfg.autoencoder.training
    subsets, loaders = split_dataset(
        dataset,
        ae_cfg.validation_data_proportion,
        ae_cfg.test_data_proportion,
        ae_cfg.batch_size,
        weighted_sampler=False,
    )

    return dataset, subsets, loaders


def load_models(path: Path, log: Logger) -> List[nn.Module]:
    """Load saved models from disk.

    The file structure is something like:
    ```
    path
    ├── model_6ddf70
    │   └── final_model.pt
    └── model_b5a011
        └── final_model.pt
    ```
    """

    log.debug(f"Models directory is {path}")

    models_dirs = [dir for dir in os.listdir(path) if dir.startswith("model")]
    log.debug(f"AE directories are {models_dirs}")

    models = []
    for dir in models_dirs:
        model = torch.load(path / dir / "final_model.pt")
        with open(path / dir / "model_config.json", "r") as f:
            model_config = json.load(f)
        model.model_name = model_config.get("model_name")
        model.eval()
        models.append(model)

    return models


def save_model(model: nn.Module, model_dir: Path) -> None:
    torch.save(model, model_dir / "final_model.pt")
    # Save extra info about the model, for future reference
    with open(model_dir / "model_config.json", "w") as f:
        json.dump(model.model_dict, f, indent=2)


def train_models(
    models: List[nn.Module],
    models_dir: Path,
    train_loader: TrainLoader,
    val_loader: ValidationLoader,
    max_epochs: int,
    patience: int,
):

    for model in models:
        # A deterministic code to differentiate models made from the same class
        code = model.get_code()
        model_dir = models_dir / f"model_{code}"

        # TODO: Find a better solution
        if model_dir.exists():
            model_dir = Path(f"{model_dir}_")

        # This is to control where checkpoints are saved
        checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            save_last=True,  # Create a symlink called `last.ckpt`
        )

        early_stopping_cb = EarlyStopping(
            monitor="val_loss", mode="min", patience=patience
        )

        tb_logger = TensorBoardLogger(
            save_dir=os.path.dirname(os.path.dirname(model_dir)),
            version=model.model_name,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=tb_logger,
            callbacks=[checkpoint_callback, early_stopping_cb],
        )

        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
        )

        save_model(model, model_dir)


def train_aes(cfg, device, run_dir, loaders):
    ae_kwargs = {
        "input_dim": cfg.dataset.input_dim,
        "hidden_dim": cfg.autoencoder.hidden_dim,
        "latent_dim": cfg.autoencoder.latent_dim,
        "learning_rate": cfg.autoencoder.training.learning_rate,
        "kl_factor": cfg.autoencoder.kl_factor,
    }

    aes = get_models(AE_MODELS, cfg.autoencoder.models, device, **ae_kwargs)

    # TODO: Does this work for multiruns as well?
    models_dir = run_dir / "aes"

    train_models(
        aes,
        models_dir,
        loaders["train"],
        loaders["validation"],
        cfg.autoencoder.training.max_epochs,
        patience=cfg.training.patience,
    )

    return aes, models_dir


def train_clfs(cfg, device, run_dir, loaders):

    clf_kwargs = {
        "input_dim": loaders["train"].dataset.dataset.input_dim,
        "output_dim": loaders["train"].dataset.dataset.output_dim,
        "hidden_dims": cfg.classifier.hidden_dims,
    }
    clfs = get_models(CLF_MODELS, cfg.classifier.models, device, **clf_kwargs)

    # TODO: Does this work for multiruns as well?
    models_dir = run_dir / "clfs"

    train_models(
        clfs,
        models_dir,
        loaders["train"],
        loaders["validation"],
        cfg.autoencoder.training.max_epochs,
        patience=cfg.training.patience,
    )

    return clfs, models_dir


def save_plot(fig: Figure, plot_name: str, dir: Path) -> Path:
    time = get_time()
    plot_file_path = dir / f"{plot_name}_{time}.svg"
    fig.savefig(plot_file_path)
    return plot_file_path
