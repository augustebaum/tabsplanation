from datetime import datetime

import pytask
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import tabsplanation.models.classifier
from config import BLD
from data.cake_on_sea.utils import hash_
from tabsplanation.data import split_dataset, SyntheticDataset


def get_time() -> str:
    return datetime.now().isoformat()


cfg_path = BLD / "config.yaml"

cfg = OmegaConf.load(cfg_path)

# if cfg is a dict, do
# cfg = cfg.model
# if cfg is a list, extract all keys called "model" and
# process each of them as dicts

for cfg in [cfg]:
    data_dir = BLD / "data" / "cake_on_sea" / hash_(cfg.data)
    depends_on = {
        "xs": data_dir / "xs.npy",
        "ys": data_dir / "ys.npy",
        "coefs": data_dir / "coefs.npy",
    }

    id_ = hash_(cfg.model)
    produces = {
        "tensorboard_logger": BLD / "models",
        "model": BLD / "models" / id_ / "model.pt",
        "config": BLD / "models" / id_ / "config.yaml",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_train_model(depends_on, produces, cfg=cfg):
        pl.seed_everything(cfg.seed, workers=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # TODO: Replace with DataModule
        # Note that I no longer need to specify nb_dims here
        # I can just ask to generate a dataset with the right
        # number of dims (granted, this is wasted space since
        # I can just take a dataset with 250 columns and ask
        # to remove the last ones)
        dataset = SyntheticDataset(
            depends_on["xs"],
            depends_on["ys"],
            # depends_on["coefs"],
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

        # TODO: Read this from model.class
        model_class = tabsplanation.models.classifier.Classifier
        model = model_class(**cfg.model.args)

        early_stopping_cb = EarlyStopping(
            monitor="val_loss", mode="min", patience=cfg.training.patience
        )

        version = f"{model.__class__.__name__}_{hash_(cfg.model)}_{get_time()}"
        tb_logger = TensorBoardLogger(
            save_dir=produces["tensorboard_logger"],
            version=version,
        )

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

        OmegaConf.save(cfg, produces["config"])
        torch.save(model, produces["model"])
