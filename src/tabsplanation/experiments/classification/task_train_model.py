import pytask
import torch

from tabsplanation.config import BLD


dataset_dir = BLD / "data" / "cake_on_sea" / "dataset_id"


@pytask.mark.depends_on(
    {
        "data": {
            "dataset": dataset_dir / "dataset.pkl",
            "subsets": dataset_dir / "subsets.pkl",
            "loaders": dataset_dir / "loaders.pkl",
        }
    }
)
def task_train_model(depends_on):
    pl.seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = depends_on["dataset"]
    subsets = depends_on["subsets"]
    loaders = depends_on["loaders"]

    for model in models:

        early_stopping_cb = EarlyStopping(
            monitor="val_loss", mode="min", patience=patience
        )

        tb_logger = TensorBoardLogger(
            save_dir=produces["tensorboard_logger"],
            version=model.model_name,
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=tb_logger,
            callbacks=[
                # checkpoint_callback,
                early_stopping_cb,
            ],
        )

        trainer.fit(
            model=model,
            train_dataloaders=loaders["train"],
            val_dataloaders=loaders["validation"],
        )

    for model in models:
        produces["clf"]
