import hashlib
import json

import lightning as pl
from torch import optim


class BaseModel(pl.LightningModule):
    def __init__(self, learning_rate: float, model_name: str = None):
        super(BaseModel, self).__init__()

        self.learning_rate = learning_rate
        self.model_name = model_name or self.__class__.__name__

        self.save_hyperparameters("learning_rate")

    @classmethod
    def new(cls, **kwargs):
        return cls(**kwargs)

    @property
    def model_dict(self) -> dict:
        modules = {k: repr(v) for k, v in self._modules.items()}
        return {
            "model_name": self.model_name,
            "modules": modules,
            "hparams": self.hparams,
        }

    def get_code(self) -> str:
        """Combine all the information about the model and
        hyperparameters into a unique identifier."""
        # Produce a string from the model_dict reproducibly
        model_str = json.dumps(self.model_dict, sort_keys=True, ensure_ascii=True)
        # Hash the resulting string
        hash = hashlib.sha256(model_str.encode("ascii")).hexdigest()
        # Maybe ambitious?
        return hash[:6]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()})
        return loss
