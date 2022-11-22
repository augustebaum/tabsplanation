import pytask
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from tabsplanation.config import BLD
from tabsplanation.tabsplanation.classifier import Classifier

models_dict = {"Classifier": Classifier}

# TODO: Eventually I'll want to be able to give a config like this one:
# Training two models with everything equal except hidden layers
# cfg_string = """
# - seed: 42
#   dataset:
#     distribution: uniform
#     nb_uncorrelated_dims: 2
#     nb_dims: 250
#   classifier:
#     model_name: Classifier
#     hidden_dims: [50, 50, 50]
#   training:
#     validation_data_proportion: 0.2
#     test_data_proportion: 0.2
#     batch_size: 200
#     learning_rate: 0.001
#     max_epochs: -1
#     patience: 5
# - seed: 42
#   dataset:
#     distribution: uniform
#     nb_uncorrelated_dims: 2
#     nb_dims: 250
#   classifier:
#     model_name: Classifier
#     hidden_dims: [50, 25]
#   training:
#     validation_data_proportion: 0.2
#     test_data_proportion: 0.2
#     batch_size: 200
#     learning_rate: 0.001
#     max_epochs: -1
#     patience: 5
# """


@pytask.mark.produces(BLD / "models" / "classifier" / "model.pt")
def task_create_untrained_model(produces):

    cfg_string = """
seed: 42
dataset:
  distribution: uniform
  nb_uncorrelated_dims: 2
  nb_dims: 250
  nb_classes: 3
classifier:
  class_name: Classifier
  hidden_dims: [50, 25]
training:
  validation_data_proportion: 0.2
  test_data_proportion: 0.2
  batch_size: 200
  learning_rate: 0.001
  max_epochs: -1
  patience: 5
"""

    cfg = OmegaConf.create(cfg_string)

    pl.seed_everything(cfg.seed, workers=True)

    clf_kwargs = {
        # "input_dim": loaders["train"].dataset.dataset.input_dim,
        "input_dim": cfg.dataset.nb_dims,
        # "output_dim": loaders["train"].dataset.dataset.output_dim,
        "output_dim": cfg.dataset.nb_classes,
        "hidden_dims": cfg.classifier.hidden_dims,
    }

    model_cls = models_dict[cfg.classifier.class_name]
    model = model_cls.new(**clf_kwargs)

    torch.save(model, produces)
