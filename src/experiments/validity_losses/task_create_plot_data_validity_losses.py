import random
from typing import Dict, Iterator, List, TypeAlias, TypedDict

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from config import BLD_PLOT_DATA
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import ModelCfg, TaskTrainModel
from experiments.shared.utils import clone_config, get_object, read, setup, Task, write
from tabsplanation.explanations.losses import ValidityLoss
from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.models import AutoEncoder, Classifier
from tabsplanation.types import ExplanationPath, PositiveInt, RelativeFloat, Seed


class ExplainerCfg:
    class_name: str
    args: Dict


class ValidityLossesCfg(TypedDict):
    seed: int
    nb_seeds: PositiveInt
    classifier: ModelCfg
    autoencoder: ModelCfg
    explainers: List[ExplainerCfg]
    losses: List[ValidityLoss]


DataModuleName: TypeAlias = str
ExplainerName: TypeAlias = str
ValidityLossName: TypeAlias = str


# Just kind of gave up on this one...
class AA(TypedDict):
    dataset: Dataset
    classifier: Classifier
    autoencoders: Dict[Seed, AutoEncoder]


ValidityLossesDependsOn: TypeAlias = Dict[DataModuleName, AA]


class ValidityLossesResult(TypedDict):
    data_module: DataModuleName
    path_method: ExplainerName
    loss: ValidityLossName
    seed: Seed
    validity_rate: RelativeFloat


class TaskCreatePlotDataValidityLosses(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "validity_losses"
        super(TaskCreatePlotDataValidityLosses, self).__init__(cfg, output_dir)

        setup(cfg.seed)
        seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

        # Make sure self.cfg is not modified
        cfg = clone_config(self.cfg)

        # For each dataset
        for data_module_cfg in cfg.data_modules:
            # Get the DataModule
            task_dataset = TaskGetDataModule.task_dataset(data_module_cfg)
            self.task_deps.append(task_dataset)

            # Train a classifier
            classifier_cfg = cfg.classifier
            classifier_cfg.data_module = data_module_cfg
            task_classifier = TaskTrainModel(classifier_cfg)

            # Add it to the dependencies
            self.task_deps.append(task_classifier)
            data_module_name = data_module_cfg.dataset.class_name
            self.depends_on[data_module_name] = {
                "dataset": task_dataset.produces,
                "classifier": task_classifier.produces,
                "autoencoders": {},
            }
            # Short-hand
            autoencoder_deps = self.depends_on[data_module_name]["autoencoders"]

            # Then for each seed
            for seed in seeds:
                # Train an autoencoder
                autoencoder_cfg = cfg.autoencoder
                autoencoder_cfg.data_module = data_module_cfg
                autoencoder_cfg.seed = seed
                task_autoencoder = TaskTrainModel(autoencoder_cfg)

                # Add it to the dependencies
                self.task_deps.append(task_autoencoder)
                autoencoder_deps[seed] = task_autoencoder.produces

        self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        results = []

        for data_module_name, values in depends_on.items():
            device = setup(cfg.seed)

            classifier_cfg = OmegaConf.load(values["classifier"]["full_config"])

            data_module = TaskGetDataModule.read_data_module(
                values["dataset"], classifier_cfg.data_module, device
            )

            classifier = read(values["classifier"]["model"], device=device)

            for seed, autoencoder_path in values["autoencoders"].items():
                autoencoder = read(autoencoder_path["model"], device=device)

                for path_method in cfg.explainers:

                    for loss_fn in cfg.losses:
                        validity_rate = TaskCreatePlotDataValidityLosses.validity_rate(
                            data_module, classifier, autoencoder, loss_fn, path_method
                        )

                        result = {
                            "data_module": data_module_name,
                            "path_method": path_method,
                            "seed": seed,
                            "loss": loss_fn,
                            "validity_rate": validity_rate,
                        }

                        results.append(result)

        write(results, produces["results"])

    @staticmethod
    def validity_rate(data_module, classifier, autoencoder, loss_fn, explainer):
        torch.cuda.empty_cache()
        test_x = data_module.test_data[0][:20_000]

        y_predict = classifier.predict(test_x)
        target = random_targets_like(y_predict, data_module.dataset.output_dim)

        loss_fn = get_object(loss_fn)

        explainer_cls = get_object(explainer.class_name)
        explainer_hparams = explainer.args.hparams

        explainer = explainer_cls(classifier, autoencoder, explainer_hparams, loss_fn)

        paths: Iterator[ExplanationPath] = explainer.get_counterfactuals_iterator(
            test_x, target
        )

        validity: List[bool] = [path.is_valid() for path in paths]

        validity_rate = sum(validity) / len(validity)
        return validity_rate
