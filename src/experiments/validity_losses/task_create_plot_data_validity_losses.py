import random
from typing import Dict, List, TypeAlias, TypedDict

from torch.utils.data import Dataset

from config import BLD_PLOT_DATA, EXPERIMENT_CONFIGS
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import ModelCfg, TaskTrainModel
from experiments.shared.utils import read, setup, Task
from tabsplanation.explanations.losses import ValidityLoss
from tabsplanation.models import AutoEncoder, Classifier
from tabsplanation.types import PositiveInt, RelativeFloat, Seed


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

        import pdb

        pdb.set_trace()
        # For each dataset
        for data_module_cfg in self.cfg.data_modules:
            # Get the DataModule
            task_dataset = TaskGetDataModule.task_dataset(data_module_cfg)
            self.task_deps.append(task_dataset)

            # Train a classifier
            classifier_cfg = self.cfg.classifier
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
                autoencoder_cfg = self.cfg.autoencoder
                autoencoder_cfg.data_module = data_module_cfg
                autoencoder_cfg.seed = seed
                task_autoencoder = TaskTrainModel(autoencoder_cfg)

                # Add it to the dependencies
                self.task_deps.append(task_autoencoder)
                autoencoder_deps[seed] = task_autoencoder.produces

        self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        for data_module_name, values in depends_on.items():

            device = setup(cfg.seed)

            data_module = TaskGetDataModule.read_data_module(
                values["dataset"], cfg.data_module, device
            )

            classifier = read(values["classifier"])

            for seed, autoencoder_path in values["autoencoders"].items():
                autoencoder = read(autoencoder_path)

                path_methods = cfg.explainers
                for path_method in path_methods:

                    loss_fns = cfg.losses
                    for loss_fn in loss_fns:
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

                        result.append(result)

    @staticmethod
    def validity_rate(data_module, classifier, autoencoder, loss_fn, path_method):
        pass


from omegaconf import OmegaConf

cfg = OmegaConf.load(EXPERIMENT_CONFIGS / "validity_losses.yaml")
task = TaskCreatePlotDataValidityLosses(cfg)
task, task_def = task.define_task()
exec(task_def)
