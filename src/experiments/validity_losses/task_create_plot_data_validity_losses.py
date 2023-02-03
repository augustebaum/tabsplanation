import random
from typing import Dict, List, TypedDict

from config import BLD_PLOT_DATA, EXPERIMENT_CONFIGS
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import ModelCfg, TaskTrainModel
from experiments.shared.utils import setup, Task
from tabsplanation.explanations.losses import ValidityLoss
from tabsplanation.types import PositiveInt


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


class TaskCreatePlotDataValidityLosses(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "validity_losses"
        super(TaskCreatePlotDataValidityLosses, self).__init__(cfg, output_dir)

        setup(cfg.seed)
        seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

        # For each dataset
        for data_module_cfg in self.cfg.data_modules:
            # Get the DataModule
            task_data_module = TaskGetDataModule(data_module_cfg)
            self.task_deps.append(task_data_module)

            # Train a classifier
            classifier_cfg = self.cfg.classifier
            classifier_cfg.data_module = data_module_cfg
            task_classifier = TaskTrainModel(classifier_cfg)

            # Add it to the dependencies
            self.task_deps.append(task_classifier)
            data_module_name = data_module_cfg.dataset.class_name
            self.depends_on[data_module_name] = {
                "data_module": task_data_module.produces,
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
        pass

    def validity_rate(data_module, path_method):
        pass
        # test_data = data_module.test_dataloader()


from omegaconf import OmegaConf

cfg = OmegaConf.load(EXPERIMENT_CONFIGS / "validity_losses.yaml")
task, task_def = TaskCreatePlotDataValidityLosses(cfg).define_task()
import pdb

pdb.set_trace()

exec(task_def)
