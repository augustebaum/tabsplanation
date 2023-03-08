import itertools
import random
from typing import Dict, List, Optional, TypeAlias, TypedDict

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from config import BLD_PLOT_DATA
from experiments.shared.data.task_get_data_module import TaskGetDataModule
from experiments.shared.task_train_model import ModelCfg, TaskTrainModel
from experiments.shared.utils import clone_config, get_object, read, setup, Task, write
from tabsplanation.explanations.nice_path_regularized import random_targets_like
from tabsplanation.models import AutoEncoder, Classifier
from tabsplanation.types import B, H, PositiveInt, RelativeFloat, S, Seed, Tensor


class ExplainerCfg:
    class_name: str
    args: Dict


class ValidityLossCfg:
    class_name: str
    args: Optional[Dict]


class ValidityLossesCfg(TypedDict):
    seed: int
    nb_seeds: PositiveInt
    classifier: ModelCfg
    autoencoder: ModelCfg
    explainers: List[ExplainerCfg]
    losses: List[ValidityLossCfg]


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
            print(f"Dataset: {data_module.dataset.__class__.__name__}")

            classifier = read(values["classifier"]["model"], device=device)

            for seed, autoencoder_path in values["autoencoders"].items():
                print(f"Seed: {seed}")
                autoencoder = read(autoencoder_path["model"], device=device)

                for path_method, loss_fn in tqdm(
                    list(itertools.product(cfg.explainers, cfg.losses))
                ):
                    validity_rate = TaskCreatePlotDataValidityLosses.validity_rate(
                        data_module, classifier, autoencoder, loss_fn, path_method
                    )

                    result = {
                        "data_module": data_module_name,
                        "path_method": path_method,
                        "seed": seed,
                        "loss": loss_fn.name,
                        "validity_rate": validity_rate,
                    }

                    results.append(result)

        write(results, produces["results"])

    @staticmethod
    def validity_rate(data_module, classifier, autoencoder, loss_fn, explainer):
        torch.cuda.empty_cache()

        explainer_cls = get_object(explainer.class_name)
        explainer_hparams = explainer.args.hparams

        loss_cls = get_object(loss_fn.class_name)
        loss_fn = loss_cls() if loss_fn.args is None else loss_cls(**loss_fn.args)

        explainer = explainer_cls(classifier, autoencoder, explainer_hparams, loss_fn)

        nb_valid = 0
        for test_x, _ in data_module.test_dataloader():
            test_x = test_x.to(classifier.device)

            y_predict = classifier.predict(test_x)
            target = random_targets_like(y_predict, data_module.dataset.output_dim)

            cfs: Tensor[S, B, H] = explainer.get_cfs(test_x, target)
            cf_preds = classifier.predict(cfs)

            path_numbers, step_numbers = torch.where((target == cf_preds).T)
            valid_path_numbers = path_numbers.unique()
            nb_valid = nb_valid + len(valid_path_numbers)
            # validity_rate = nb_valid / nb_paths

            # batch_size = len(test_x)
            # validity_rate += validity_rate * batch_size

        validity_rate = nb_valid / len(data_module.test_set)

        return validity_rate
