from typing import Any, TypedDict

from config import BLD_DATA
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.data.task_preprocess_forest_cover import (
    TaskPreprocessForestCover,
)
from experiments.shared.utils import setup, Task, write
from tabsplanation.data import CakeOnSeaDataModule, CakeOnSeaDataset, ForestCoverDataset
from tabsplanation.types import RelativeFloat


def init_CakeOnSeaDataset(depends_on, cfg, device):
    return CakeOnSeaDataset(
        xs_path=depends_on["xs"],
        ys_path=depends_on["ys"],
        coefs_path=depends_on["coefs"],
        nb_dims=cfg.nb_dims,
        device=device,
    )


def init_ForestCoverDataset(depends_on, cfg, device):
    return ForestCoverDataset(
        csv_path=depends_on["xs"],
        device=device,
    )


DatasetCfg = Any


class DataModuleCfg(TypedDict):
    dataset: DatasetCfg
    validation_data_proportion: RelativeFloat
    test_data_proportion: RelativeFloat
    batch_size: int
    correct_for_class_imbalance: bool


class TaskGetDataModule(Task):
    dataset_map = {
        "tabsplanation.data.CakeOnSeaDataset": {
            "task": TaskCreateCakeOnSea,
            "init_fn": init_CakeOnSeaDataset,
        },
        "tabsplanation.data.ForestCoverDataset": {
            "task": TaskPreprocessForestCover,
            "init_fn": init_CakeOnSeaDataset,
        },
    }

    def __init__(self, cfg):
        output_dir = BLD_DATA / "data_modules"
        super(TaskGetDataModule, self).__init__(cfg, output_dir)

        dataset_cfg = self.cfg.dataset
        dataset_task_cls = TaskGetDataModule.dataset_map[dataset_cfg.class_name]["task"]
        dataset_task = dataset_task_cls(dataset_cfg.args)

        self.task_deps = [dataset_task]

        self.depends_on = dataset_task.produces
        self.produces |= {"data_module": self.produces_dir / "data_module.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        device = setup(cfg.seed)

        init_dataset = TaskGetDataModule.dataset_map[cfg.dataset.class_name]["init_fn"]

        dataset = init_dataset(depends_on, cfg.dataset.args, device)

        data_module = CakeOnSeaDataModule(dataset=dataset, **cfg.args)
        write(data_module, produces["data_module"])
