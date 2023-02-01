from typing import Any, TypedDict

from config import BLD_DATA
from experiments.shared.utils import get_configs, setup, Task, write
from tabsplanation.data import CakeOnSeaDataModule
from tabsplanation.types import RelativeFloat


DatasetCfg = Any


class DataModuleCfg(TypedDict):
    dataset: DatasetCfg
    validation_data_proportion: RelativeFloat
    test_data_proportion: RelativeFloat
    batch_size: int
    correct_for_class_imbalance: bool


class TaskGetDataModule(Task):
    def __init__(self, cfg):
        import pdb

        pdb.set_trace()
        output_dir = BLD_DATA
        super(TaskGetDataModule, self).__init__(cfg, output_dir)

        # self.depends_on

        self.produces |= {"data_module": self.produces_dir / "data_module.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        setup(cfg.seed)

        dataset_cfg = cfg.args.dataset
        dataset_cls = from_object_name(dataset_cfg.class_name)
        dataset = dataset_cls(dataset_cfg.args)

        data_module = CakeOnSeaDataModule(dataset=dataset, **cfg.args)
        write(data_module, produces["data_module"])


cfgs = get_configs()
for cfg in cfgs:
    # task = TaskCreateCakeOnSea(cfg)
    pass

    # @pytask.mark.task(id=task.id_)
    # @pytask.mark.produces(task.produces)
    # def task_create_cake_on_sea(produces, cfg=task.cfg):
