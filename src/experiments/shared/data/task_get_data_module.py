from config import BLD_DATA
from experiments.shared.utils import get_configs, setup, Task, write
from tabsplanation.data import CakeOnSeaDataModule


class TaskGetDataModule(Task):
    def __init__(self, cfg):
        self.cfg = cfg.data
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
    task = TaskCreateCakeOnSea(cfg)

    # @pytask.mark.task(id=task.id_)
    # @pytask.mark.produces(task.produces)
    # def task_create_cake_on_sea(produces, cfg=task.cfg):
