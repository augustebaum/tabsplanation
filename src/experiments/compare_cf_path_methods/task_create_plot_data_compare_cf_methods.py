import pytask
import torch
from omegaconf import OmegaConf

from config import BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import (
    get_configs,
    get_module_object,
    hash_,
    save_config,
    setup,
)
from tabsplanation.data import SyntheticDataset


def _get_method(method_name):
    return get_module_object("tabsplanation.explanations", method_name)


class TaskCreatePlotDataCfPathMethods:
    def __init__(self, cfg):
        self.cfg = cfg

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)

        self.depends_on = task_create_cake_on_sea.produces

        for method in self.cfg.methods:
            task_train_autoencoder = TaskTrainModel(method.args.autoencoder)
            task_train_classifier = TaskTrainModel(method.args.classifier)

            self.depends_on = (
                self.depends_on
                | {f"autoencoder_{method.class_name}": task_train_autoencoder.produces}
                | {f"classifier_{method.class_name}": task_train_classifier.produces}
            )

        self.id_ = hash_(self.cfg)
        plot_data_dir = BLD_PLOT_DATA / "cf_path_methods" / self.id_
        self.produces = {
            "config": plot_data_dir / "config.yaml",
            # "results": plot_data_dir / "results.csv",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        device = setup(cfg.seed)

        dataset = SyntheticDataset(
            depends_on["xs"],
            depends_on["ys"],
            depends_on["coefs"],
            cfg.data.nb_dims,
            device,
        )
        # normalize = dataset.normalize
        # normalize_inverse = dataset.normalize_inverse

        results = {}

        for method_cfg in cfg.methods:
            # Recover the class from its name
            method_class = _get_method(method_cfg.class_name)
            # import pdb

            # pdb.set_trace()

            autoencoder = torch.load(
                depends_on[f"autoencoder_{method_cfg.class_name}"]["model"]
            )
            classifier = torch.load(
                depends_on[f"classifier_{method_cfg.class_name}"]["model"]
            )
            # Instantiate method using parameters from config
            kwargs = OmegaConf.to_object(method_cfg.args) | {
                "autoencoder": autoencoder,
                "classifier": classifier,
            }
            method = method_class(**kwargs)

            # TODO: Use test loader
            xs, ys = dataset[:5]
            # for i, x in test_data:
            for x in xs:
                # path = method.get_counterfactuals(x, target_map[i])
                path = method.get_counterfactuals(x, 2)
                # measurements = _measure(path)
                print(path)
                # results[method_cfg.class_name] = measurements

        # with open(produces["paths"], "wb") as paths_file:
        #     pickle.dump(paths, paths_file)


cfgs = get_configs("compare_cf_methods")
_task_class = TaskCreatePlotDataCfPathMethods

for cfg in cfgs:
    task = _task_class(cfg)

    @pytask.mark.task(id=task.id_)
    @pytask.mark.depends_on(task.depends_on)
    @pytask.mark.produces(task.produces)
    def task_create_plot_data_cf_path_methods(depends_on, produces, cfg=task.cfg):
        _task_class.task_function(depends_on, produces, cfg)
        save_config(cfg, produces["config"])
