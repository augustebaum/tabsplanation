import random

from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, setup, Task


class TaskCreatePlotDataValidityLosses(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "validity_losses"
        super(TaskCreatePlotDataValidityLosses, self).__init__(cfg, output_dir)

        # task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)

        setup(cfg.seed)
        seeds = [random.randrange(100_000) for _ in range(cfg.nb_seeds)]

        for seed in seeds:
            for loss_fn in self.cfg.losses:
                for data_module_cfg in self.cfg.data_modules:
                    task_get_data_module = TaskGetDataModule(data_module_cfg)

                    classifier_cfg = self.cfg.classifier
                    classifier_cfg["output_dim"] = None
                    task_train_classifier = TaskTrainModel(classifier_cfg)

        # task_train_autoencoder = TaskTrainModel(self.cfg.autoencoder)

        # task_train_path_regularized_autoencoder = TaskTrainPathRegAe(self.cfg)
        # self.depends_on = task_create_cake_on_sea.produces
        # # self.depends_on |= {"classifier": task_train_classifier.produces}
        # self.depends_on |= {
        #     "path_regularized_autoencoder": task_train_path_regularized_autoencoder.produces,
        #     "autoencoder": task_train_autoencoder.produces,
        #     "classifier": task_train_classifier.produces,
        # }

        # self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        # for seed in seeds:
        #     for loss_fn in losses:
        #         for data_module in data_modules:
        # pass
        path_method = path_method_cls(
            classifier=classifier,
            autoencoder=autoencoder,
            loss_fn=loss,
            hparams=hparams,
        )

    def validity_rate(data_module, path_method):
        test_data = data_module.test_dataloader()


task, task_definition = define_task("validity_losses", TaskCreatePlotDataValidityLosses)
exec(task_definition)
