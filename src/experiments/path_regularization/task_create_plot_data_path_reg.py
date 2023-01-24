from config import BLD_PLOT_DATA
from experiments.path_regularization.task_train_path_reg_ae import TaskTrainPathRegAe
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import define_task, get_data_module, Task
from tabsplanation.explanations.nice_path_regularized import PathRegularizedNICE


class TaskCreatePlotDataPathRegularization(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "path_reg"
        super(TaskCreatePlotDataPathRegularization, self).__init__(cfg, output_dir)

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)
        # task_train_classifier = TaskTrainModel(self.cfg.classifier)
        task_train_autoencoder = TaskTrainPathRegAe(self.cfg)
        self.depends_on = task_create_cake_on_sea.produces
        # self.depends_on |= {"classifier": task_train_classifier.produces}
        self.depends_on |= {"autoencoder": task_train_autoencoder.produces}

        # self.produces |= {"results": self.produces_dir / "results.pkl"}

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        autoencoder = depends_on["autoencoder"]["model"]

        # PathRegularizedNICE()


task, task_definition = define_task("path_reg", TaskCreatePlotDataPathRegularization)
exec(task_definition)
