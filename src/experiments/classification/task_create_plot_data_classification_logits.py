import torch

from config import BLD_PLOT_DATA
from experiments.shared.data.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import Task
from tabsplanation.data import CakeOnSeaDataset
from tabsplanation.types import Tensor


class TaskCreatePlotDataClassificationLogits(Task):
    def __init__(self, cfg):
        output_dir = BLD_PLOT_DATA / "classification_logits"
        super(TaskCreatePlotDataClassificationLogits, self).__init__(cfg, output_dir)

        self.depends_on = (
            TaskCreateCakeOnSea(self.cfg).produces
            | TaskTrainModel(self.cfg.classifier).produces
        )

        self.produces |= {
            "x0": self.produces_dir / "x0.pt",
            "logits": self.produces_dir / "logits.pt",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        # Make a grid of points in two dimensions
        x = torch.linspace(cfg.lo, cfg.hi, steps=cfg.nb_steps)
        inputs_x: Tensor["nb_points", 2] = torch.cartesian_prod(x, x)

        # TODO: Transform the dataset into a DataModule
        # and load that DataModule with a key "dataset"
        # dataset = depends_on["dataset"]
        dataset = CakeOnSeaDataset(
            depends_on["xs"],
            depends_on["ys"],
            depends_on["coefs"],
            cfg.data.nb_dims,
            "cpu",
        )

        inputs = dataset.fill_from_2d_point(inputs_x)

        normalized_inputs = dataset.normalize(inputs)

        model = torch.load(depends_on["model"])
        logits = model(normalized_inputs).detach()

        torch.save(x, produces["x0"])
        torch.save(logits, produces["logits"])
