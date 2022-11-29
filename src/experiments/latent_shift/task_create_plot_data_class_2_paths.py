import pytask
import torch
from omegaconf import OmegaConf

from config import BLD_DATA, BLD_MODELS, BLD_PLOT_DATA
from experiments.shared.task_create_cake_on_sea import TaskCreateCakeOnSea
from experiments.shared.task_train_model import TaskTrainModel
from experiments.shared.utils import get_configs, hash_, save_config, setup
from tabsplanation.data import SyntheticDataset


class TaskCreatePlotDataClass2Paths:
    def __init__(self, cfg):
        self.cfg = cfg

        task_create_cake_on_sea = TaskCreateCakeOnSea(self.cfg)
        self.depends_on = task_create_cake_on_sea.produces

        task_train_model = TaskTrainModel(self.cfg)
        self.depends_on |= task_train_model.produces


cfgs = get_configs("latent_shift")

for cfg in cfgs:
    data_dir = BLD_DATA / "cake_on_sea" / hash_(cfg.data)
    depends_on = {
        "xs": data_dir / "xs.npy",
        "ys": data_dir / "ys.npy",
        "coefs": data_dir / "coefs.npy",
    }

    model_dir = BLD_MODELS / hash_(cfg.model) / "model.pt"
    depends_on |= {"model": model_dir}

    id_ = hash_(cfg)
    plot_data_dir = BLD_PLOT_DATA / "class_2_paths" / id_
    produces = {
        "config": plot_data_dir / "config.yaml",
        "paths": "paths.pkl"
        # "x0": plot_data_dir / "x0.pt",
        # "logits": plot_data_dir / "logits.pt",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_create_plot_data_class_2_paths(depends_on, produces, cfg=cfg):

        inputs_denorm = []

        # Cover class 2 (4 corners and middle)
        margin = 2

        inputs_denorm = np.c_[np.linspace(35 + margin, 45 - margin), np.ones(50) * 43]

        inputs = dataset.normalize(torch.tensor(inputs_denorm).to(torch.float))

        # ae = [ae for ae in aes if ae.model_name == "VAE"][0]
        # clf = clfs[0]

        # target_map = {0: 2, 1: None, 2: 0}

        # def get_target_class(input):
        #     return target_map[np.argmax(clf.softmax(input).detach()).item()]

        torch.set_printoptions(precision=3, sci_mode=False)
        paths = [
            make_path(
                input=input,
                target_class=get_target_class(input),
                clf=clf,
                ae=ae,
            )
            for input in inputs
        ]

        for path in paths:
            path.explained_input.input = dataset.normalize_inverse(
                path.explained_input.input
            )
            path.xs = dataset.normalize_inverse(path.xs)

        torch.save(logits, produces["logits"])

        save_config(cfg, produces["config"])
