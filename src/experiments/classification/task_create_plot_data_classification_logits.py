import pytask
import torch
from omegaconf import OmegaConf

from config import BLD_DATA, BLD_MODELS, BLD_PLOT_DATA
from tabsplanation.data import SyntheticDataset
from tabsplanation.types import Tensor
from utils import get_configs, hash_


cfgs = get_configs()

# if cfg is a dict, do
# cfg = cfg.model
# if cfg is a list, extract all keys called "model" and
# process each of them as dicts

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
    plot_data_dir = BLD_PLOT_DATA / "classification_logits" / id_
    produces = {
        "config": plot_data_dir / "config.yaml",
        "x0": plot_data_dir / "x0.pt",
        "logits": plot_data_dir / "logits.pt",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_create_plot_data_classification_logits(depends_on, produces, cfg=cfg):
        # pl.seed_everything(cfg.seed, workers=True)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        plot_data_cfg = cfg.plot_data_classification_logits

        # Make a grid of points in two dimensions
        x = torch.linspace(
            plot_data_cfg.lo, plot_data_cfg.hi, steps=plot_data_cfg.nb_steps
        )
        inputs_x: Tensor["nb_points", 2] = torch.cartesian_prod(x, x)

        # TODO: Transform the dataset into a DataModule
        # and load that DataModule with a key "dataset"
        # dataset = depends_on["dataset"]
        dataset = SyntheticDataset(
            depends_on["xs"],
            depends_on["ys"],
            depends_on["coefs"],
            cfg.data.nb_dims,
            "cpu",
        )
        # TODO: Fill the rest of the dimensions using the coefficients
        # inputs = dataset.fill_2d_point(inputs_x)
        inputs = torch.hstack(
            [inputs_x, torch.zeros((len(inputs_x), cfg.data.nb_dims - 2))]
        )

        normalized_inputs = dataset.normalize(inputs)

        model = torch.load(depends_on["model"])
        logits = model(normalized_inputs).detach()

        torch.save(x, produces["x0"])
        torch.save(logits, produces["logits"])

        OmegaConf.save(cfg, produces["config"])
