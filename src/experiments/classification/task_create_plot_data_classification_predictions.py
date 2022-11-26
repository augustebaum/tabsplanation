import pytask
import torch
from omegaconf import OmegaConf

from config import BLD
from data.cake_on_sea.utils import hash_
from tabsplanation.data import SyntheticDataset
from tabsplanation.types import Tensor


cfg_path = BLD / "config.yaml"

cfg = OmegaConf.load(cfg_path)

# if cfg is a dict, do
# cfg = cfg.model
# if cfg is a list, extract all keys called "model" and
# process each of them as dicts

for cfg in [cfg]:
    data_dir = BLD / "data" / "cake_on_sea" / hash_(cfg.data)
    depends_on = {
        "xs": data_dir / "xs.npy",
        "ys": data_dir / "ys.npy",
        "coefs": data_dir / "coefs.npy",
    }

    model_dir = BLD / "models" / hash_(cfg.model) / "model.pt"
    depends_on |= {"model": model_dir}

    id_ = hash_(cfg)
    plot_data_dir = BLD / "plot_data" / "classification_predictions" / id_
    produces = {
        "config": plot_data_dir / "config.yaml",
        "inputs": plot_data_dir / "inputs.pt",
        "outputs": plot_data_dir / "outputs.pt",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_create_plot_data_classification_predictions(depends_on, produces, cfg=cfg):
        # pl.seed_everything(cfg.seed, workers=True)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        plot_data_cfg = cfg.plot_data_classification_predictions

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
            # depends_on["coefs"],
            cfg.data.nb_dims,
            "cpu",
        )
        # Fill the rest of the dimensions using the coefficients
        # inputs = dataset.fill_2d_point(inputs_x)
        inputs = torch.hstack(
            [inputs_x, torch.zeros((len(inputs_x), cfg.data.nb_dims - 2))]
        )

        normalized_inputs = dataset.normalize(inputs)

        model = torch.load(depends_on["model"])
        outputs = model.softmax(normalized_inputs).detach()

        torch.save(inputs, produces["inputs"])
        torch.save(outputs, produces["outputs"])

        OmegaConf.save(cfg, produces["config"])
