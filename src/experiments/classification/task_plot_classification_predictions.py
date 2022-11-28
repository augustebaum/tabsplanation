import matplotlib.pyplot as plt
import pytask
import torch
from omegaconf import OmegaConf

from config import BLD_PLOT_DATA, BLD_PLOTS
from experiments.shared.utils import get_configs, get_map_img, hash_, setup


cfgs = get_configs("classification")


for cfg in cfgs:
    plot_data_dir = BLD_PLOT_DATA / "classification_logits" / hash_(cfg)
    depends_on = {
        "config": plot_data_dir / "config.yaml",
        "x": plot_data_dir / "x0.pt",
        "logits": plot_data_dir / "logits.pt",
    }

    id_ = hash_(cfg)
    plot_dir = BLD_PLOTS / "classification_predictions" / id_
    produces = {
        "config": plot_dir / "config.yaml",
        "plot": plot_dir / "plot.svg",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_plot_classification_predictions(depends_on, produces):

        setup(cfg.seed)
        fig, ax = plt.subplots(layout="constrained")

        x = torch.load(depends_on["x"])
        logits = torch.load(depends_on["logits"])
        prbs = logits.softmax(dim=-1)

        inputs = torch.cartesian_prod(x, x)
        ax.scatter(inputs[:, 0], inputs[:, 1], c=prbs, alpha=0.5, marker="s", zorder=1)

        ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)

        cfg_plot_data = cfg.plot_data_classification_logits
        ax.axis(
            [cfg_plot_data.lo, cfg_plot_data.hi, cfg_plot_data.lo, cfg_plot_data.hi]
        )

        fig.savefig(produces["plot"])
        plt.show(block=True)

        OmegaConf.save(cfg, produces["config"])
