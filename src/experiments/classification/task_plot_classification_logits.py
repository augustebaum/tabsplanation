import matplotlib.pyplot as plt
import pytask
import torch
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import OmegaConf

from config import BLD_PLOT_DATA, BLD_PLOTS
from experiments.shared.utils import get_configs, get_map_img, hash_, setup


cfgs = get_configs("classification")

for cfg in cfgs:
    plot_data_dir = BLD_PLOT_DATA / "classification_logits" / hash_(cfg)
    depends_on = {
        "x": plot_data_dir / "x0.pt",
        "logits": plot_data_dir / "logits.pt",
    }

    id_ = hash_(cfg)
    plot_dir = BLD_PLOTS / "classification_logits" / id_
    produces = {
        "config": plot_dir / "config.yaml",
        "plot": plot_dir / "plot.svg",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_plot_classification_logits(depends_on, produces):

        x = torch.load(depends_on["x"])
        logits = torch.load(depends_on["logits"])

        setup(cfg.seed)

        nb_steps = len(x)

        x0, x1 = torch.meshgrid(x, x)

        logits_0 = logits[:, 0]

        fig, ax = plt.subplots(layout="constrained")
        cs = ax.contourf(
            x0,
            x1,
            logits_0.reshape((nb_steps, nb_steps)),
            zorder=1,
            cmap=LinearSegmentedColormap.from_list("", ["white", "red"]),
            norm=plt.Normalize(),
        )
        plt.colorbar(cs)

        ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)

        lo = x[0]
        hi = x[-1]
        ax.axis([lo, hi, lo, hi])

        fig.savefig(produces["plot"])
        plt.show(block=True)

        OmegaConf.save(cfg, produces["config"])
