import matplotlib.pyplot as plt
import pytask
import torch
from omegaconf import OmegaConf

from config import BLD, BLD_PLOT_DATA, BLD_PLOTS
from data.cake_on_sea.utils import hash_


cfg_path = BLD / "config.yaml"

cfg = OmegaConf.load(cfg_path)

# if cfg is a dict, do
# cfg = cfg.model
# if cfg is a list, extract all keys called "model" and
# process each of them as dicts

for cfg in [cfg]:
    plot_name = "classification_predictions"

    plot_data_dir = BLD_PLOT_DATA / plot_name / hash_(cfg)
    depends_on = {
        "config": plot_data_dir / "config.yaml",
        "inputs": plot_data_dir / "inputs.pt",
        "outputs": plot_data_dir / "outputs.pt",
    }

    id_ = hash_(cfg)
    plot_dir = BLD_PLOTS / plot_name / id_
    produces = {
        "config": plot_dir / "config.yaml",
        "plot": plot_dir / "plot.svg",
    }

    @pytask.mark.task(id=id_)
    @pytask.mark.depends_on(depends_on)
    @pytask.mark.produces(produces)
    def task_plot_classification_predictions(depends_on, produces):

        # set_matplotlib_style()
        fig, ax = plt.subplots(layout="constrained")

        inputs = torch.load(depends_on["inputs"])
        outputs = torch.load(depends_on["outputs"])

        ax.scatter(
            inputs[:, 0], inputs[:, 1], c=outputs, alpha=0.5, marker="s", zorder=1
        )
        ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)
        cfg_plot_data = cfg.plot_data_classification_predictions
        ax.axis(
            [cfg_plot_data.lo, cfg_plot_data.hi, cfg_plot_data.lo, cfg_plot_data.hi]
        )

        fig.savefig(produces["plot"])
        plt.show(block=True)

        OmegaConf.save(cfg, produces["config"])
        # if cfg.plots.save:
        #     plot_file_path = save_plot(fig, "classification_probas", run_dir)
        #     # log.info(f"Plot saved at path {plot_file_path}")
        # if cfg.plots.show:
        #     plt.show(block=True)

        # plt.cla()

        # # Make colors from prediction
        # # Apply argmax
        # # Gives a column of indices
        # predictions = outputs.argmax(axis=1, keepdims=True)
        # # Now shape it into colors
        # # colors = np.zeros_like(outputs)
        # # np.put_along_axis(colors, predictions, 1, axis=1)
        # # torch.zeros_like(outputs).scatter_(
        # index = torch.tensor([range(len(predictions))])
        # src = torch.ones((1, len(predictions)))
        # colors = torch.zeros_like(outputs).scatter_(0, index, src)

        # ax.scatter(
        #     inputs[:, 0],
        #     inputs[:, 1],
        #     c=colors,
        #     alpha=0.5,
        #     marker="s",
        #     zorder=1,
        # )
        # ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)
        # ax.axis([lo, hi, lo, hi])

        # if cfg.plots.save:
        #     plot_file_path = save_plot(fig, "classification_predictions", run_dir)
        #     log.info(f"Plot saved at path {plot_file_path}")
        # if cfg.plots.show:
        #     plt.show(block=True)

        # def min_max_normalize(tensor):
        #     return (tensor - tensor.max()) / (tensor.max() - tensor.min())
        # x0, x1 = torch.meshgrid(x, x)

        # for i, clf in enumerate(clfs):
        #     logits = clf(normalized_inputs).detach()[:, 0]
        #     # logits_class_0 = min_max_normalize(logits_class_0)

        #     fig, ax = plt.subplots(layout="constrained")
        #     cs = ax.contourf(
        #         x0,
        #         x1,
        #         logits.reshape((len(x), len(x))),
        #         zorder=1,
        #         cmap=LinearSegmentedColormap.from_list("", ["white", "red"]),
        #         norm=plt.Normalize(),
        #     )
        #     plt.colorbar(cs)
        #     # ax.scatter(inputs[:, 0], inputs[:, 1], c=outputs, alpha=0.5, marker="s", zorder=1)
        #     ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)
        #     ax.axis([lo, hi, lo, hi])

        #     # if cfg.plots.save:
        #     plot_file_path = save_plot(fig, f"classification_clf{i}_class_0", run_dir)
        #     log.info(f"Plot saved at path {plot_file_path}")
        #     # if cfg.plots.show:
        #     #     plt.show(block=True)

        #     plt.close()
