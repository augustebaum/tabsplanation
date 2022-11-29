"""Training and testing a classifier."""

import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from omegaconf import DictConfig, OmegaConf

from experiments.experiment import get_data, load_models, save_plot, train_clfs
from experiments.latent_shift.plot import get_map_img
from experiments.utils import setup
from tabsplanation.types import Tensor


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../config", config_name="classification_config"
)
def classification(cfg: DictConfig) -> None:

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(cfg)}")

    device, run_dir, outputs_dir = setup(cfg.seed)
    log.info("Finished setup")
    log.info(f"Storing in {run_dir}")

    dataset, subsets, loaders = get_data(cfg, device)
    log.info("Finished loading data")

    # Train or load classifiers
    if cfg.load_classifiers_from:
        loaded_run_dir = outputs_dir / cfg.load_classifiers_from / "clfs"
        clfs = load_models(loaded_run_dir, log)
    else:
        clfs, clfs_dir = train_clfs(cfg, device, run_dir, loaders)

    # Plot
    # inputs, _ = subsets["test"][:]

    clf = clfs[0]

    margin = 20
    lo = 0 - margin
    hi = 50 + margin

    # Set all correlated columns to their mean, and make the first two
    # dimensions trace a grid from lo to hi
    x = torch.linspace(lo, hi, steps=50)
    inputs_x: Tensor["nb_points", 2] = torch.cartesian_prod(x, x)

    inputs = dataset.fill_from_2d_point(inputs_x)
    # means: Tensor[1, "input_dim"] = dataset.normalize.mean
    # inputs: Tensor["nb_points", "input_dim"] = means.repeat(len(inputs_x), 1)
    # inputs[:, [0, 1]] = inputs_x

    # The first two columns are normalized grid, everything else is zero
    normalized_inputs = dataset.normalize(inputs)

    outputs = clf.softmax(normalized_inputs).detach()

    fig, ax = plt.subplots(layout="constrained")

    ax.scatter(inputs[:, 0], inputs[:, 1], c=outputs, alpha=0.5, marker="s", zorder=1)
    ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)
    ax.axis([lo, hi, lo, hi])

    if cfg.plots.save:
        plot_file_path = save_plot(fig, "classification_probas", run_dir)
        log.info(f"Plot saved at path {plot_file_path}")
    if cfg.plots.show:
        plt.show(block=True)

    # plt.cla()

    # Plot of logits
    # # def min_max_normalize(tensor):
    # #     return (tensor - tensor.max()) / (tensor.max() - tensor.min())
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

    #     if cfg.plots.save:
    #         plot_file_path = save_plot(fig, f"classification_clf{i}_class_0", run_dir)
    #         log.info(f"Plot saved at path {plot_file_path}")
    #     if cfg.plots.show:
    #         plt.show(block=True)

    #     plt.close()
