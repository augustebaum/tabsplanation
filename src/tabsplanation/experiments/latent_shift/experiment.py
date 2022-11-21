"""An experiment to compare different types of AEs in terms of
fitness for the latent shift CF generation technique.

The result of this experiment is a grid with all possible combinations
of auto-encoders and classifiers as desired; each plot shows how
the perturbed examples achieve a different prediction and whether
they are far away from the original prediction, depending on the
relative latent shift.

As the latent shift moves away from zero,
we expect the prediction probability to increase with relative latent shift,
because we are following the gradient until the probability increases/decreases.
"""

import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tabsplanation.explanations import make_path
from ..experiment import get_data, load_models, save_plot, train_aes, train_clfs
from ..utils import setup
from .plot import plot_experiment

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../../config", config_name="latent_shift_config"
)
def latent_shift(cfg: DictConfig) -> None:

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(cfg)}")

    device, run_dir, outputs_dir = setup(cfg.seed)
    log.info("Finished setup")

    dataset, subsets, loaders = get_data(cfg, device)
    log.info("Finished loading data")

    # Train or load autoencoders
    if cfg.load_autoencoders_from:
        loaded_run_dir = outputs_dir / cfg.load_autoencoders_from / "aes"
        aes = load_models(loaded_run_dir, log)
    else:
        aes, aes_dir = train_aes(cfg, device, run_dir, loaders)
        log.info(f"Saving model information in {aes_dir}")

    # Train or load classifiers
    if cfg.load_classifiers_from:
        loaded_run_dir = outputs_dir / cfg.load_classifiers_from / "clfs"
        clfs = load_models(loaded_run_dir, log)
    else:
        clfs, clfs_dir = train_clfs(cfg, device, run_dir, loaders)

    # inputs, _ = subsets["test"][:2]

    inputs_denorm = []

    # Cover class 2 (4 corners and middle)
    # margin = 2
    # inputs_denorm.append([40.0, 40.0])
    # inputs_denorm.append([35.0 + margin, 35.0 + margin])
    # inputs_denorm.append([35.0 + margin, 45.0 - margin])
    # inputs_denorm.append([45.0 - margin, 35.0 + margin])
    # inputs_denorm.append([45.0 - margin, 45.0 - margin])

    inputs_denorm = np.c_[np.linspace(37, 43), np.ones(50) * 43]

    # In class 0, right under dead zone
    # inputs_denorm.append([40.0, 25.0])
    # inputs_denorm.append([40.0, 20.0])
    # inputs_denorm.append([40.0, 15.0])
    # inputs_denorm.append([40.0, 10.0])
    # inputs_denorm.append([40.0, 5.0])

    # inputs_denorm = np.c_[np.ones(50) * 40, np.linspace(0, 25)]

    inputs = dataset.normalize(torch.tensor(inputs_denorm).to(torch.float))
    print(inputs)

    ae = [ae for ae in aes if ae.model_name == "VAE"][0]
    clf = clfs[0]

    target_map = {0: 2, 1: None, 2: 0}

    def get_target_class(input):
        return target_map[np.argmax(clf.softmax(input).detach()).item()]

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

    # path_xs = dataset.normalize_inverse(path.xs)

    # for path in paths:
    #     assert path.original_class == 2
    #     assert path.target_class == 0
    #     ax.plot(path.shifts, path.prbs_new, color="b")
    #     ax.plot(path.shifts, path.prbs_old, color="r")

    # plt.show()

    # torch.set_printoptions(precision=3, sci_mode=False)
    # print("xs:", path_xs)
    # print("ys:", path.ys)
    # print("lambdas:", path.shifts)

    # fig, ax = plt.subplots()
    from .plot import map_plot

    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    for i, path in enumerate(paths):
        map_plot(ax, [path])
        x = dataset.normalize_inverse(inputs[i])

        plot_file_path = save_plot(fig, f"path_{x[0]:05.2f}-{x[1]:05.2f}", run_dir)
        log.info(f"Plot saved at path {plot_file_path}")
        plt.cla()
    # plt.show()

    # # fig = plot_experiment(
    # #     inputs, target_map, aes, clfs, dataset.normalize_inverse
    # # )

    # if cfg.plots.save:
    #     plot_file_path = save_plot(fig, "latent_shift", run_dir)
    #     log.info(f"Plot saved at path {plot_file_path}")
    # if cfg.plots.show:
    #     plt.show(block=True)
