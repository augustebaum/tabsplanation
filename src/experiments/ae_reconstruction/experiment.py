"""Compare different types of AEs in terms of reconstruction error.

For a given `x`, we measure the distance between `x` and `decode(encode(x))`,
but also `x` and `decode(encode(decode(encode(x))))`, etc.

This tells us whether the AE is stable and good at reconstructing points
that _it made itself_, as well as if it's good at reconstructing random points.
"""
import logging

import hydra
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf

from ..experiment import get_data, load_models, save_plot, train_aes
from ..utils import setup
from .plot import plot_roundtrip_test


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="ae_reconstruction_config",
)
def ae_reconstruction(cfg: DictConfig) -> Figure:

    log.info(f"Running experiment with config:\n{OmegaConf.to_yaml(cfg)}")

    device, run_dir, outputs_dir = setup(cfg.seed)
    # log.info("Finished setup")

    dataset, subsets, loaders = get_data(cfg, device)
    # log.info("Finished loading data")

    if cfg.load_autoencoders_from:
        loaded_models_dir = outputs_dir / cfg.load_autoencoders_from / "aes"
        # log.info(f"Loading models from {loaded_models_dir}")
        aes = load_models(loaded_models_dir, log)
    else:
        aes, models_dir = train_aes(cfg, device, run_dir, loaders)
        # log.info(f"Saving model information in {models_dir}")

    # Plot roundtrip for various points
    margin = 0
    lo = 0 - margin
    hi = 50 + margin

    x = torch.linspace(lo, hi, steps=50)
    inputs = torch.cartesian_prod(x, x)

    # ae = [ae for ae in aes if ae.model_name == "VAE"][0]
    for ae in aes:
        outputs = dataset.normalize_inverse(
            ae.decode(ae.encode(dataset.normalize(inputs))).detach()
        )

        fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")

        ax.scatter(inputs[:, 0], inputs[:, 1], color="b", s=1, zorder=2)
        ax.scatter(outputs[:, 0], outputs[:, 1], color="r", s=1, zorder=2)

        from matplotlib.collections import LineCollection

        ax.add_collection(
            LineCollection(
                torch.stack([inputs, outputs], dim=1), linewidths=1, zorder=1
            )
        )

        from ..latent_shift.plot import get_map_img

        ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=3)

        ax.axis([lo, hi, lo, hi])

        plot_file_path = save_plot(fig, f"ae_map_{ae.model_name}", run_dir)
        log.info(f"Plot saved at path {plot_file_path}")
        # plt.show()

    # # Roundtrip test boxplot
    # inputs, _ = subsets["test"][:]
    # fig = plot_roundtrip_test(inputs, aes, cfg.nb_roundtrips, dataset.normalize_inverse)

    # if cfg.plots.save:
    #     plot_file_path = save_plot(fig, "ae_reconstruction_boxplot", run_dir)
    #     log.info(f"Plot saved at path {plot_file_path}")

    # # Visual roundtrip test (just see if points are mapped back to [0, 50])
    # inputs, _ = subsets["test"][:]

    # outputs = [ae.decode(ae.encode(inputs)).detach() for ae in aes]
    # # print([ae.encode(inputs).detach() for ae in aes])
    # outputs = [dataset.normalize_inverse(output) for output in outputs]

    # inputs = dataset.normalize_inverse(inputs)

    # ae_names = [ae.model_name for ae in aes]

    # fig, ax = plt.subplots(nrows=1, ncols=len(aes), squeeze=False)
    # for col, name in enumerate(ae_names):
    #     ax[0, col].scatter(outputs[col][:, 0], outputs[col][:, 1], s=0.2)
    #     ax[0, col].axis([0, 50, 0, 50])
    #     ax[0, col].set_title(name)

    # if cfg.plots.save:
    #     plot_file_path = save_plot(fig, "ae_reconstruction_visual", run_dir)
    #     log.info(f"Plot saved at path {plot_file_path}")

    # if cfg.plots.show:
    #     plt.show(block=True)
