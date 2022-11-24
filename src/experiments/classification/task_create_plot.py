def task_create_plot(depends_on, produces):

    set_matplotlib_style()
    fig, ax = plt.subplots(layout="constrained")

    ax.scatter(inputs[:, 0], inputs[:, 1], c=outputs, alpha=0.5, marker="s", zorder=1)
    ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)
    ax.axis([lo, hi, lo, hi])

    if cfg.plots.save:
        plot_file_path = save_plot(fig, "classification_probas", run_dir)
        log.info(f"Plot saved at path {plot_file_path}")
    if cfg.plots.show:
        plt.show(block=True)

    plt.cla()

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
    x0, x1 = torch.meshgrid(x, x)

    for i, clf in enumerate(clfs):
        logits = clf(normalized_inputs).detach()[:, 0]
        # logits_class_0 = min_max_normalize(logits_class_0)

        fig, ax = plt.subplots(layout="constrained")
        cs = ax.contourf(
            x0,
            x1,
            logits.reshape((len(x), len(x))),
            zorder=1,
            cmap=LinearSegmentedColormap.from_list("", ["white", "red"]),
            norm=plt.Normalize(),
        )
        plt.colorbar(cs)
        # ax.scatter(inputs[:, 0], inputs[:, 1], c=outputs, alpha=0.5, marker="s", zorder=1)
        ax.imshow(get_map_img(), origin="upper", extent=[0, 50, 0, 50], zorder=2)
        ax.axis([lo, hi, lo, hi])

        # if cfg.plots.save:
        plot_file_path = save_plot(fig, f"classification_clf{i}_class_0", run_dir)
        log.info(f"Plot saved at path {plot_file_path}")
        # if cfg.plots.show:
        #     plt.show(block=True)

        plt.close()
