def plot_class_2_paths():
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    for i, path in enumerate(paths):
        map_plot(ax, [path])
        x = dataset.normalize_inverse(inputs[i])

        plot_file_path = save_plot(fig, f"path_{x[0]:05.2f}-{x[1]:05.2f}", run_dir)
        log.info(f"Plot saved at path {plot_file_path}")
        plt.cla()
