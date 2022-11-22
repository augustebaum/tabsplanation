@pytask.mark.depends_on(
    {
        "model": "model.pt",
        "data": {"a": SRC / "data" / "a.pkl", "b": SRC / "data" / "b.pkl"},
    }
)
def task_create_plot_data():
    pl.seed_everything(cfg.seed, workers=True)
    clf = clfs[0]

    margin = 20
    lo = 0 - margin
    hi = 50 + margin

    # Set all correlated columns to their mean, and make the first two
    # dimensions trace a grid from lo to hi
    x = torch.linspace(lo, hi, steps=50)
    inputs_x: Tensor["nb_points", 2] = torch.cartesian_prod(x, x)
    means: Tensor[1, "input_dim"] = dataset.normalize.mean
    inputs: Tensor["nb_points", "input_dim"] = means.repeat(len(inputs_x), 1)
    inputs[:, [0, 1]] = inputs_x

    # The first two columns are normalized grid, everything else is zero
    normalized_inputs = dataset.normalize(inputs)

    outputs = clf.softmax(normalized_inputs).detach()

    torch.save(produces["inputs"], inputs)
    torch.save(produces["outputs"], outputs)
