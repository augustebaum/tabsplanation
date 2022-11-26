# tabsplanation

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/augustebaum/tabsplanation/main.svg)](https://results.pre-commit.ci/latest/github/augustebaum/tabsplanation/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To get started, create the environment with
```console
$ conda/mamba env create
```
or 
```console
$ poetry install
```

By default, any output is captured by `pytask`. Hence, for training
a model, it is recommended to use `tensorboard`.

In the root of the project, run
```console
tensorboard --log_dir=bld/models/lightning_logs
```
and follow the instructions.

## Rationale

This repository contains various experiments exploring a technique called
Latent Shift, coined by Cohen _et al._ in their paper "Gifsplanation".

The point of `pytask` is to offer an elegant way to make data-related
workflows cacheable, i.e. when each individual part of a workflow
produces outputs, these outputs are saved so they can be re-used
later and the workflow can be skipped.

Previously the same codebase was written with `hydra` in mind: `hydra`
offers a somewhat intuitive system to parametrize an experiment using
a configuration file, usually written in YAML.
In this case an experiment workflow would be written as follows:
```python
# `cfg` contains all configuration information about the workflow
@hydra.main(config="my_config.yaml")
def my_experiment(cfg):
    # Step 1
    data = create_data(cfg)
  
    # Step 2
    if load_models:
        models = load_models(path)
    else:
        models = train_models(cfg, data)
    
    # Step 3
    plot_data = create_plot_data(models, data)
    
    # Step 4
    plot = create_plot(plot_data)
    
    # Step 5
    show_plot(plot)
```
You can see that the "get models" part is cacheable: there is an
option to ask the system to load models from a specific part.
Indeed, of all the steps in the workflow, this step is by far
the most time-consuming. However, the other steps are still
re-run, every time.
What is more, when loading models, the option is currently to
pass a directory path; so when doing this, one must be certain
that the data that was loaded with `get_data` is exactly the
same as the one used to train the loaded models; the book-keeping
has to be done manually, which is error-prone and frustrating.

Instead, it would be saner to divide up the workflow into each
step, and let `pytask` handle the caching:
```python
def task_create_dataset(depends_on, produces):
    # Load config
    cfg = depends_on["config"]

    # Create the dataset according to config
    data = create_dataset(cfg)

    # Cache results
    save(data, produces["data"])
```
and similarly for all the other steps.
Now, instead of getting `hydra` to run `my_experiment`,
you could just ask `pytask` to run the `show_plot` task
according to a configuration file, and let _it_ figure
out what needs to be done to make that happen.
In particular, if you ask it to run `show_plot` two
times with the same config, the second run should
be very quick because everything is cached; hence
you can afford to run the whole workflow even if
it's just to tweak the plot visuals.

## Credits

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter)
and the
[cookiecutter-pytask-project](https://github.com/pytask-dev/cookiecutter-pytask-project)
template.
