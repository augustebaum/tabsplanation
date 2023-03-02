import os
import sys

import pytask
from omegaconf import OmegaConf

from config import BLD_RUNS, EXPERIMENTS_PATH
from experiments.shared.utils import get_object, get_time, write


if __name__ == "__main__":
    config_filename = "config.yaml"

    pytask_options = {
        # "verbose": 2,
        "s": True,
        "pdb": True,
        "pdbcls": ("IPython.terminal.debugger", "TerminalPdb"),
    }

    experiment_name = sys.argv[1]
    pytask_args = sys.argv[2:]

    experiment_path = EXPERIMENTS_PATH / experiment_name

    config_path = experiment_path / config_filename
    config = OmegaConf.load(config_path)

    def write_config(config, path):
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "w") as f:
            OmegaConf.save(config, f)

    run_dir = BLD_RUNS / get_time()
    write_config(config, run_dir / "config.yaml")
    OmegaConf.resolve(config)
    write_config(config, run_dir / "full_config.yaml")

    parse_config = get_object(
        f"experiments.{experiment_name}.parse_config.parse_config"
    )
    tasks_to_collect = parse_config(config)

    write(tasks_to_collect, run_dir / "tasks.pkl")

    session = pytask.main(
        {"paths": [EXPERIMENTS_PATH / "task_run.py"], **pytask_options}
    )
