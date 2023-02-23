import sys

import pytask
from omegaconf import OmegaConf

from config import EXPERIMENTS_PATH
from experiments.shared.utils import get_object

config_filename = "config.yaml"
run_path = EXPERIMENTS_PATH / "task_run.py"

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
OmegaConf.resolve(config)

parse_config = get_object(f"experiments.{experiment_name}.parse_config.parse_config")
global tasks_to_collect
tasks_to_collect = parse_config(config)

session = pytask.main({"paths": [run_path], **pytask_options})
