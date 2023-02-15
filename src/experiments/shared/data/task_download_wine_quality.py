import requests

from config import BLD_DATA
from experiments.shared.utils import Task

data_link = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"


class TaskDownloadWineQuality(Task):
    def __init__(self, cfg):
        super(TaskDownloadWineQuality, self).__init__(cfg, BLD_DATA / "wine_quality")
        self.produces = {
            "red_wine_quality": BLD_DATA / "wine" / "raw_red_wine_quality.csv",
            "white_wine_quality": BLD_DATA / "wine" / "raw_white_wine_quality.csv",
        }

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        res_red = requests.get(data_link + "/winequality-red.csv")
        with open(produces["red_wine_quality"], "wb") as f:
            f.write(res_red.content)

        res_white = requests.get(data_link + "/winequality-white.csv")
        with open(produces["white_wine_quality"], "wb") as f:
            f.write(res_white.content)
