import pandas as pd

from config import BLD_DATA
from experiments.shared.data.task_download_wine_quality import TaskDownloadWineQuality
from experiments.shared.utils import Task


class TaskPreprocessWineQuality(Task):
    def __init__(self, cfg):
        super(TaskPreprocessWineQuality, self).__init__(cfg, BLD_DATA / "wine_quality")
        task = TaskDownloadWineQuality(cfg)
        self.task_deps = [task]
        self.depends_on = task.produces
        self.produces = BLD_DATA / "wine" / "wine.csv"

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        df_red = pd.read_csv(depends_on["red_wine_quality"], sep=";")
        df_white = pd.read_csv(depends_on["white_wine_quality"], sep=";")
        df = pd.concat([df_red, df_white])

        # Use quality rating as the class
        def class_fn(quality: int):
            if quality == 6:
                return 1
            elif quality > 6:
                return 2
            return 0

        df["quality"] = df.quality.map({k: class_fn(k) for k in range(10 + 1)})

        df.to_csv(produces, index=False)
