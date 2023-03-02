import io
import zipfile

import requests

from config import BLD_DATA

from experiments.shared.utils import Task

data_file = BLD_DATA / "online_news_popularity" / "raw_online_news_popularity.csv"
data_link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"


class TaskDownloadOnlineNewsPopularity(Task):
    def __init__(self, cfg):
        super(TaskDownloadOnlineNewsPopularity, self).__init__(cfg, "")
        self.produces = data_file

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        res = requests.get(data_link)
        # Open zip file
        with zipfile.ZipFile(io.BytesIO(res.content), "r") as zip:
            # Open csv file within zip
            csv_filename = [
                filename for filename in zip.namelist() if filename.endswith(".csv")
            ][0]
            with zip.open(csv_filename) as csv_file:
                # write csv to produces
                with open(produces, "wb") as f:
                    f.write(csv_file.read())
