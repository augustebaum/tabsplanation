import gzip

import pytask
import requests

from config import BLD_DATA

data_file = BLD_DATA / "forest_cover" / "raw_forest_cover.csv"
data_link = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)


@pytask.mark.depends_on(data_link)
@pytask.mark.produces(data_file)
def task_download_forest_cover(depends_on, produces):
    res = requests.get(depends_on)
    with open(produces, "wb") as f:
        f.write(gzip.decompress(res.content))
