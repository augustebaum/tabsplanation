import pandas as pd
import pytask

from config import BLD_DATA


@pytask.mark.depends_on(BLD_DATA / "forest_cover" / "raw_forest_cover.csv")
@pytask.mark.produces(BLD_DATA / "forest_cover" / "forest_cover.csv")
def task_preprocess_forest_cover(depends_on, produces):
    df = pd.read_csv(depends_on)
    import pdb

    pdb.set_trace()
