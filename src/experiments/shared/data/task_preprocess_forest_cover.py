import pandas as pd

from config import BLD_DATA
from experiments.shared.utils import Task


class TaskPreprocessForestCover(Task):
    def __init__(self, cfg):
        super(TaskPreprocessForestCover, self).__init__(cfg, BLD_DATA / "forest_cover")
        self.depends_on = BLD_DATA / "forest_cover" / "raw_forest_cover.csv"
        self.produces = BLD_DATA / "forest_cover" / "forest_cover.csv"

    @classmethod
    def task_function(cls, depends_on, produces, cfg):
        # Only take certain columns
        columns = {
            "Elevation_m": 0,
            "Aspect_azimuth_deg": 1,
            "Slope_deg": 2,
            "Horizontal_Distance_To_Hydrology_m": 3,
            "Vertical_Distance_To_Hydrology_m": 4,
            "Horizontal_Distance_To_Roadways_m": 5,
            "Hillshade_9am_index": 6,
            "Hillshade_Noon_index": 7,
            "Hillshade_3pm_index": 8,
            "Horizontal_Distance_To_Fire_Points_m": 9,
            "Cover_Type": 54,
        }
        df = pd.read_csv(depends_on, header=None)

        df = df[columns.values()]
        df.columns = columns.keys()
        # 0-based class numbers
        df.iloc[:, -1] -= 1

        df.to_csv(produces, index=False)
