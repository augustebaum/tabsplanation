import pandas as pd

from config import BLD_DATA
from experiments.shared.data.task_download_online_news_popularity import (
    TaskDownloadOnlineNewsPopularity,
)
from experiments.shared.utils import Task


class TaskPreprocessOnlineNewsPopularity(Task):
    def __init__(self, cfg):
        super(TaskPreprocessOnlineNewsPopularity, self).__init__(
            cfg, BLD_DATA / "online_news_popularity"
        )
        self.task_deps = [TaskDownloadOnlineNewsPopularity(None)]
        self.depends_on = (
            BLD_DATA / "online_news_popularity" / "raw_online_news_popularity.csv"
        )
        self.produces = (
            BLD_DATA / "online_news_popularity" / "online_news_popularity.csv"
        )

    @classmethod
    def task_function(cls, depends_on, produces, cfg):

        column_names = [
            "url",
            "timedelta",
            "n_tokens_title",
            "n_tokens_content",
            "n_unique_tokens",
            "n_non_stop_words",
            "n_non_stop_unique_tokens",
            "num_hrefs",
            "num_self_hrefs",
            "num_imgs",
            "num_videos",
            "average_token_length",
            "num_keywords",
            "data_channel_is_lifestyle",
            "data_channel_is_entertainment",
            "data_channel_is_bus",
            "data_channel_is_socmed",
            "data_channel_is_tech",
            "data_channel_is_world",
            "kw_min_min",
            "kw_max_min",
            "kw_avg_min",
            "kw_min_max",
            "kw_max_max",
            "kw_avg_max",
            "kw_min_avg",
            "kw_max_avg",
            "kw_avg_avg",
            "self_reference_min_shares",
            "self_reference_max_shares",
            "self_reference_avg_shares",
            "weekday_is_monday",
            "weekday_is_tuesday",
            "weekday_is_wednesday",
            "weekday_is_thursday",
            "weekday_is_friday",
            "weekday_is_saturday",
            "weekday_is_sunday",
            "is_weekend",
            "LDA_00",
            "LDA_01",
            "LDA_02",
            "LDA_03",
            "LDA_04",
            "global_subjectivity",
            "global_sentiment_polarity",
            "global_rate_positive_words",
            "global_rate_negative_words",
            "rate_positive_words",
            "rate_negative_words",
            "avg_positive_polarity",
            "min_positive_polarity",
            "max_positive_polarity",
            "avg_negative_polarity",
            "min_negative_polarity",
            "max_negative_polarity",
            "title_subjectivity",
            "title_sentiment_polarity",
            "abs_title_subjectivity",
            "abs_title_sentiment_polarity",
            "shares",
        ]

        df = pd.read_csv(depends_on, header=0, names=column_names)

        columns_to_remove = [
            "url",
            "data_channel_is_lifestyle",
            "data_channel_is_entertainment",
            "data_channel_is_bus",
            "data_channel_is_socmed",
            "data_channel_is_tech",
            "data_channel_is_world",
            "weekday_is_monday",
            "weekday_is_tuesday",
            "weekday_is_wednesday",
            "weekday_is_thursday",
            "weekday_is_friday",
            "weekday_is_saturday",
            "weekday_is_sunday",
            "is_weekend",
        ]
        # Only take certain columns
        for col in columns_to_remove:
            column_names.remove(col)

        df = df[column_names]

        def percentile(percentage):
            def _quantile(x):
                return x.quantile(percentage / 100)

            return _quantile

        def get_shares_percentiles(percentiles):
            aggs = map(percentile, percentiles)
            percentiles_series = df.shares.agg(aggs)
            return list(percentiles_series)

        percentiles = [0, 50, 75, 95, 100]
        shares_percentiles = get_shares_percentiles(percentiles)

        import pdb

        pdb.set_trace()
        # Use thresholds on number of shares as the class
        df["popularity"] = pd.cut(
            df.shares, shares_percentiles, labels=range(len(percentiles) - 1)
        ).fillna(0)
        del df["shares"]

        df.to_csv(produces, index=False)
