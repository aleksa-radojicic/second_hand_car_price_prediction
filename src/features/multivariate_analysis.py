from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src import config
from src.logger import log_message
from src.utils import (Dataset, Metadata, PipelineMetadata,
                       log_feature_info_dict, preprocess_init)


class MACleaner:
    CF_PREFIX: str = "cf_"
    pipe_meta: PipelineMetadata
    cached_metadata: Optional[Metadata]
    verbose: int

    def __init__(self, pipe_meta: PipelineMetadata, verbose: int = 0):
        self.pipe_meta = pipe_meta
        self.cached_metadata = None
        self.verbose = verbose

    @property
    def metadata(self) -> Metadata:
        return self.pipe_meta.data

    @metadata.setter
    def metadata(self, metadata):
        self.pipe_meta.data = metadata

    @staticmethod
    @preprocess_init
    def ma_irregular_label_rows(
        df: Dataset, metadata: Metadata
    ) -> Tuple[Dataset, Metadata]:
        idx_to_remove = metadata.idx_to_remove

        df_cars_equal_price_install_amt = df.loc[
            df[config.LABEL] == df.ai_installment_amount, :
        ]

        # Remove cars where 'price' = 'ai_installment_amount'
        df.drop(df_cars_equal_price_install_amt.index, axis=0, inplace=True)

        idx_to_remove.extend(df_cars_equal_price_install_amt.index.tolist())

        return df, metadata

    @staticmethod
    @preprocess_init
    def ma_low_kilometerage_cars(
        df: Dataset, metadata: Metadata
    ) -> Tuple[Dataset, Metadata]:
        idx_to_remove = metadata.idx_to_remove

        low_kilometerage_amount = 500
        low_kilometerage_cars = df.loc[
            df["gi_kilometerage"] < low_kilometerage_amount,
            ["name", "short_url", "price", "gi_kilometerage"],
        ]
        # Drop cars where 'gi_kilometerage' < 500
        df.drop(low_kilometerage_cars.index, axis=0, inplace=True)

        idx_to_remove.extend(low_kilometerage_cars.index.tolist())

        return df, metadata

    @staticmethod
    @preprocess_init
    def ma_high_seats_cars(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        idx_to_remove = metadata.idx_to_remove

        high_seats_no = 5
        high_seats = df.loc[df.ai_seats_no > high_seats_no, ["name", "ai_seats_no"]]

        extreme_high_seats_no = 7
        more_than_7_seats = high_seats[high_seats.ai_seats_no > extreme_high_seats_no]

        # Drop cars where 'ai_seats_no' > {extreme_high_seats_no}
        df.drop(more_than_7_seats.index, axis=0, inplace=True)

        idx_to_remove.extend(more_than_7_seats.index.tolist())

        return df, metadata

    @staticmethod
    @preprocess_init
    def ma_oldtimers(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        idx_to_remove = metadata.idx_to_remove

        oldtimers = df.loc[
            df.o_Oldtimer,
            [
                "name",
                "gi_production_year",
                "price",
                "short_url",
                "gi_kilometerage",
                "gi_engine_capacity",
                "gi_engine_power",
                "ai_seats_no",
                "o_Restauriran",
            ],
        ]

        seats_no = 2
        oldtimer_2_seats = oldtimers[oldtimers.ai_seats_no == seats_no]

        # Drop oldtimer cars with 'ai_seats_no' = {seats_no}
        df.drop(oldtimer_2_seats.index, inplace=True)

        idx_to_remove.extend(oldtimer_2_seats.index.tolist())

        return df, metadata

    @staticmethod
    @preprocess_init
    def ma_finalize(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy

        features_info["features_to_delete"].remove("gi_battery_capacity")
        features_info["features_to_delete"].remove("ai_range_on_full_battery_km")

        cols_nan_strategy["const_0"].extend(
            ["gi_battery_capacity", "ai_range_on_full_battery_km"]
        )

        # Exclude nominal features that have too big cardinality at the moment
        nominal_to_delete = ["location", "gi_brand", "gi_model"]
        features_info["features_to_delete"].extend(nominal_to_delete)

        return df, metadata

    @staticmethod
    @preprocess_init
    def clean(df: Dataset, metadata=metadata) -> Tuple[Dataset, Metadata]:
        df, metadata = MACleaner.ma_irregular_label_rows(df=df, metadata=metadata)
        df, metadata = MACleaner.ma_low_kilometerage_cars(df=df, metadata=metadata)
        df, metadata = MACleaner.ma_high_seats_cars(df=df, metadata=metadata)
        df, metadata = MACleaner.ma_oldtimers(df=df, metadata=metadata)
        df, metadata = MACleaner.ma_finalize(df=df, metadata=metadata)

        return df, metadata

    @preprocess_init
    def start(self, df: Dataset, y=None) -> Dataset:
        if not self.cached_metadata:
            self.cached_metadata = self.metadata

        metadata = self.cached_metadata

        df, metadata = MACleaner.clean(df, metadata)

        log_feature_info_dict(
            metadata.features_info, "adding data type prefix to columns", self.verbose
        )

        log_message("Added data type prefix to columns successfully.", self.verbose)

        self.metadata = metadata
        return df
