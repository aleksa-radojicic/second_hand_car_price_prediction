from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from src import config

from src.config import FeaturesInfo
from src.utils import init_cols_nan_strategy, preprocess_init


class MACleaner:
    CF_PREFIX = "cf_"

    cols_nan_strategy = init_cols_nan_strategy()
    idx_to_remove: List[int] = []

    def __init__(
        self,
        features_info: FeaturesInfo,
        cols_nan_strategy: Dict[str, List[str]],
        idx_to_remove: List[int],
    ):
        self.features_info = features_info
        self.cols_nan_strategy = cols_nan_strategy
        self.idx_to_remove = idx_to_remove

    @preprocess_init
    def ma_irregular_label_rows(
        self,
        df: pd.DataFrame,
        features_info: FeaturesInfo,
        cols_nan_strategy: Dict[str, List[str]],
        idx_to_remove: List[int],
    ) -> Tuple[pd.DataFrame, FeaturesInfo, Dict[str, List[str]], List[int]]:
        df_cars_equal_price_install_amt = df.loc[
            df[config.LABEL] == df.ai_installment_amount, :
        ]

        # Remove cars where 'price' = 'ai_installment_amount'
        df.drop(df_cars_equal_price_install_amt.index, axis=0, inplace=True)

        idx_to_remove.extend(df_cars_equal_price_install_amt.index.tolist())

        return df, features_info, cols_nan_strategy, idx_to_remove

    @preprocess_init
    def ma_low_kilometerage_cars(
        self,
        df: pd.DataFrame,
        features_info: FeaturesInfo,
        cols_nan_strategy: Dict[str, List[str]],
        idx_to_remove: List[int],
    ) -> Tuple[pd.DataFrame, FeaturesInfo, Dict[str, List[str]], List[int]]:

        low_kilometerage_amount = 500
        low_kilometerage_cars = df.loc[
            df["gi_kilometerage"] < low_kilometerage_amount,
            ["name", "short_url", "price", "gi_kilometerage"],
        ]
        # Drop cars where 'gi_kilometerage' < 500
        df.drop(low_kilometerage_cars.index, axis=0, inplace=True)

        idx_to_remove.extend(low_kilometerage_cars.index.tolist())

        return df, features_info, cols_nan_strategy, idx_to_remove

    @preprocess_init
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        features_info = self.features_info
        cols_nan_strategy = self.cols_nan_strategy
        idx_to_remove = self.idx_to_remove

        df, features_info, cols_nan_strategy, idx_to_remove = (
            self.ma_irregular_label_rows(
                df=df,
                features_info=features_info,
                cols_nan_strategy=cols_nan_strategy,
                idx_to_remove=idx_to_remove,
            )
        )
        df, features_info, cols_nan_strategy, idx_to_remove = (
            self.ma_low_kilometerage_cars(
                df=df,
                features_info=features_info,
                cols_nan_strategy=cols_nan_strategy,
                idx_to_remove=idx_to_remove,
            )
        )

        self.features_info = features_info
        self.cols_nan_strategy = cols_nan_strategy
        self.idx_to_remove = idx_to_remove
        return df
