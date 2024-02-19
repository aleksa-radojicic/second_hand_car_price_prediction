import inspect
from typing import Tuple

import pandas as pd

from src.config import FeaturesInfo
from src.utils import preprocess_init


class UACleaner:
    CF_PREFIX = "cf_"

    features_info: FeaturesInfo

    def __init__(self, features_info: FeaturesInfo = None):
        self.features_info = features_info

    @preprocess_init
    def initial_clean(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        # Transform column type to string
        df.columns = df.columns.astype("string")

        # Prefix columns from table general_informations with "gi_"
        # and additional_informations with "ai"
        id_1_col_idx = df.columns.get_loc("id_1")
        id_2_col_idx = df.columns.get_loc("id_2")

        columns_from_gi = df.columns[id_1_col_idx + 1 : id_2_col_idx].values
        columns_from_ai = df.columns[id_2_col_idx + 1 :].values

        # Add prefix 'gi_' to columns from table general_informations
        df.rename(
            columns=dict(zip(columns_from_gi, "gi_" + columns_from_gi)), inplace=True
        )
        # Add prefix 'ai_' to columns from table additional_informations
        df.rename(
            columns=dict(zip(columns_from_ai, "ai_" + columns_from_ai)), inplace=True
        )

        # Remove redundant ids
        del df["id_1"], df["id_2"]

        # Remove gi_fixed_price that is poorly scraped
        del df["gi_fixed_price"]

        return df, features_info

    def _get_feature_name(self) -> str:
        """Returns name of the feature from the function that called this one."""
        function_name = inspect.stack()[1].function
        feature_name = function_name[len(self.CF_PREFIX) :]

        return feature_name

    @preprocess_init
    def cf_name(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Add 'name' to 'other' features
        features_info["other"].append(feature_name)

        return df, features_info

    @preprocess_init
    def cf_short_url(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Add 'short_url' to 'other' features
        features_info["other"].append(feature_name)

        return df, features_info

    @preprocess_init
    def cf_price(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Remove '.' from values and transform to numerical
        df[feature_name] = pd.to_numeric(
            df[feature_name].str.slice(stop=-1).str.replace(".", ""), errors="raise"
        )

        # Remove cars that had price = 1
        df = df[df[feature_name] != 1]

        cars_price_less_than_100 = df.loc[df[feature_name] < 100, feature_name]
        # Remove cars that had price < 100
        df = df.drop(cars_price_less_than_100.index, axis=0)

        return df, features_info

    @preprocess_init
    def cf_listing_followers_no(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Transform to numerical
        if (df[feature_name] == "").sum() == pd.to_numeric(
            df[feature_name]
        ).isna().sum():
            df[feature_name] = pd.to_numeric(df[feature_name])
        else:
            raise ValueError(
                "There is a listing_followers_no value that is probably incorrectly parsed."
            )

        # Added 'listing_followers_no' to numerical features
        features_info["numerical"].append(feature_name)

        return df, features_info

    @preprocess_init
    def cf_location(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Add 'location' to 'nominal' features
        features_info["nominal"].append(feature_name)

        return df, features_info

    @preprocess_init
    def cf_images_no(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Transformed to numerical
        df[feature_name] = pd.to_numeric(df[feature_name])

        # Add 'images_no' to 'numerical' features
        features_info["numerical"].append(feature_name)

        return df, features_info

    @preprocess_init
    def clean_individual_columns(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        df, features_info = self.cf_name(df=df, features_info=features_info)
        df, features_info = self.cf_short_url(df=df, features_info=features_info)
        df, features_info = self.cf_price(df=df, features_info=features_info)
        df, features_info = self.cf_listing_followers_no(
            df=df, features_info=features_info
        )
        df, features_info = self.cf_location(df=df, features_info=features_info)
        df, features_info = self.cf_images_no(df=df, features_info=features_info)

        return df, features_info

    @preprocess_init
    def clean(self, df: pd.DataFrame, features_info: FeaturesInfo) -> pd.DataFrame:
        df, features_info = self.initial_clean(df=df, features_info=features_info)
        df, features_info = self.clean_individual_columns(
            df=df, features_info=features_info
        )
        self.features_info = features_info
        return df
