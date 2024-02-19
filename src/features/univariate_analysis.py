import inspect
from typing import Tuple

import pandas as pd

from src.config import FeaturesInfo
from src.utils import initialize_features_info, preprocess_init


class UACleaner:
    CF_PREFIX = "cf_"

    features_info = initialize_features_info()

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
            df[feature_name].str.slice(stop=-1).str.replace(".", ""),
            errors="raise",
            downcast="unsigned",
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
            df[feature_name], downcast="unsigned"
        ).isna().sum():
            df[feature_name] = pd.to_numeric(df[feature_name], downcast="unsigned")
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

        # Convert 'location' to categorical type (nominal)
        df[feature_name] = pd.Categorical(df[feature_name], ordered=False)

        # Add 'location' to 'nominal' features
        features_info["nominal"].append(feature_name)

        return df, features_info

    @preprocess_init
    def cf_images_no(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Transformed to numerical
        df[feature_name] = pd.to_numeric(df[feature_name], downcast="unsigned")

        # Add 'images_no' to 'numerical' features
        features_info["numerical"].append(feature_name)

        return df, features_info

    @preprocess_init
    def cf_safety(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()
        prefix = "s_"

        # Create data frame with dummy columns
        df_safety_dummies = df[feature_name].str.get_dummies(sep=",").add_prefix(prefix)
        # Extend the data frame with dummy columns
        df = pd.concat([df, df_safety_dummies], axis=1)

        # Delete 'safety' column
        del df[feature_name]

        safety_columns = df_safety_dummies.columns.tolist()

        # Convert all remaining safety columns to boolean
        df[safety_columns] = df[safety_columns].astype("boolean")

        # Add all remaining safety columns to 'binary' features
        features_info["binary"].extend(safety_columns)

        return df, features_info

    @preprocess_init
    def cf_equipment(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()
        prefix = "e_"

        # Create data frame with dummy columns
        df_equipment_dummies = (
            df[feature_name].str.get_dummies(sep=",").add_prefix(prefix)
        )
        # Extend the data frame with dummy columns
        df = pd.concat([df, df_equipment_dummies], axis=1)

        # Delete 'equipment' column
        del df[feature_name]

        equipment_columns = df_equipment_dummies.columns.tolist()

        # Convert all remaining equipment columns to boolean
        df[equipment_columns] = df[equipment_columns].astype("boolean")

        # Add all remaining equipment columns to 'binary' features
        features_info["binary"].extend(equipment_columns)

        return df, features_info

    @preprocess_init
    def cf_other(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()
        prefix = "o_"

        # Create data frame with dummy columns
        df_other_dummies = df[feature_name].str.get_dummies(sep=",").add_prefix(prefix)
        # Extend the data frame with dummy columns
        df = pd.concat([df, df_other_dummies], axis=1)

        # Delete 'other' column
        del df[feature_name]

        other_columns = df_other_dummies.columns.tolist()

        # Convert all remaining other columns to boolean
        df[other_columns] = df[other_columns].astype("boolean")

        # Add all other columns to 'binary' features
        features_info["binary"].extend(other_columns)

        return df, features_info

    @preprocess_init
    def cf_description(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        feature_name = self._get_feature_name()

        # Add 'description' to 'other' features
        features_info["other"].append(feature_name)

        return df, features_info

    @preprocess_init
    def c_general_informations(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:
        pd.set_option("mode.chained_assignment", None)

        # Delete new cars
        new_cars_cond = df.gi_condition == "Novo vozilo"
        df = df.loc[~new_cars_cond, :]

        # Delete 'gi_condition' feature
        del df["gi_condition"]

        # Strip 'km', remove '.' and convert 'gi_kilometerage' to numerical
        df.gi_kilometerage = pd.to_numeric(
            df.gi_kilometerage.str.rstrip("km").str.replace(".", ""),
            downcast="unsigned",
        )

        # Remove '.' and convert 'gi_production_year' to numerical
        df.gi_production_year = pd.to_numeric(
            df.gi_production_year.str.rstrip("."), downcast="unsigned"
        )

        # Strip 'cm3' and convert 'gi_engine_capacity' to numerical
        df.gi_engine_capacity = pd.to_numeric(
            df.gi_engine_capacity.str.rstrip("cm3"), errors="raise", downcast="unsigned"
        )

        # Extract only value of kW (ignore KS which stands for horse powers)
        df.gi_engine_power = pd.to_numeric(
            df.gi_engine_power.str.split("/", n=1).str.get(0), downcast="unsigned"
        )

        # Strip 'kWh' and convert 'gi_battery_capacity' to numerical
        df.gi_battery_capacity = pd.to_numeric(
            df.gi_battery_capacity.str.rstrip("kWh"), downcast="unsigned"
        )

        nominal_cols = [
            "gi_brand",
            "gi_model",
            "gi_body_type",
            "gi_fuel_type",
            "gi_trade_in",
        ]
        numerical_cols = [
            "gi_kilometerage",
            "gi_production_year",
            "gi_engine_capacity",
            "gi_engine_power",
            "gi_battery_capacity",
        ]
        other_cols = ["gi_certified"]

        # Convert nominal columns to categorical type (nominal)
        for col in nominal_cols:
            df[col] = pd.Categorical(df[col], ordered=False)

        # Add nominal_cols columns to 'nominal' features
        features_info["nominal"].extend(nominal_cols)

        # Add numerical_cols columns to 'numerical' features
        features_info["numerical"].extend(numerical_cols)

        # Add other_cols columns to 'other' features
        features_info["other"].extend(other_cols)

        pd.set_option("mode.chained_assignment", "warn")

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
        df, features_info = self.cf_safety(df=df, features_info=features_info)
        df, features_info = self.cf_equipment(df=df, features_info=features_info)
        df, features_info = self.cf_other(df=df, features_info=features_info)
        df, features_info = self.cf_description(df=df, features_info=features_info)
        df, features_info = self.c_general_informations(df=df, features_info=features_info)

        return df, features_info

    @preprocess_init
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        features_info = self.features_info
        
        df, features_info = self.initial_clean(df=df, features_info=features_info)
        df, features_info = self.clean_individual_columns(
            df=df, features_info=features_info
        )
        self.features_info = features_info
        return df
