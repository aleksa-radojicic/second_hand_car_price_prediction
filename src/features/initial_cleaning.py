import inspect
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import FeaturesInfo
from src.utils import initialize_features_info, preprocess_init


class InitialCleaner:
    CF_PREFIX = "cf_"

    features_info = initialize_features_info()

    @preprocess_init
    def initial_preparation(
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
        df[feature_name] = pd.to_numeric(
            df[feature_name], downcast="unsigned", errors="raise"
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
        df[feature_name] = pd.to_numeric(
            df[feature_name], downcast="unsigned", errors="raise"
        )

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

        # Strip 'km' and spaces and remove '.'  from 'gi_kilometerage'
        df.gi_kilometerage = (
            df.gi_kilometerage.str.rstrip("km").str.replace(".", "").str.strip()
        )

        # Remove '.' and strip spaces from 'gi_production_year'
        df.gi_production_year = df.gi_production_year.str.rstrip(".").str.strip()

        # Strip 'cm3' and spaces from 'gi_engine_capacity'
        df.gi_engine_capacity = df.gi_engine_capacity.str.rstrip("cm3").str.strip()

        # Extract only value of kW (ignore KS which stands for horse powers) and remove spaces
        df.gi_engine_power = (
            df.gi_engine_power.str.split("/", n=1).str.get(0).str.strip()
        )

        # Strip spaces and 'do: ' from 'gi_certified', replace 'Nije atestiran' with NA and transform to datetime
        df.gi_certified = pd.to_datetime(
            df.gi_certified.str.strip()
            .str.lstrip("do: ")
            .replace({"Nije atestiran": np.nan}),
            format="%m.%Y",
            errors="raise",
        )

        # Strip 'kWh' and spaces from 'gi_battery_capacity'
        df.gi_battery_capacity = df.gi_battery_capacity.str.rstrip("kWh").str.strip()

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

        # Convert nominal columns to categorical types (nominal)
        for col in nominal_cols:
            df[col] = pd.Categorical(df[col], ordered=False)

        # Convert numerical columns to numerical types
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors="raise", downcast="unsigned")

        features_info["nominal"].extend(nominal_cols)
        features_info["numerical"].extend(numerical_cols)
        features_info["other"].extend(other_cols)

        pd.set_option("mode.chained_assignment", "warn")

        return df, features_info

    @preprocess_init
    def c_additional_informations(
        self, df: pd.DataFrame, features_info: FeaturesInfo
    ) -> Tuple[pd.DataFrame, FeaturesInfo]:

        pd.set_option("mode.chained_assignment", None)

        # Strip 'Euro' and spaces from 'ai_engine_emission_class'
        df.ai_engine_emission_class = df.ai_engine_emission_class.str.lstrip(
            "Euro"
        ).str.strip()

        # Strip spaces and map 'ai_doors_no' so that True represents 4/5 doors and False 2/3 doors
        df.ai_doors_no = df.ai_doors_no.str.strip().map(
            {"4/5 vrata": True, "2/3 vrata": False}
        )

        # Strip 'sedišta' and spaces from 'ai_seats_no'
        df.ai_seats_no = df.ai_seats_no.str.rstrip("sedišta").str.strip()

        # Keep only cars that have steering wheele on the right side
        df = df.loc[df.ai_steering_wheel_side.str.strip() != "Desni volan", :]

        # Delete 'ai_steering_wheel_side' feature (no longer useful)
        del df["ai_steering_wheel_side"]

        # Strip spaces from 'ai_registered_until', replace 'Nije registrovan' with NA and transform to datetime
        df.ai_registered_until = pd.to_datetime(
            df.ai_registered_until.str.strip().replace({"Nije registrovan": np.nan}),
            format="%m.%Y.",
            errors="raise",
        )

        # Strip spaces and map 'ai_credit' so that True represents 'DA' and False <NA>
        df.ai_credit = df.ai_credit.str.strip().map({"DA": True, np.nan: False})

        # Strip '€' and spaces from 'ai_deposit'
        df.ai_deposit = df.ai_deposit.str.rstrip("€").str.strip()

        # Strip '€' and spaces from 'ai_installment_amount'
        df.ai_installment_amount = df.ai_installment_amount.str.rstrip("€").str.strip()

        # Strip spaces and map 'ai_interest_free_credit' so that True represents 'DA' and False <NA>
        df.ai_interest_free_credit = df.ai_interest_free_credit.str.strip().map(
            {"DA": True, np.nan: False}
        )

        # Strip spaces and map 'ai_leasing' so that True represents 'DA' and False <NA>
        df.ai_leasing = df.ai_leasing.str.strip().map({"DA": True, np.nan: False})

        # Strip '€' and spaces from 'ai_cash_payment'
        df.ai_cash_payment = df.ai_cash_payment.str.rstrip("€").str.strip()

        binary_cols = [
            "ai_doors_no",
            "ai_credit",
            "ai_interest_free_credit",
            "ai_leasing",
        ]
        ordinal_cols = ["ai_engine_emission_class", "ai_damage"]
        nominal_cols = [
            "ai_floating_flywheel",
            "ai_gearbox_type",
            "ai_air_conditioning",
            "ai_color",
            "ai_interior_material",
            "ai_interior_color",
            "ai_propulsion",
            "ai_vehicle_origin",
            "ai_ownership",
            "ai_import_country",
            "ai_sales_method",
        ]
        numerical_cols = [
            "ai_seats_no",
            "ai_deposit",
            "ai_installment_no",
            "ai_installment_amount",
            "ai_cash_payment",
            "ai_range_on_full_battery_km",
        ]
        other_cols = [
            "ai_registered_until",
        ]

        # Convert binary columns to boolean
        df[binary_cols] = df[binary_cols].astype("boolean")

        # Convert ordinal columns to categorical types (ordinal)
        for col in ordinal_cols:
            df[col] = pd.Categorical(df[col], ordered=True)

        # Convert nominal columns to categorical types (nominal)
        for col in nominal_cols:
            df[col] = pd.Categorical(df[col], ordered=False)

        # Convert numerical columns to numerical types
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors="raise", downcast="unsigned")

        features_info["binary"].extend(binary_cols)
        features_info["ordinal"].extend(ordinal_cols)
        features_info["nominal"].extend(nominal_cols)
        features_info["numerical"].extend(numerical_cols)
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
        df, features_info = self.c_general_informations(
            df=df, features_info=features_info
        )
        df, features_info = self.c_additional_informations(
            df=df, features_info=features_info
        )

        return df, features_info

    @preprocess_init
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        features_info = self.features_info

        df, features_info = self.initial_preparation(df=df, features_info=features_info)
        df, features_info = self.clean_individual_columns(
            df=df, features_info=features_info
        )
        self.features_info = features_info
        return df
