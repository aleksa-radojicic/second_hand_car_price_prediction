import inspect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.utils import CustomTransformer
from src.utils import Dataset, Metadata, preprocess_init

CF_PREFIX: str = "cf_"


@dataclass
class InitialCleanerConfig:
    oldtimers_flag: bool = True
    high_seats_cars_flag: bool = True
    low_kilometerage_cars_flag: bool = True


class InitialCleaner(CustomTransformer):
    oldtimers_flag: bool
    high_seats_cars_flag: bool
    low_kilometerage_cars_flag: bool

    def __init__(
        self,
        oldtimers_flag: bool = True,
        high_seats_cars_flag: bool = True,
        low_kilometerage_cars_flag: bool = True,
    ):
        super().__init__()
        self.oldtimers_flag = oldtimers_flag
        self.high_seats_cars_flag = high_seats_cars_flag
        self.low_kilometerage_cars_flag = low_kilometerage_cars_flag

    @staticmethod
    @preprocess_init
    def initial_preparation(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        # Transform column type to string
        df.columns = df.columns.astype("string")

        # Prefix columns from table general_informations with "gi_"
        # and additional_informations with "ai"
        id_1_col_idx = df.columns.get_loc("id_1")
        id_2_col_idx = df.columns.get_loc("id_2")

        columns_from_gi = df.columns[id_1_col_idx + 1 : id_2_col_idx].values  # type: ignore
        columns_from_ai = df.columns[id_2_col_idx + 1 :].values  # type: ignore

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

        return df, metadata

    @staticmethod
    def _get_feature_name() -> str:
        """Returns name of the feature from the function that called this one."""
        function_name = inspect.stack()[1].function
        feature_name = function_name[len(CF_PREFIX) :]

        return feature_name

    @staticmethod
    @preprocess_init
    def cf_name(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        features_info = metadata.features_info

        # Add 'name' to 'other' features
        features_info["other"].append(feature_name)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_short_url(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        features_info = metadata.features_info

        # Add 'short_url' to 'other' features
        features_info["other"].append(feature_name)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_price(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        # Remove '.' from values and transform to numerical
        df[feature_name] = pd.to_numeric(
            df[feature_name].str.slice(stop=-1).str.replace(".", ""),
            errors="raise",
            downcast="unsigned",
        )

        # Remove cars that had price = 1
        price_1_cars = df[df[feature_name] == 1]
        df = df.drop(price_1_cars.index, axis=0)

        cars_price_less_than_100 = df.loc[df[feature_name] < 100, feature_name]
        # Remove cars that had price < 100
        df = df.drop(cars_price_less_than_100.index, axis=0)

        # Remove cars that had price > 80_000
        cars_price_bigger_than_80_000 = df[df[feature_name] > 80_000]
        df = df.drop(cars_price_bigger_than_80_000.index, axis=0)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_listing_followers_no(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        features_info = metadata.features_info

        # Transform to numerical
        df[feature_name] = pd.to_numeric(
            df[feature_name], downcast="unsigned", errors="raise"
        )

        # Added 'listing_followers_no' to numerical features
        features_info["numerical"].append(feature_name)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_location(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        features_info = metadata.features_info

        # Convert 'location' to categorical type (nominal)
        df[feature_name] = pd.Categorical(
            df[feature_name].astype("object"), ordered=False
        )

        # Add 'location' to 'nominal' features
        features_info["nominal"].append(feature_name)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_images_no(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        features_info = metadata.features_info

        # Transformed to numerical
        df[feature_name] = pd.to_numeric(
            df[feature_name], downcast="unsigned", errors="raise"
        )

        # Add 'images_no' to 'numerical' features
        features_info["numerical"].append(feature_name)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_safety(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()
        prefix = "s_"

        features_info = metadata.features_info

        # Create data frame with dummy columns
        df_safety_dummies = df[feature_name].str.get_dummies(sep=",").add_prefix(prefix)
        # Extend the data frame with dummy columns
        df = pd.concat([df, df_safety_dummies], axis=1)

        # Delete 'safety' column
        del df[feature_name]

        safety_columns = df_safety_dummies.columns.tolist()

        # Convert all remaining safety columns to boolean
        df[safety_columns] = df[safety_columns].astype("boolean")

        # Fix column names
        safety_columns_fixed = (
            df[safety_columns]
            .columns.str.strip()
            .str.replace(r"[- ]", "_", regex=True)
            .str.replace("/", "_ili_")
        ).tolist()

        df.rename(columns=dict(zip(safety_columns, safety_columns_fixed)), inplace=True)

        # Add all remaining safety columns to 'binary' features
        features_info["binary"].extend(safety_columns_fixed)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_equipment(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()
        prefix = "e_"

        features_info = metadata.features_info

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

        # Fix column names
        equipment_columns_fixed = (
            df[equipment_columns]
            .columns.str.strip()
            .str.replace(r"[- ]", "_", regex=True)
            .str.replace("/", "_ili_")
        ).tolist()

        df.rename(
            columns=dict(zip(equipment_columns, equipment_columns_fixed)), inplace=True
        )

        # Add all remaining equipment columns to 'binary' features
        features_info["binary"].extend(equipment_columns_fixed)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_other(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()
        prefix = "o_"

        features_info = metadata.features_info

        # Create data frame with dummy columns
        df_other_dummies = df[feature_name].str.get_dummies(sep=",").add_prefix(prefix)
        # Extend the data frame with dummy columns
        df = pd.concat([df, df_other_dummies], axis=1)

        # Delete 'other' column
        del df[feature_name]

        other_columns = df_other_dummies.columns.tolist()

        # Convert all remaining other columns to boolean
        df[other_columns] = df[other_columns].astype("boolean")

        # Fix column names
        other_columns_fixed = (
            df[other_columns]
            .columns.str.strip()
            .str.replace(r"[- ]", "_", regex=True)
            .str.replace("/", "_ili_")
        ).tolist()

        df.rename(columns=dict(zip(other_columns, other_columns_fixed)), inplace=True)

        # Delete taxi cars and the column
        taxi_cars = df[df.o_Taxi == True]
        df = df.drop(taxi_cars.index, axis=0)
        del df["o_Taxi"]
        other_columns_fixed.remove("o_Taxi")

        # Delete driving school cars and the column
        car_school_vehicles = df[df.o_Vozilo_auto_škole == True]
        df = df.drop(car_school_vehicles.index, axis=0)
        del df["o_Vozilo_auto_škole"]
        other_columns_fixed.remove("o_Vozilo_auto_škole")

        # Add all remaining other columns to 'binary' features
        features_info["binary"].extend(other_columns_fixed)

        return df, metadata

    @staticmethod
    @preprocess_init
    def cf_description(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        feature_name = InitialCleaner._get_feature_name()

        features_info = metadata.features_info

        # Add 'description' to 'other' features
        features_info["other"].append(feature_name)

        return df, metadata

    @staticmethod
    @preprocess_init
    def c_general_informations(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        features_info = metadata.features_info

        pd.set_option("mode.chained_assignment", None)

        # Delete new cars
        new_cars = df[df.gi_condition == "Novo vozilo"]
        df = df.drop(new_cars.index, axis=0)

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

        # Extract only value of KS (stands for horse powers) and remove spaces
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
            df[col] = pd.Categorical(df[col].astype("object"), ordered=False)

        # Convert numerical columns to numerical types
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors="raise", downcast="unsigned")

        features_info["nominal"].extend(nominal_cols)
        features_info["numerical"].extend(numerical_cols)
        features_info["other"].extend(other_cols)

        pd.set_option("mode.chained_assignment", "warn")

        return df, metadata

    @staticmethod
    @preprocess_init
    def c_additional_informations(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        features_info = metadata.features_info

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
        right_side_steering_wheel = df[
            df.ai_steering_wheel_side.str.strip() == "Desni volan"
        ]
        df = df.drop(right_side_steering_wheel.index, axis=0)

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
            df[col] = pd.Categorical(df[col].astype("object"), ordered=True)

        # Convert nominal columns to categorical types (nominal)
        for col in nominal_cols:
            df[col] = pd.Categorical(df[col].astype("object"), ordered=False)

        # Convert numerical columns to numerical types
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors="raise", downcast="unsigned")

        features_info["binary"].extend(binary_cols)
        features_info["ordinal"].extend(ordinal_cols)
        features_info["nominal"].extend(nominal_cols)
        features_info["numerical"].extend(numerical_cols)
        features_info["other"].extend(other_cols)

        pd.set_option("mode.chained_assignment", "warn")

        return df, metadata

    @staticmethod
    @preprocess_init
    def new_cars(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        """Insight gained in UnivariateAnalysis."""

        idx_to_remove = metadata.idx_to_remove
        df_cars_with_0_kilometerage = df[df.gi_kilometerage == 0]

        # Remove cars with 'gi_kilometerage' = 0
        df.drop(df_cars_with_0_kilometerage.index, inplace=True)
        idx_to_remove.extend(df_cars_with_0_kilometerage.index.to_list())

        return df, metadata

    @staticmethod
    @preprocess_init
    def irregular_label_rows(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        """Insight gained in MultivariateAnalysis."""

        idx_to_remove = metadata.idx_to_remove

        df_cars_equal_price_install_amt = df.loc[
            df.price == df.ai_installment_amount, :
        ]

        # Remove cars where 'price' = 'ai_installment_amount'
        df.drop(df_cars_equal_price_install_amt.index, axis=0, inplace=True)

        idx_to_remove.extend(df_cars_equal_price_install_amt.index.tolist())

        return df, metadata

    @staticmethod
    @preprocess_init
    def oldtimers(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        """Insight gained in MultivariateAnalysis."""

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
    def high_seats_cars(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        """Insight gained in MultivariateAnalysis."""

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
    def low_kilometerage_cars(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        """Insight gained in MultivariateAnalysis."""

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
    def c_remove_certain_rows(
        df: Dataset,
        metadata: Metadata,
        oldtimers_flag: bool = True,
        high_seats_cars_flag: bool = True,
        low_kilometerage_cars_flag: bool = True,
    ) -> tuple[Dataset, Metadata]:
        df, metadata = InitialCleaner.new_cars(df, metadata)
        df, metadata = InitialCleaner.irregular_label_rows(df, metadata)

        if oldtimers_flag:
            df, metadata = InitialCleaner.oldtimers(df, metadata)
        if high_seats_cars_flag:
            df, metadata = InitialCleaner.high_seats_cars(df=df, metadata=metadata)
        if low_kilometerage_cars_flag:
            df, metadata = InitialCleaner.low_kilometerage_cars(
                df=df, metadata=metadata
            )
        return df, metadata

    @staticmethod
    @preprocess_init
    def drop_features_correlated_with_label(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        features_info = metadata.features_info

        features_correlated_with_label = [
            "ai_deposit",
            "ai_installment_no",
            "ai_installment_amount",
            "ai_cash_payment",
        ]
        df.drop(columns=features_correlated_with_label, inplace=True)
        # Update features info
        features_info["numerical"] = [
            f for f in features_info["numerical"] if f not in features_correlated_with_label
        ]
        return df, metadata

    @staticmethod
    @preprocess_init
    def clean_individual_columns(
        df: Dataset, metadata: Metadata
    ) -> tuple[Dataset, Metadata]:
        df, metadata = InitialCleaner.cf_name(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_short_url(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_price(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_listing_followers_no(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_location(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_images_no(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_safety(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_equipment(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_other(df=df, metadata=metadata)
        df, metadata = InitialCleaner.cf_description(df=df, metadata=metadata)
        df, metadata = InitialCleaner.c_general_informations(df=df, metadata=metadata)
        df, metadata = InitialCleaner.c_additional_informations(
            df=df, metadata=metadata
        )
        return df, metadata

    @preprocess_init
    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        df, metadata = InitialCleaner.initial_preparation(df=df, metadata=metadata)
        df, metadata = InitialCleaner.clean_individual_columns(df=df, metadata=metadata)
        df, metadata = InitialCleaner.c_remove_certain_rows(
            df=df,
            metadata=metadata,
            oldtimers_flag=self.oldtimers_flag,
            high_seats_cars_flag=self.high_seats_cars_flag,
            low_kilometerage_cars_flag=self.low_kilometerage_cars_flag,
        )
        df, metadata = InitialCleaner.drop_features_correlated_with_label(
            df=df, metadata=metadata
        )
        return df, metadata
