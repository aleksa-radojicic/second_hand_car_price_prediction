from typing import Tuple

import numpy as np
import pandas as pd

from src.features.utils import CustomTransformer
from src.logger import log_message
from src.utils import (Dataset, Metadata, PipelineMetadata,
                       log_feature_info_dict, preprocess_init)

CF_PREFIX = "cf_"


class UACleaner(CustomTransformer):
    verbose: int

    def __init__(self, pipe_meta: PipelineMetadata, verbose: int = 0):
        super().__init__(pipe_meta)
        self.verbose = verbose

    @staticmethod
    @preprocess_init
    def ua_nominal_features(
        df: Dataset, metadata: Metadata
    ) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy

        # Drop empty categories
        for col in features_info["nominal"]:
            df[col] = df[col].cat.remove_unused_categories()

        # Replace ' ' category for 'gi_body_type' with NaN
        df.loc[df.gi_body_type == "", "gi_body_type"] = np.nan

        # Group simillar categories for 'gi_fuel_type'
        df.gi_fuel_type = pd.Categorical(
            df.gi_fuel_type.astype("object").replace(
                {
                    "Hibridni pogon (Benzin)": "Hibridni pogon",
                    "Hibridni pogon (Dizel)": "Hibridni pogon",
                    "Plug-in hibrid": "Hibridni pogon",
                    "Metan CNG": "Benzin + Metan (CNG)",
                }
            ),
            ordered=False,
        )

        # Group simillar categories for 'ai_gearbox_type'
        df.ai_gearbox_type = pd.Categorical(
            df.ai_gearbox_type.astype("object").replace(
                {
                    "Automatski": "Automatski / poluautomatski",
                    "Poluautomatski": "Automatski / poluautomatski",
                }
            )
        )

        constant_strat_cols = [
            "ai_floating_flywheel",
            "ai_interior_material",
            "ai_interior_color",
            "ai_ownership",
            "ai_import_country",
            "ai_sales_method",
        ]
        modus_strat_cols = [
            col for col in features_info["nominal"] if col not in constant_strat_cols
        ]
        cols_nan_strategy["const_unknown"].extend(constant_strat_cols)
        cols_nan_strategy["modus"].extend(modus_strat_cols)

        return df, metadata

    @staticmethod
    @preprocess_init
    def ua_ordinal_features(
        df: Dataset, metadata: Metadata
    ) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy

        # Drop empty categories
        for col in features_info["ordinal"]:
            df[col] = df[col].cat.remove_unused_categories()

        # Correct order of 'ai_damage' categories
        df.ai_damage = df.ai_damage.cat.reorder_categories(
            [
                "Nije oštećen",
                "Oštećen - u voznom stanju",
                "Oštećen - nije u voznom stanju",
            ]
        )

        modus_strat_cols = ["ai_engine_emission_class", "ai_damage"]
        cols_nan_strategy["modus"].extend(modus_strat_cols)

        return df, metadata

    @staticmethod
    @preprocess_init
    def ua_numerical_features(
        df: Dataset, metadata: Metadata
    ) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy
        idx_to_remove = metadata.idx_to_remove

        const_strat_cols_zero = ["listing_followers_no"]

        cols_scheduled_for_deletion = [
            "gi_battery_capacity",
            "ai_deposit",
            "ai_installment_no",
            "ai_cash_payment",
            "ai_range_on_full_battery_km",
        ]
        median_strat_cols = [
            col
            for col in features_info["numerical"]
            if col not in const_strat_cols_zero + cols_scheduled_for_deletion
        ]

        df_cars_with_0_kilometerage = df[df.gi_kilometerage == 0]

        # Remove cars with 'gi_kilometerage' = 0
        df.drop(df_cars_with_0_kilometerage.index, inplace=True)
        idx_to_remove.extend(df_cars_with_0_kilometerage.index.to_list())

        # Replace extreme value of 'gi_engine_capacity' with NaN (will be replaced with median)
        df.loc[df.gi_engine_capacity > 0.2 * 1e8, "gi_engine_capacity"] = np.nan

        features_info["features_to_delete"].extend(cols_scheduled_for_deletion)

        cols_nan_strategy["const_0"].extend(const_strat_cols_zero)
        cols_nan_strategy["median"].extend(median_strat_cols)

        return df, metadata

    @staticmethod
    @preprocess_init
    def ua_binary_features(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy

        cols_scheduled_for_deletion = [
            "e_Fabrički_ugrađeno_dečije_sedište",
            "e_Volan_u_kombinaciji_drvo_ili_koža",
            "o_Oldtimer",
            "o_Prilagođeno_invalidima",
            "o_Restauriran",
            "o_Test_vozilo",
            "o_Tuning",
        ]

        const_false_strat_cols = ["ai_credit", "ai_interest_free_credit", "ai_leasing"]
        modus_strat_cols = [
            col
            for col in features_info["binary"]
            if col not in cols_scheduled_for_deletion + const_false_strat_cols
        ]

        cols_nan_strategy["const_false"].extend(const_false_strat_cols)
        cols_nan_strategy["modus"].extend(modus_strat_cols)

        features_info["features_to_delete"].extend(cols_scheduled_for_deletion)

        return df, metadata

    @staticmethod
    @preprocess_init
    def ua_other_features(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy

        # Subset of other columns
        other_columns = ["gi_certified", "ai_registered_until"]
        today_date = np.datetime64("2024-01")

        # Transform 'gi_certified' and 'ai_registered_until' to difference of '2024-01' date and corresponding dates
        df.gi_certified = pd.to_numeric(
            (df.gi_certified - today_date).dt.days.astype("Int64"),  # type: ignore
            downcast="signed",
        )
        df.ai_registered_until = pd.to_numeric(
            (df.ai_registered_until - today_date).dt.days.astype("Int64"),  # type: ignore
            downcast="signed",
        )

        cols_nan_strategy["const_0"].extend(other_columns)

        for col in other_columns:
            features_info["other"].remove(col)
        features_info["numerical"].extend(other_columns)

        return df, metadata

    @staticmethod
    @preprocess_init
    def clean(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        df, metadata = UACleaner.ua_nominal_features(df=df, metadata=metadata)
        df, metadata = UACleaner.ua_ordinal_features(df=df, metadata=metadata)
        df, metadata = UACleaner.ua_numerical_features(df=df, metadata=metadata)
        df, metadata = UACleaner.ua_binary_features(df=df, metadata=metadata)
        df, metadata = UACleaner.ua_other_features(df=df, metadata=metadata)

        return df, metadata

    def transform(self, df: Dataset, y=None) -> Dataset:
        log_message("Performing cleaning from Univariate Analysis...", self.verbose)

        df, self.output_metadata = UACleaner.clean(df, self.input_metadata)

        log_feature_info_dict(
            self.output_metadata.features_info,
            "performing cleaning from Univariate Analysis",
            self.verbose,
        )

        log_message("Performed cleaning from Univariate Analysis.", self.verbose)
        return df
