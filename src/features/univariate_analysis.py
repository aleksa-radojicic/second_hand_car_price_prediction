from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import FeaturesInfo
from src.utils import init_cols_nan_strategy, preprocess_init


class UACleaner:
    CF_PREFIX = "cf_"

    cols_nan_strategy = init_cols_nan_strategy()
    idx_to_remove: List[int] = []

    def __init__(self, features_info: FeaturesInfo):
        self.features_info = features_info

    @preprocess_init
    def ua_nominal_features(
        self,
        df: pd.DataFrame,
        features_info: FeaturesInfo,
        cols_nan_strategy: Dict[str, List[str]],
        idx_to_remove: List[int],
    ) -> Tuple[pd.DataFrame, FeaturesInfo, Dict[str, List[str]], List[int]]:
        # Drop empty categories
        for col in features_info["nominal"]:
            df[col] = df[col].cat.remove_unused_categories()

        # Replace ' ' category for 'gi_body_type' with NaN
        df.loc[df.gi_body_type == "", "gi_body_type"] = np.nan

        # Group simillar categories for 'gi_fuel_type'
        df.gi_fuel_type = pd.Categorical(
            df.gi_fuel_type.astype("string").replace(
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
            df.ai_gearbox_type.astype("string").replace(
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

        return df, features_info, cols_nan_strategy, idx_to_remove

    @preprocess_init
    def ua_ordinal_features(
        self,
        df: pd.DataFrame,
        features_info: FeaturesInfo,
        cols_nan_strategy: Dict[str, List[str]],
        idx_to_remove: List[int],
    ) -> Tuple[pd.DataFrame, FeaturesInfo, Dict[str, List[str]], List[int]]:
        # Drop empty categories
        for col in features_info["ordinal"]:
            df[col] = df[col].cat.remove_unused_categories()

        modus_strat_cols = ["ai_engine_emission_class", "ai_damage"]
        cols_nan_strategy["modus"].extend(modus_strat_cols)

        return df, features_info, cols_nan_strategy, idx_to_remove

    @preprocess_init
    def ua_numerical_features(
        self,
        df: pd.DataFrame,
        features_info: FeaturesInfo,
        cols_nan_strategy: Dict[str, List[str]],
        idx_to_remove: List[int],
    ) -> Tuple[pd.DataFrame, FeaturesInfo, Dict[str, List[str]], List[int]]:
        const_strat_cols_zero = ["listing_followers_no"]
        cols_scheduled_for_deletion = [
            "gi_battery_capacity",
            "ai_deposit",
            "ai_installment_no",
            "ai_cash_payment",
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
        print("Dropped 'gi_battery_capacity' (too many zero values)")
        print("Dropped 'ai_deposit' (label leakage)")
        print("Dropped 'ai_installment_no' (label leakage)")
        print("Dropped 'ai_cash_payment' (label leakage)")
        print("Dropped 'ai_range_on_full_battery_km' (too many zero values)")

        cols_nan_strategy["const_0"].extend(const_strat_cols_zero)
        cols_nan_strategy["median"].extend(median_strat_cols)

        return df, features_info, cols_nan_strategy, idx_to_remove

    @preprocess_init
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        features_info = self.features_info
        cols_nan_strategy = self.cols_nan_strategy
        idx_to_remove = self.idx_to_remove

        df, features_info, cols_nan_strategy, idx_to_remove = self.ua_nominal_features(
            df=df,
            features_info=features_info,
            cols_nan_strategy=cols_nan_strategy,
            idx_to_remove=idx_to_remove,
        )
        df, features_info, cols_nan_strategy, idx_to_remove = self.ua_ordinal_features(
            df=df,
            features_info=features_info,
            cols_nan_strategy=cols_nan_strategy,
            idx_to_remove=idx_to_remove,
        )
        df, features_info, cols_nan_strategy, idx_to_remove = (
            self.ua_numerical_features(
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
