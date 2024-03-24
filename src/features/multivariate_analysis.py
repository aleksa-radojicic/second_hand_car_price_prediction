from dataclasses import dataclass

from src.features.utils import CustomTransformer
from src.utils import Dataset, Metadata, preprocess_init


@dataclass
class MACleanerConfig:
    finalize_flag: bool = True


class MACleaner(CustomTransformer):
    finalize_flag: bool

    def __init__(self, finalize_flag: bool = True):
        super().__init__()
        self.finalize_flag = finalize_flag

    @staticmethod
    @preprocess_init
    def ma_finalize(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
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

    @preprocess_init
    def clean(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        if self.finalize_flag:
            df, metadata = MACleaner.ma_finalize(df=df, metadata=metadata)

        return df, metadata

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.clean(df, metadata)
