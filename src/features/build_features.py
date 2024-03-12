from dataclasses import dataclass, field
from typing import Tuple

from sklearn.pipeline import FunctionTransformer, Pipeline

import src.features.initial_cleaning as ic
import src.features.multivariate_analysis as ma
import src.features.univariate_analysis as ua
from src.features.other_transformers import (CategoryTypesTransformer,
                                             ColumnsDropper,
                                             FinalColumnTransformer,
                                             MissingValuesHandler)
from src.utils import Dataset, Metadata, PipelineMetadata, preprocess_init


@dataclass
class FeaturesBuilderConfig:
    ma_cleaner: ma.MAConfig = field(default_factory=ma.MAConfig)
    verbose: int = 0

class FeaturesBuilder:
    cfg: FeaturesBuilderConfig

    def __init__(self, cfg: FeaturesBuilderConfig = FeaturesBuilderConfig()):
        self.cfg = cfg

    @preprocess_init
    def initial_build(
        self, df: Dataset, metadata: Metadata, verbose: int = 0
    ) -> Tuple[Dataset, Metadata]:
        ic_obj = ic.InitialCleaner(PipelineMetadata(), verbose=verbose)
        df = ic_obj.start(df)
        metadata = ic_obj.metadata

        return df, metadata

    def make_pipeline(self, metadata: Metadata) -> Pipeline:
        pipe_meta = PipelineMetadata(metadata)

        ft = FunctionTransformer

        # Define transformers
        ua_transformer = ft(ua.UACleaner(pipe_meta, self.cfg.verbose).start)
        ma_transformer = ft(
            ma.MACleaner(
                pipe_meta=pipe_meta, cfg=self.cfg.ma_cleaner, verbose=self.cfg.verbose
            ).start
        )
        columns_dropper = ft(ColumnsDropper(pipe_meta, self.cfg.verbose).start)
        cat_handler = CategoryTypesTransformer(pipe_meta, self.cfg.verbose)
        nan_handler = MissingValuesHandler(pipe_meta, self.cfg.verbose)
        final_ct = FinalColumnTransformer(pipe_meta, self.cfg.verbose)

        # Create pipeline
        data_transformation_pipeline = Pipeline(
            [
                ("ua", ua_transformer),
                ("ma", ma_transformer),
                ("col_dropper", columns_dropper),
                ("cat_handler", cat_handler),
                ("nan_handler", nan_handler),
                ("final_ct", final_ct),
            ]
        )
        data_transformation_pipeline.set_output(transform="pandas")
        return data_transformation_pipeline
