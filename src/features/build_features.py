from typing import Tuple

import numpy as np
from sklearn.pipeline import FunctionTransformer, Pipeline

import src.features.initial_cleaning as ic
import src.features.multivariate_analysis as ma
import src.features.univariate_analysis as ua
from src.data.make_dataset import DatasetMaker
from src.features.other_transformers import (CategoryTypesTransformer,
                                             ColumnsDropper,
                                             FinalColumnTransformer,
                                             MissingValuesHandler)
from src.utils import (Dataset, Metadata, PipelineMetadata, preprocess_init,
                       train_test_split_custom)


class FeaturesBuilder:
    @preprocess_init
    def initial_build(
        self, df: Dataset, metadata: Metadata, verbose: int = 0
    ) -> Tuple[Dataset, Metadata]:
        ic_obj = ic.InitialCleaner(PipelineMetadata(), verbose=verbose)
        df = ic_obj.start(df)
        metadata = ic_obj.metadata

        return df, metadata

    @staticmethod
    def make_pipeline(metadata: Metadata, verbose: int = 0) -> Pipeline:
        pipe_meta = PipelineMetadata(metadata)

        ft = FunctionTransformer

        # Define transformers
        ua_transformer = ft(ua.UACleaner(pipe_meta, verbose).start)
        ma_transformer = ft(ma.MACleaner(pipe_meta, verbose).start)
        columns_dropper = ft(ColumnsDropper(pipe_meta, verbose).start)
        cat_handler = CategoryTypesTransformer(pipe_meta, verbose)
        nan_handler = MissingValuesHandler(pipe_meta, verbose)
        final_ct = FinalColumnTransformer(pipe_meta, verbose)

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