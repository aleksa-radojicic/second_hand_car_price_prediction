from typing import Tuple

import numpy as np
from sklearn.pipeline import FunctionTransformer, Pipeline

import src.features.initial_cleaning as ic
import src.features.multivariate_analysis as ma
import src.features.univariate_analysis as ua
from src.data.make_dataset import DatasetMaker
from src.features.other_transformers import (CategoryTypesTransformer,
                                             ColumnsDropper)
from src.utils import (Dataset, Metadata, PipelineMetadata, preprocess_init,
                       train_test_split_custom)


class FeaturesBuilder:
    verbose: int

    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    @preprocess_init
    def initial_build(
        self, df: Dataset, metadata: Metadata
    ) -> Tuple[Dataset, Metadata]:
        ic_obj = ic.InitialCleaner(PipelineMetadata(), self.verbose)
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

        # Create pipeline
        data_transformation_pipeline = Pipeline(
            [
                ("ua", ua_transformer),
                ("ma", ma_transformer),
                ("col_dropper", columns_dropper),
                ("cat_handler", cat_handler),
            ]
        )
        data_transformation_pipeline.set_output(transform="pandas")
        return data_transformation_pipeline


def main():
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    df_interim, metadata_interim = FeaturesBuilder(verbose=2).initial_build(
        df=df_raw, metadata=metadata_raw
    )

    df_train, df_test = train_test_split_custom(df_interim)

    pipe = FeaturesBuilder.make_pipeline(metadata_interim, verbose=2)

    df_train_processed = pipe.fit_transform(df_train)

    print(df_train.shape)
    print(df_train_processed.shape)


if __name__ == "__main__":
    main()
