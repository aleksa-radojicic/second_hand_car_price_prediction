from typing import Tuple

import numpy as np
from sklearn.pipeline import FunctionTransformer, Pipeline

import src.features.initial_cleaning as ic
import src.features.multivariate_analysis as ma
import src.features.univariate_analysis as ua
from src.data.make_dataset import DatasetMaker
from src.logger import logging
from src.utils import (Dataset, Metadata, PipelineMetadata, preprocess_init,
                       train_test_split_custom)


class FeaturesBuilder:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose

    @staticmethod
    @preprocess_init
    def initial_build(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        ic_obj = ic.InitialCleaner(PipelineMetadata())
        df = ic_obj.clean(df)
        metadata = ic_obj.metadata

        return df, metadata

    @staticmethod
    def make_pipeline(metadata: Metadata) -> Pipeline:
        pipe_meta = PipelineMetadata(metadata)

        ua_transformer = FunctionTransformer(ua.UACleaner(pipe_meta).clean)
        ma_transformer = FunctionTransformer(ma.MACleaner(pipe_meta).clean)

        data_transformation_pipeline = Pipeline(
            [
                ("univariate_analysis", ua_transformer),
                ("multivariate_analysis", ma_transformer),
            ]
        )
        data_transformation_pipeline.set_output(transform="pandas")
        return data_transformation_pipeline


def main():
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    df_interim, metadata_interim = FeaturesBuilder.initial_build(
        df=df_raw, metadata=metadata_raw
    )

    logging.info("Metadata interim")
    logging.info(f"{metadata_interim=}")

    df_train, df_test = train_test_split_custom(df_interim)

    pipe = FeaturesBuilder.make_pipeline(metadata_interim)

    df_train_processed = pipe.fit_transform(df_train)

    print(df_train.shape)
    print(df_train_processed.shape)


if __name__ == "__main__":
    main()
