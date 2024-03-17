from dataclasses import dataclass, field

from sklearn.pipeline import Pipeline

import src.features.initial_cleaning as ic
import src.features.multivariate_analysis as ma
import src.features.univariate_analysis as ua
from src.features.other_transformers import (CategoryTypesTransformer,
                                             ColumnsDropper,
                                             FinalColumnTransformer,
                                             MissingValuesHandler)
from src.utils import (Dataset, Metadata, PipelineMetadata,
                       create_pipeline_metadata_list, preprocess_init)


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
    ) -> tuple[Dataset, Metadata]:
        ic_obj = ic.InitialCleaner(
            pipe_meta=PipelineMetadata("", metadata, Metadata()),
            verbose=verbose,
        )
        df = ic_obj.fit_transform(df)  # type: ignore
        metadata = ic_obj.output_metadata
        return df, metadata

    def make_pipeline(self, metadata: Metadata, verbose: int = 0) -> Pipeline:
        pipeline_steps: list[str] = [
            ua.UACleaner.__name__,
            ma.MACleaner.__name__,
            ColumnsDropper.__name__,
            CategoryTypesTransformer.__name__,
            MissingValuesHandler.__name__,
            FinalColumnTransformer.__name__,
        ]
        pipe_metas: list[PipelineMetadata] = create_pipeline_metadata_list(
            steps=pipeline_steps, init_metadata=metadata
        )

        # Define transformers
        ua_transformer = ua.UACleaner(pipe_metas[0], verbose)
        ma_transformer = ma.MACleaner(
            pipe_meta=pipe_metas[1], cfg=self.cfg.ma_cleaner, verbose=verbose
        )

        columns_dropper = ColumnsDropper(pipe_metas[2], verbose)
        cat_handler = CategoryTypesTransformer(pipe_metas[3], verbose)
        nan_handler = MissingValuesHandler(pipe_metas[4], verbose)
        final_ct = FinalColumnTransformer(pipe_metas[5], verbose)

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
