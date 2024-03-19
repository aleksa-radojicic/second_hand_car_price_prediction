from dataclasses import dataclass, field

import src.features.initial_cleaning as ic
import src.features.multivariate_analysis as ma
import src.features.univariate_analysis as ua
from src.features.other_transformers import (CategoryTypesTransformer,
                                             ColumnsDropper,
                                             FinalColumnTransformer,
                                             FinalColumnTransformerConfig,
                                             MissingValuesHandler,
                                             PreprocessorPipeline)
from src.utils import (Dataset, Metadata, PipelineMetadata,
                       create_pipeline_metadata_list, preprocess_init)


@dataclass
class FeaturesBuilderConfig:
    init_cleaner: ic.InitialCleanerConfig = field(
        default_factory=ic.InitialCleanerConfig
    )
    ua_cleaner: ua.UACleanerConfig = field(default_factory=ua.UACleanerConfig)
    ma_cleaner: ma.MACleanerConfig = field(default_factory=ma.MACleanerConfig)
    final_ct: FinalColumnTransformerConfig = field(
        default_factory=FinalColumnTransformerConfig
    )
    verbose: int = 0


class FeaturesBuilder:
    cfg: FeaturesBuilderConfig

    def __init__(self, cfg: FeaturesBuilderConfig = FeaturesBuilderConfig()):
        self.cfg = cfg

    @preprocess_init
    def initial_build(
        self, df: Dataset, metadata: Metadata, verbose: int = 0
    ) -> tuple[Dataset, Metadata]:
        ic_obj = (
            ic.InitialCleaner(
                self.cfg.init_cleaner.oldtimers_flag,
                self.cfg.init_cleaner.high_seats_cars_flag,
                self.cfg.init_cleaner.low_kilometerage_cars_flag,
            )
            .set_pipe_meta(PipelineMetadata("", metadata, Metadata()))
            .set_verbose(verbose)
        )
        df = ic_obj.fit_transform(df)  # type: ignore
        metadata = ic_obj.output_metadata
        return df, metadata

    def make_pipeline(
        self, pipe_step_names: list[str], metadata: Metadata, verbose: int = 0
    ) -> PreprocessorPipeline:
        pipe_metas: list[PipelineMetadata] = create_pipeline_metadata_list(
            steps=pipe_step_names, init_metadata=metadata
        )

        # Define transformers
        ua_transformer = (
            ua.UACleaner().set_pipe_meta(pipe_metas[0]).set_verbose(verbose)
        )
        ma_transformer = (
            ma.MACleaner(self.cfg.ma_cleaner.finalize_flag)
            .set_pipe_meta(pipe_metas[1])
            .set_verbose(verbose)
        )

        columns_dropper = (
            ColumnsDropper().set_pipe_meta(pipe_metas[2]).set_verbose(verbose)
        )
        cat_handler = (
            CategoryTypesTransformer().set_pipe_meta(pipe_metas[3]).set_verbose(verbose)
        )
        nan_handler = (
            MissingValuesHandler().set_pipe_meta(pipe_metas[4]).set_verbose(verbose)
        )
        final_ct = (
            FinalColumnTransformer(
                nominal_encoder=self.cfg.final_ct.nominal_encoder,
                numerical_encoder=self.cfg.final_ct.numerical_encoder,
            )
            .set_pipe_meta(pipe_metas[5])
            .set_verbose(verbose)
        )

        # Create pipeline
        data_transformation_pipeline = PreprocessorPipeline(
            [
                *zip(
                    pipe_step_names,
                    (
                        ua_transformer,
                        ma_transformer,
                        columns_dropper,
                        cat_handler,
                        nan_handler,
                        final_ct,
                    ),
                )
            ]
        )
        data_transformation_pipeline.set_output(transform="pandas")
        return data_transformation_pipeline
