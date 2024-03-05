from typing import List, Optional, Tuple

import numpy as np

from src.logger import log_message
from src.utils import (Dataset, Metadata, PipelineMetadata,
                       log_feature_info_dict, preprocess_init)


def prefix_ds_metadata_columns(
    ds: Dataset, columns: List[str], prefix: str
) -> Tuple[Dataset, List[str]]:
    ds.rename(
        columns={c: f"{prefix}{c}" for c in columns},
        inplace=True,
    )
    columns = [f"{prefix}{c}" for c in columns]
    return ds, columns


class ColumnsDropper:
    """Drops columns scheduled for deletion from the data frame and updates
    other columns list."""

    pipe_meta: PipelineMetadata
    cached_metadata: Optional[Metadata]
    verbose: int

    def __init__(self, pipe_meta: PipelineMetadata, verbose: int = 0):
        self.pipe_meta = pipe_meta
        self.cached_metadata = None
        self.verbose = verbose

    @property
    def metadata(self) -> Metadata:
        return self.pipe_meta.data

    @metadata.setter
    def metadata(self, metadata):
        self.pipe_meta.data = metadata

    @staticmethod
    @preprocess_init
    def drop(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info

        # Note: features_info['features_to_delete'] is copied because
        # values for key 'features_to_delete' are altered in the loop and
        # otherwise it would mess the loop

        columns_to_delete = features_info["features_to_delete"]

        for k, values in features_info.items():
            if k == "features_to_delete":
                continue
            features_info[k] = [
                value for value in values if value not in columns_to_delete
            ]

        df.drop(columns=columns_to_delete, axis=1, inplace=True)

        return df, metadata

    @preprocess_init
    def start(self, df: Dataset, y=None) -> Dataset:
        log_message("Dropping columns scheduled for deletion...", self.verbose)

        if not self.cached_metadata:
            self.cached_metadata = self.metadata

        metadata = self.cached_metadata

        df, metadata = ColumnsDropper.drop(df, metadata)

        log_feature_info_dict(
            metadata.cols_nan_strategy,
            "dropping columns scheduled for deletion",
            self.verbose,
        )

        log_message(
            "Dropped columns scheduled for deletion successfully.", self.verbose
        )

        self.metadata = metadata

        return df


class ColumnsMetadataPrefixer:
    """Adds prefix to column names denoting its data type."""

    pipe_meta: PipelineMetadata
    cached_metadata: Optional[Metadata]
    verbose: int

    NUM_COLS_PREFIX: str = "numerical__"
    BIN_COLS_PREFIX: str = "binary__"
    ORD_COLS_PREFIX: str = "ordinal__"
    NOM_COLS_PREFIX: str = "nominal__"

    def __init__(self, pipe_meta: PipelineMetadata, verbose: int = 0) -> None:
        self.pipe_meta = pipe_meta
        self.cached_metadata = None
        self.verbose = verbose

    @property
    def metadata(self) -> Metadata:
        return self.pipe_meta.data

    @metadata.setter
    def metadata(self, metadata):
        self.pipe_meta.data = metadata

    @staticmethod
    @preprocess_init
    def prefix(df: Dataset, metadata: Metadata) -> Tuple[Dataset, Metadata]:
        features_info = metadata.features_info

        df, features_info["numerical"] = prefix_ds_metadata_columns(
            df, features_info["numerical"], ColumnsMetadataPrefixer.NUM_COLS_PREFIX
        )
        df, features_info["binary"] = prefix_ds_metadata_columns(
            df, features_info["binary"], ColumnsMetadataPrefixer.BIN_COLS_PREFIX
        )
        df, features_info["ordinal"] = prefix_ds_metadata_columns(
            df, features_info["ordinal"], ColumnsMetadataPrefixer.ORD_COLS_PREFIX
        )
        df, features_info["nominal"] = prefix_ds_metadata_columns(
            df, features_info["nominal"], ColumnsMetadataPrefixer.NOM_COLS_PREFIX
        )

        return df, metadata

    @preprocess_init
    def start(self, df: Dataset, y=None) -> Dataset:
        if not self.cached_metadata:
            self.cached_metadata = self.metadata

        metadata = self.cached_metadata

        df, metadata = ColumnsMetadataPrefixer.prefix(df, metadata)

        log_feature_info_dict(
            metadata.cols_nan_strategy,
            "adding data type prefix to columns",
            self.verbose,
        )

        log_message("Added data type prefix to columns successfully.", self.verbose)

        self.metadata = metadata
        return df
