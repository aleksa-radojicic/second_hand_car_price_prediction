from typing import List, Optional, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src import config
from src.config import FeaturesInfo
from src.logger import log_message
from src.utils import (COLUMNS_NAN_STRATEGY_MAP, ColsNanStrategy, Dataset,
                       Metadata, PipelineMetadata, log_feature_info_dict,
                       preprocess_init)


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


class CategoryTypesTransformer:
    pipe_meta: PipelineMetadata
    cached_metadata: Optional[Metadata]
    verbose: int
    n_jobs: int
    column_transformer: Optional[ColumnTransformer] = None

    def __init__(
        self, pipe_meta: PipelineMetadata, verbose: int = 0, n_jobs: int = 1
    ) -> None:
        self.pipe_meta = pipe_meta
        self.cached_metadata = None
        self.verbose = verbose
        self.n_jobs = n_jobs

    @property
    def metadata(self) -> Metadata:
        return self.pipe_meta.data

    @metadata.setter
    def metadata(self, metadata):
        self.pipe_meta.data = metadata

    @preprocess_init
    def _get_column_transformer(self, features_info: FeaturesInfo) -> ColumnTransformer:
        binary_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=config.UNKNOWN_VALUE_BINARY,
        )
        ordinal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=config.UNKNOWN_VALUE_ORDINAL,
        )
        nominal_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=config.UNKNOWN_VALUE_NOMINAL,
        )

        column_transformer = ColumnTransformer(
            [
                ("binary", binary_encoder, features_info["binary"]),
                ("ordinal", ordinal_encoder, features_info["ordinal"]),
                ("nominal", nominal_encoder, features_info["nominal"]),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
            verbose=bool(self.verbose),
            n_jobs=self.n_jobs,
        )
        column_transformer.set_output(transform="pandas")
        return column_transformer

    def init_column_transformer(self):
        if not self.cached_metadata:
            self.cached_metadata = self.metadata

        features_info = self.cached_metadata.features_info

        self.column_transformer = self._get_column_transformer(features_info)

    def fit(self, df: Dataset, y=None, **params):
        if not self.column_transformer:
            self.init_column_transformer()

        return self.column_transformer.fit(df, y=None, **params)  # type: ignore

    def transform(self, df, **params):
        log_message("Handling category types...", self.verbose)

        if not self.column_transformer:
            raise Exception(
                f"Column transformer is None and should be instance of {ColumnTransformer.__name__}"
            )

        if not self.cached_metadata:
            raise TypeError(
                f"Cached metadata is None and should be instance of {Metadata.__name__}"
            )

        features_info = self.cached_metadata.features_info

        log_feature_info_dict(
            features_info,
            "handling category types",
            self.verbose,
        )

        log_message("Handled category types successfully.", self.verbose)

        return self.column_transformer.transform(df, **params)

    def set_output(*args, **kwargs):
        pass


class MissingValuesHandler:
    pipe_meta: PipelineMetadata
    cached_metadata: Optional[Metadata]
    verbose: int
    n_jobs: int
    column_transformer: Optional[ColumnTransformer] = None

    def __init__(
        self, pipe_meta: PipelineMetadata, verbose: int = 0, n_jobs: int = 1
    ) -> None:
        self.pipe_meta = pipe_meta
        self.cached_metadata = None
        self.verbose = verbose
        self.n_jobs = n_jobs

    @property
    def metadata(self) -> Metadata:
        return self.pipe_meta.data

    @metadata.setter
    def metadata(self, metadata):
        self.pipe_meta.data = metadata

    @preprocess_init
    def _get_column_transformer(
        self, cols_nan_strategy: ColsNanStrategy
    ) -> ColumnTransformer:
        transformers: list[Tuple[str, SimpleImputer, List[str]]] = []

        for nan_strategy, imputer_obj in COLUMNS_NAN_STRATEGY_MAP.items():
            columns_for_applying = cols_nan_strategy[nan_strategy]

            transformers.append((nan_strategy, imputer_obj, columns_for_applying))

        column_transformer = ColumnTransformer(
            transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
            verbose=bool(self.verbose),
            n_jobs=self.n_jobs,
        )
        column_transformer.set_output(transform="pandas")
        return column_transformer

    @staticmethod
    def filter_col_nan_strategy_map(
        cols_nan_strategy: ColsNanStrategy,
    ) -> ColsNanStrategy:
        return cols_nan_strategy

    def init_column_transformer(self):
        if not self.cached_metadata:
            self.cached_metadata = self.metadata

        cols_nan_strategy = self.cached_metadata.cols_nan_strategy
        cols_nan_strategy_filtered = MissingValuesHandler.filter_col_nan_strategy_map(
            cols_nan_strategy
        )

        self.column_transformer = self._get_column_transformer(
            cols_nan_strategy_filtered
        )

    def fit(self, X, y=None, **params):
        if not self.column_transformer:
            self.init_column_transformer()
        return self.column_transformer.fit(X, y=None, **params)  # type: ignore

    def transform(self, X, **params) -> Dataset:
        log_message("Handling missing values...", self.verbose)

        if not self.cached_metadata:
            raise TypeError(
                f"Cached metadata is None and should be instance of {Metadata.__name__}"
            )
        features_info = self.cached_metadata.features_info

        if not self.column_transformer:
            raise Exception(
                f"Column transformer is None and should be instance of {ColumnTransformer.__name__}"
            )

        df: Dataset = self.column_transformer.transform(X, **params)  # type: ignore

        log_feature_info_dict(
            features_info,
            "handling missing values",
            self.verbose,
        )

        log_message("Handled missing values successfully.", self.verbose)
        return df

    def set_output(*args, **kwargs):
        pass


class FinalColumnTransformer:
    pipe_meta: PipelineMetadata
    cached_metadata: Optional[Metadata]
    verbose: int
    n_jobs: int
    column_transformer: Optional[ColumnTransformer] = None

    def __init__(
        self, pipe_meta: PipelineMetadata, verbose: int = 0, n_jobs: int = 1
    ) -> None:
        self.pipe_meta = pipe_meta
        self.cached_metadata = None
        self.verbose = verbose
        self.n_jobs = n_jobs

    @property
    def metadata(self) -> Metadata:
        return self.pipe_meta.data

    @metadata.setter
    def metadata(self, metadata):
        self.pipe_meta.data = metadata

    @preprocess_init
    def _get_column_transformer(self, features_info: FeaturesInfo) -> ColumnTransformer:
        column_transformer = ColumnTransformer(
            [
                ("numerical", "passthrough", features_info["numerical"]),
                ("binary", "passthrough", features_info["binary"]),
                ("ordinal", "passthrough", features_info["ordinal"]),
                ("nominal", OneHotEncoder(sparse_output=False), features_info["nominal"]),
                ("label", "passthrough", [config.LABEL]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
            verbose=bool(self.verbose),
            n_jobs=self.n_jobs,
        )
        column_transformer.set_output(transform="pandas")
        return column_transformer

    def init_column_transformer(self):
        if not self.cached_metadata:
            self.cached_metadata = self.metadata

        features_info = self.cached_metadata.features_info

        self.column_transformer = self._get_column_transformer(features_info)

    def fit(self, df: Dataset, y=None, **params):
        if not self.column_transformer:
            self.init_column_transformer()

        return self.column_transformer.fit(df, y=None, **params)  # type: ignore

    def transform(self, df, **params):
        log_message("Applying final column transformer...", self.verbose)

        if not self.column_transformer:
            raise Exception(
                f"Column transformer is None and should be instance of {ColumnTransformer.__name__}"
            )

        if not self.cached_metadata:
            raise TypeError(
                f"Cached metadata is None and should be instance of {Metadata.__name__}"
            )

        features_info = self.cached_metadata.features_info

        log_feature_info_dict(
            features_info,
            "applying final column transformer",
            self.verbose,
        )

        log_message("Applied final column transformer successfully.", self.verbose)

        return self.column_transformer.transform(df, **params)

    def set_output(*args, **kwargs):
        pass
