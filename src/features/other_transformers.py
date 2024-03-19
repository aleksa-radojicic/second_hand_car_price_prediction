from dataclasses import dataclass
from typing import Self

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   RobustScaler)

from src import config
from src.features.utils import CustomTransformer
from src.utils import (COLUMNS_NAN_STRATEGY_MAP, Dataset, Metadata,
                       PipelineMetadata, preprocess_init)


def prefix_ds_metadata_columns(
    ds: Dataset, columns: list[str], prefix: str
) -> tuple[Dataset, list[str]]:
    ds.rename(
        columns={c: f"{prefix}{c}" for c in columns},
        inplace=True,
    )
    columns = [f"{prefix}{c}" for c in columns]
    return ds, columns


class ColumnsDropper(CustomTransformer):
    """Drops columns scheduled for deletion from the data frame and updates
    other columns list."""

    def __init__(self, pipe_meta: PipelineMetadata, verbose: int = 0):
        super().__init__(pipe_meta, verbose)

    @staticmethod
    @preprocess_init
    def drop(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        features_info = metadata.features_info
        cols_nan_strategy = metadata.cols_nan_strategy

        # Note: features_info['features_to_delete'] is copied because
        # values for key 'features_to_delete' are altered in the loop and
        # otherwise it would mess the loop

        columns_to_delete = features_info["features_to_delete"]

        # Delete columns from features_info
        for k, values in features_info.items():
            if k == "features_to_delete":
                continue
            features_info[k] = [
                value for value in values if value not in columns_to_delete
            ]

        # Delete columns from cols_nan_strategy
        for k, values in cols_nan_strategy.items():
            cols_nan_strategy[k] = [
                value for value in values if value not in columns_to_delete
            ]

        # Delete columns from dataset
        df.drop(columns=columns_to_delete, axis=1, inplace=True)

        return df, metadata

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.drop(df, metadata)


class ColumnsMetadataPrefixer(CustomTransformer):
    """Adds prefix to column names denoting its data type."""

    NUM_COLS_PREFIX: str = "numerical__"
    BIN_COLS_PREFIX: str = "binary__"
    ORD_COLS_PREFIX: str = "ordinal__"
    NOM_COLS_PREFIX: str = "nominal__"

    def __init__(self, pipe_meta: PipelineMetadata, verbose: int = 0) -> None:
        super().__init__(pipe_meta, verbose)

    @staticmethod
    @preprocess_init
    def prefix(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
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

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.prefix(df, metadata)


class CategoryTypesTransformer(CustomTransformer):
    n_jobs: int
    column_transformer: ColumnTransformer

    def __init__(
        self, pipe_meta: PipelineMetadata, verbose: int = 0, n_jobs: int = 1
    ) -> None:
        super().__init__(pipe_meta, verbose)
        self.n_jobs = n_jobs

    @preprocess_init
    def _create_column_transformer(self) -> ColumnTransformer:
        features_info = self.input_metadata.features_info

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

    def fit(self, df: Dataset, y=None, **params) -> Self:
        self.column_transformer = self._create_column_transformer()
        self.column_transformer.fit(df, y=None, **params)
        return self

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.column_transformer.transform(df), self.input_metadata  # type: ignore


class MissingValuesHandler(CustomTransformer):
    n_jobs: int
    column_transformer: ColumnTransformer

    def __init__(
        self, pipe_meta: PipelineMetadata, verbose: int = 0, n_jobs: int = 1
    ) -> None:
        super().__init__(pipe_meta, verbose)
        self.n_jobs = n_jobs

    @preprocess_init
    def _create_column_transformer(self) -> ColumnTransformer:
        cols_nan_strategy = self.input_metadata.cols_nan_strategy
        transformers: list[tuple[str, SimpleImputer, list[str]]] = []

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

    def fit(self, X, y=None, **params) -> Self:
        self.column_transformer = self._create_column_transformer()
        self.column_transformer.fit(X, y=None, **params)
        return self

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.column_transformer.transform(df), self.input_metadata  # type: ignore


@dataclass
class FinalColumnTransformerConfig:
    nominal_encoder: str = "ordinal"
    numerical_encoder: str = "passthrough"


class FinalColumnTransformer(CustomTransformer):
    cfg: FinalColumnTransformerConfig
    n_jobs: int
    column_transformer: ColumnTransformer

    def __init__(
        self,
        pipe_meta: PipelineMetadata,
        verbose: int = 0,
        cfg: FinalColumnTransformerConfig = FinalColumnTransformerConfig(),
        n_jobs: int = 1,
    ) -> None:
        super().__init__(pipe_meta, verbose)
        self.cfg = cfg
        self.n_jobs = n_jobs

    @preprocess_init
    def _create_column_transformer(self) -> ColumnTransformer:
        features_info = self.input_metadata.features_info

        numerical_encoder = {
            "passthrough": "passthrough",
            "standardscaler": StandardScaler(),
            "robustscaler": RobustScaler(),
            "minmaxscaler": MinMaxScaler(),
        }[self.cfg.numerical_encoder]

        nominal_encoder = {
            "onehot": OneHotEncoder(sparse_output=False),
            "ordinal": OrdinalEncoder(),
        }[self.cfg.nominal_encoder]

        column_transformer = ColumnTransformer(
            [
                ("numerical", numerical_encoder, features_info["numerical"]),
                ("binary", "passthrough", features_info["binary"]),
                ("ordinal", "passthrough", features_info["ordinal"]),
                ("nominal", nominal_encoder, features_info["nominal"]),
                # NOTE: This should be in the InitialCleaning
                # ("label", "passthrough", [config.LABEL]),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
            verbose=bool(self.verbose),
            n_jobs=self.n_jobs,
        )
        column_transformer.set_output(transform="pandas")
        return column_transformer

    def fit(self, df: Dataset, y=None, **params) -> Self:
        self.column_transformer = self._create_column_transformer()
        self.column_transformer.fit(df, y=None, **params)
        return self

    def start(self, df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:
        return self.column_transformer.transform(df), self.input_metadata  # type: ignore
