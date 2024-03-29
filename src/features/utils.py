from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_regression

from src.logger import log_message
from src.utils import (Dataset, Metadata, PipelineMetadata,
                       log_feature_info_dict)


class CustomTransformer(TransformerMixin, BaseEstimator):
    verbose: int

    def set_pipe_meta(self, pipe_meta: PipelineMetadata) -> Self:
        self._pipe_meta = pipe_meta
        return self

    def set_verbose(self, verbose: int = 0) -> Self:
        self.verbose = verbose
        return self

    @property
    def input_metadata(self) -> Metadata:
        return self._pipe_meta.input_meta

    @property
    def step_name(self) -> str:
        return self._pipe_meta.step_name

    @property
    def output_metadata(self) -> Metadata:
        return self._pipe_meta.output_meta

    @output_metadata.setter
    def output_metadata(self, metadata: Metadata):
        self._pipe_meta.update_output_meta(metadata)

    def fit(self, df: Dataset, y=None) -> Self:
        return self

    def transform(self, df: Dataset) -> Dataset:
        log_message(f"{self.step_name} of the dataset started...", self.verbose)

        df, self.output_metadata = self.start(df=df, metadata=self.input_metadata)

        log_feature_info_dict(
            self.output_metadata.features_info,
            f"{self.step_name.lower()} of the dataset",
            self.verbose,
        )

        log_message(f"{self.step_name} of the dataset finished.", self.verbose)
        return df

    @staticmethod
    def start(df: Dataset, metadata: Metadata) -> tuple[Dataset, Metadata]:  # type: ignore
        pass

    def set_output(*args, **kwargs):
        pass

    # ! Needs to solve how to display pipeline without pipe_meta and verbose keys
    # ! because this approach doesn't when tuning hyperparameters.
    # def get_params(self, deep: bool = True) -> dict:
    #     """Override default get_params method to exclude information about
    #     pipe_meta and verbose. Used when displaying the transformer."""
    #     result: dict = super().get_params(deep)
    #     del result["pipe_meta"]
    #     del result["verbose"]
    #     return result


def get_anova_importance_scores(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    f_stat, p_val = f_regression(X, y)

    # Fit the feature selector and sort the results by score
    scores = pd.DataFrame(
        {"scores": f_stat, "p_val": p_val}, index=X.columns
    ).sort_values(by="scores", ascending=False)

    scores.index.name = "anova_importance_scores"

    return scores
