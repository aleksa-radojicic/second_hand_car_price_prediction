import collections
import inspect
import os
from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.display import Markdown
from IPython.core.display_functions import display
from scipy import stats

from src import config
from src.config import FeaturesInfo
from src.utils import (ColsNanStrategy, Dataset, Metadata, load_dataset,
                       load_metadata, save_dataset, save_metadata)

CF_PREFIX = "cf_"
NB_SUFFIX = "_nb"

STAGES_DICT = [
    {"name": "1_IC", "folder_path": f"{os.getcwd()}/artifacts"},
    {"name": "2_UA", "folder_path": f"{os.getcwd()}/artifacts"},
    {"name": "3_MA", "folder_path": f"{os.getcwd()}/artifacts"},
]


def get_feature_name() -> str:
    """Returns name of the feature from the function that called this one."""
    function_name = inspect.stack()[1].function
    feature_name = function_name[len(CF_PREFIX) : -len(NB_SUFFIX)]
    return feature_name


def get_nas(df: pd.DataFrame) -> pd.DataFrame:
    missing_values = df.isna().sum().sort_values(ascending=False)
    missing_values = missing_values[missing_values > 0]
    missing_percentage = (missing_values / len(df)) * 100

    # Create a DataFrame to display both counts and percentages
    result_df = pd.DataFrame(
        {"missing count": missing_values, "missing [%]": missing_percentage}
    )

    return result_df


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Expands pd.DataFrame.describe by showing missing rows by column and its percentage."""

    pandas_df_describe = df.describe()
    describe_expanded = pd.concat([pandas_df_describe, get_nas(df).T])
    return describe_expanded


def display_feature_name_heading(feature):
    display(Markdown(f"<h3>'{feature}' feature</h3>"))


def show_hist_box_numerical_col(df, numerical_col):
    fig, axs = plt.subplots(1, 2)

    df[numerical_col].plot.hist(ax=axs[0], xlabel=numerical_col)
    ax2 = df[numerical_col].plot.kde(ax=axs[0], secondary_y=True)
    ax2.set_ylim(0)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    df[numerical_col].plot.box(ax=axs[1])

    fig.tight_layout()

    print(f"Univariate analysis of '{numerical_col}' column")
    print("Histogram and box plot")
    plt.show()
    print("Descriptive statistics")
    display(df[numerical_col].describe())

    print(
        f"Variance: {stats.variation(df[numerical_col].astype('double'), ddof=1, nan_policy='omit')}"
    )
    print(
        f"Skewness: {stats.skew(df[numerical_col].astype('double'), nan_policy='omit')}"
    )
    print(
        f"Kurtosis: {stats.kurtosis(df[numerical_col].astype('double'), nan_policy='omit')}\n"
    )

    print("NA values")
    n_na_values = df[numerical_col].isna().sum()
    perc_na_values = n_na_values / df[numerical_col].shape[0] * 100
    print(f"Count [n]: {n_na_values}")
    print(f"Percentage [%]: {perc_na_values}%")


def get_value_counts_freq_with_perc(df, column):
    value_counts_freq = df.loc[:, column].value_counts(dropna=False)
    value_counts_perc = value_counts_freq / df.shape[0] * 100

    result = pd.concat([value_counts_freq, value_counts_perc], axis=1)
    result.columns.values[1] = "percentage [%]"
    return result


def save_artifacts(
    file_name: str,
    path: str,
    dataset: Dataset,
    metadata: Metadata,
):
    save_dataset(file_name, path, dataset)
    save_metadata(file_name, path, metadata)


def load_artifacts(file_name: str, path: str) -> Tuple[Dataset, Metadata]:
    dataset = load_dataset(file_name, path)
    metadata = load_metadata(file_name, path)

    return dataset, metadata


def plot_anova_importance(
    anova_scores: pd.DataFrame,
    title: str = "ANOVA",
    figsize: tuple[int, int] = (20, 10),
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    sns.barplot(x=anova_scores.scores, y=anova_scores.index.values, ax=ax)
    plt.close()
    ax.bar_label(ax.containers[0])

    fig.suptitle(title, fontsize=30)
    display(fig)
    return ax


def plot_correlation_heatmap(corr: pd.DataFrame):
    _, ax = plt.subplots(figsize=(13, 7))

    # Upper triangular heatmap of correlations with annotations
    sns.heatmap(corr, annot=True, mask=np.triu(corr), fmt=".2f", ax=ax)

    return ax


def plot_bar_correlations(corr: pd.Series, colors: list[str], by: str):
    fig, ax = plt.subplots()
    ax: plt.Axes
    sns.barplot(
        x=corr.abs(),
        y=corr.index,
        hue=colors,
        legend=True,
        ax=ax,
    )
    plt.close()
    ax.set_ylabel("")
    ax.set_title(f"Correlation of numerical features with '{by}'")
    # ax.bar_label(labels=[np.round(val.get_label(), decimals=2) for val in ax.containers[0]])
    # ax.bar_label(labels=ax.containers[1])
    for bars in ax.containers:
        ax.bar_label(bars)  # type: ignore
    display(fig)

    return ax


def test_features_info_duplicates(features_info: FeaturesInfo):
    all_feats = []
    for key in features_info:
        if key == "features_to_delete":
            continue
        all_feats.extend(features_info[key])

    # Get duplicate features
    dupe_feats_freq = {
        feat: count
        for feat, count in collections.Counter(all_feats).items()
        if count > 1
    }
    if dupe_feats_freq:
        raise AssertionError(
            f"Features info contains following duplicate features:\n{dupe_feats_freq}"
        )


def test_features_info_with_columns(df: Dataset, features_info: FeaturesInfo):
    all_feats = []
    for key in features_info:
        if key == "features_to_delete":
            continue
        all_feats.extend(features_info[key])

    all_cols = df.drop(config.LABEL, axis=1).columns.tolist()

    not_in_right_msg = f"Features not in right:\n{set(all_feats) - set(all_cols)}"
    not_in_left_msg = f"Features not in left:\n{set(all_cols) - set(all_feats)}"
    error_msg = f"{not_in_right_msg}\n{not_in_left_msg}"

    assert set(all_feats) == set(all_cols), error_msg


def test_cols_nan_strategy_duplicates(cols_nan_strategy: ColsNanStrategy):
    # Columns nan strategy all columns list
    cns_list = []
    for strategy in cols_nan_strategy:
        cns_list.extend(cols_nan_strategy[strategy])

    # Get duplicate features
    dupe_cols_freq = {
        col: freq for col, freq in collections.Counter(cns_list).items() if freq > 1
    }
    if dupe_cols_freq:
        raise AssertionError(
            f"Cols_nan_strategy contains following duplicate features:\n{dupe_cols_freq}"
        )


def test_cols_nan_strategy_with_features_info(
    cols_nan_strategy: ColsNanStrategy, features_info: FeaturesInfo
):
    all_feats = []
    for key in features_info:
        if key == "other":
            continue
        all_feats.extend(features_info[key])

    all_feats = [
        col for col in all_feats if col not in features_info["features_to_delete"]
    ]

    # Columns nan strategy all columns list
    cns_list = []
    for strategy in cols_nan_strategy:
        cns_list.extend(cols_nan_strategy[strategy])

    # NOTE: not_in_fi_msg is not needed because ColumnsDropper will handle those cases
    cols_not_in_cns: Set[str] = set(all_feats) - set(cns_list)
    not_in_cns_msg = f"Columns not in cols_nan_strategy:\n{cols_not_in_cns}"

    assert cols_not_in_cns == set(), not_in_cns_msg
