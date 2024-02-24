import collections
import inspect
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import Markdown
from IPython.core.display_functions import display
from scipy import stats

from src import config
from src.config import FeaturesInfo

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
    dataset: pd.DataFrame,
    metadata: FeaturesInfo,
    cols_nan_strategy: Optional[Dict[str, List[str]]] = None,
    idx_to_remove: Optional[List[int]] = None,
):
    with open(file=f"{path}/{file_name}_features_info.json", mode="w") as file:
        json.dump(metadata, file, indent=4)
    dataset.to_pickle(path=f"{path}/{file_name}_df.pkl")

    if cols_nan_strategy:
        with open(file=f"{path}/{file_name}_cols_nan_strategy.json", mode="w") as file:
            json.dump(cols_nan_strategy, file, indent=4)

    if idx_to_remove:
        with open(file=f"{path}/{file_name}_idx_to_remove.json", mode="w") as file:
            json.dump(idx_to_remove, file, indent=4)


def load_dataset_and_metadata(
    file_name: str, path: str
) -> Tuple[pd.DataFrame, FeaturesInfo]:
    with open(file=f"{path}/{file_name}_features_info.json", mode="r") as file:
        metadata: FeaturesInfo = json.load(file)
        data = pd.read_pickle(f"{path}/{file_name}_df.pkl")

        return data, metadata


def test_features_info_and_columns(df: pd.DataFrame, features_info: FeaturesInfo):
    all_feats = []
    for key in features_info:
        if key == "features_to_delete":
            continue
        all_feats.extend(features_info[key])

    # Get duplicate features
    dupe_feats = [
        feat for feat, count in collections.Counter(all_feats).items() if count > 1
    ]
    if dupe_feats:
        raise AssertionError(
            f"Features info contains following duplicate features: {dupe_feats}"
        )
    all_cols = df.drop(config.LABEL, axis=1).columns.tolist()

    not_in_right_msg = f"Features not in right: {set(all_feats) - set(all_cols)}"
    not_in_left_msg = f"Features not in left: {set(all_cols) - set(all_feats)}"
    error_msg = f"{not_in_right_msg}\n{not_in_left_msg}"

    assert set(all_feats) == set(all_cols), error_msg
