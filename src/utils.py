import copy
import functools
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typeguard import check_type

from src.logger import logging
import yaml


@dataclass
class GeneralConfig:
    label_col: str
    index_col: str
    dtype_backend: str
    test_size: float
    random_seed: int


FeaturesInfo = dict[str, list[str]]
"""Custom type alias representing a dictionary containing information about feature categories.

Structure
---------
{
    'numerical': list[str]
        List of column names for numerical features.
    'binary': list[str]
        List of column names for binary features.
    'ordinal': list[str]
        List of column names for ordinal features.
    'nominal': list[str]
        List of column names for nominal features.
    'derived_numerical': list[str]
        List of column names for derived numerical features.
    'derived_binary': list[str]
        List of column names for derived binary features.
    'derived_ordinal': list[str]
        List of column names for derived ordinal features.
    'derived_nominal': list[str]
        List of column names for derived nominal features.
    'other': list[str]
        List of other features.
    'features_to_delete': list[str]
        List of column names for features to be deleted.
}
"""

ColsNanStrategy = dict[str, list[str]]
IdxToRemove = list[int]


def init_features_info() -> FeaturesInfo:
    features_info: FeaturesInfo = {
        "numerical": [],
        "binary": [],
        "ordinal": [],
        "nominal": [],
        "derived_numerical": [],
        "derived_binary": [],
        "derived_ordinal": [],
        "derived_nominal": [],
        "other": [],
        "features_to_delete": [],
    }
    return features_info


def init_cols_nan_strategy() -> ColsNanStrategy:
    columns_nan_strategy: ColsNanStrategy = {
        "mean": [],
        "median": [],
        "modus": [],
        "const_unknown": [],
        "const_0": [],
        "const_false": [],
    }
    return columns_nan_strategy


COLUMNS_NAN_STRATEGY_MAP: dict[str, SimpleImputer] = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "modus": SimpleImputer(strategy="most_frequent"),
    "const_unknown": SimpleImputer(strategy="constant", fill_value=-1),
    "const_0": SimpleImputer(strategy="constant", fill_value=0),
    "const_false": SimpleImputer(strategy="constant", fill_value=0),
}


def init_idx_to_remove() -> IdxToRemove:
    return []


Dataset = pd.DataFrame


@dataclass
class Metadata:
    features_info: FeaturesInfo = field(default_factory=init_features_info)
    cols_nan_strategy: ColsNanStrategy = field(default_factory=init_cols_nan_strategy)
    idx_to_remove: IdxToRemove = field(default_factory=init_idx_to_remove)


class MetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Metadata):
            return obj.__dict__
        return super().default(obj)


@dataclass
class PipelineMetadata:
    """Class representing input and output metadata and step name.

    The idea is every Transformer class should have input metadata that will expect and use.
    After the Transformer finishes its transformations it will change output metadata which
    will use next Transformer class (if present) and so on.

    Attributes
    ----------
    step_name : str
        Name of the transformer performing transformations.
    input_meta : Metadata
        Meta data that the current transformer expects.
    output_meta : Metadata
        Meta data that is the output of the current transformer, which the next transformer
        expects as its input.
    """

    step_name: str = ""
    input_meta: Metadata = field(default_factory=Metadata)
    output_meta: Metadata = field(default_factory=Metadata)

    def update_output_meta(self, output_meta: Metadata):
        self.output_meta.__dict__ = output_meta.__dict__.copy()


def create_pipeline_metadata_list(
    steps: list[str], init_metadata: Metadata
) -> list[PipelineMetadata]:
    """Create list of PipelineMetadata using list of pipeline step names and initial
    metadata for the first step.

    Every other metadata will be initialized using the default Metadata __init__.
    """

    data: list[PipelineMetadata] = []

    # Prepare meta for first iteration
    input_meta: Metadata = init_metadata
    output_meta: Metadata = Metadata()

    steps_length: int = len(steps)

    for i in range(steps_length):
        pipe_step: str = steps[i]

        data.append(PipelineMetadata(pipe_step, input_meta, output_meta))

        # Prepare meta for next iteration
        input_meta = output_meta
        output_meta = Metadata()

    return data


def pickle_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e


def unpickle_object(filepath) -> Any:
    try:
        with open(filepath, "rb") as file_obj:
            obj: Any = pickle.load(file_obj)
            return obj
    except Exception as e:
        raise e


def json_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "w") as file_obj:
            json.dump(obj, file_obj, indent=1)

    except Exception as e:
        raise e


def preprocess_init(func):
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[pd.DataFrame, FeaturesInfo]:
        if "df" in kwargs and isinstance(kwargs["df"], pd.DataFrame):
            kwargs["df"] = kwargs["df"].copy()

        metadata_name = Metadata.__name__.lower()

        if metadata_name in kwargs and check_type(kwargs[metadata_name], Metadata):
            kwargs[metadata_name] = copy.deepcopy(kwargs[metadata_name])

        result = func(*args, **kwargs)
        return result

    return wrapper


def log_feature_info_dict(features_info: FeaturesInfo, title: str, verbose: int):
    if verbose > 1:
        features_info_str = ""
        for k, v in features_info.items():
            features_info_str += f"{k}: {v}\n"
        logging.info(f"FeaturesInfo after {title}:\n" + features_info_str)


def train_test_split_custom(
    df: Dataset, test_size: float, random_seed: int
) -> tuple[Dataset, Dataset]:
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
    )
    return df_train, df_test


def load_data(filepath: str) -> tuple[Dataset, Metadata]:
    dataset = load_dataset(filepath)
    metadata = load_metadata(filepath)
    return dataset, metadata


def load_dataset(filepath: str) -> Dataset:
    dataset = pd.read_pickle(f"{filepath}dataset.pickle")
    return dataset


def load_metadata(filepath: str) -> Metadata:
    metadata: Metadata

    with open(file=f"{filepath}metadata.json", mode="r") as file:
        metadata_dict = json.load(file)
        metadata = Metadata(**metadata_dict)
    return metadata


def save_yaml(filepath, obj):
    with open(filepath, "w") as file:
        yaml.dump(obj, file, sort_keys=False)


def load_yaml(filepath: str) -> Any:
    with open(filepath, "r") as file:
        content = yaml.safe_load(file)
        return content


def load_general_cfg() -> GeneralConfig:
    filepath = os.path.join(os.getcwd(), "config", "general.yaml")
    general_cfg = GeneralConfig(**load_yaml(filepath))
    return general_cfg


def save_data(filepath: str, dataset: Dataset, metadata: Metadata):
    save_dataset(filepath, dataset)
    save_metadata(filepath, metadata)


def save_dataset(filepath: str, dataset: Dataset) -> None:
    dataset.to_pickle(f"{filepath}dataset.pickle")


def save_metadata(filepath: str, metadata: Metadata) -> None:
    with open(file=f"{filepath}metadata.json", mode="w") as file:
        json.dump(metadata, file, cls=MetadataEncoder, indent=4)


def get_X_set(df: Dataset, label_col: str) -> Dataset:
    return df.drop(label_col, axis=1)


def get_y_set(df: Dataset, label_col: str) -> Dataset:
    return df[[label_col]]


def add_prefix(prefix: str, **kwargs) -> dict[str, Any]:
    res_dict: dict[str, Any] = {}
    for k, v in kwargs.items():
        prefixed_k = f"{prefix}{k}"
        res_dict[prefixed_k] = v
    return res_dict
