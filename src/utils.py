import copy
import functools
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from typeguard import check_type

from src import config
from src.config import FeaturesInfo
from src.logger import logging

ColsNanStrategy = Dict[str, List[str]]
IdxToRemove = List[int]


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
    columns_nan_strategy = {
        "mean": [],
        "median": [],
        "modus": [],
        "const_unknown": [],
        "const_0": [],
        "const_false": [],
    }
    return columns_nan_strategy


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
    data: Metadata = field(default_factory=Metadata)


def pickle_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e


def json_object(file_path, obj):
    import json

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file_obj:
            json.dump(obj, file_obj, indent=1)

    except Exception as e:
        raise e


def preprocess_init(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Tuple[pd.DataFrame, FeaturesInfo]:
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


def train_test_split_custom(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )
    return df_train, df_test


def load_dataset(file_name: str, path: str) -> Dataset:
    dataset = pd.read_pickle(f"{path}/{file_name}_df.pkl")
    return dataset


def load_metadata(file_name: str, path: str) -> Metadata:
    metadata: Metadata

    with open(file=f"{path}/{file_name}_metadata.json", mode="r") as file:
        metadata_dict = json.load(file)
        metadata = Metadata(**metadata_dict)
    return metadata


def save_dataset(file_name: str, path: str, dataset: Dataset) -> None:
    dataset.to_pickle(path=f"{path}/{file_name}_df.pkl")


def save_metadata(file_name: str, path: str, metadata: Metadata) -> None:
    with open(file=f"{path}/{file_name}_metadata.json", mode="w") as file:
        json.dump(metadata, file, cls=MetadataEncoder, indent=4)
