import copy
import functools
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from src.config import FeaturesInfo
from typeguard import check_type


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


def init_cols_nan_strategy() -> Dict[str, List[str]]:
    columns_nan_strategy = {
        "mean": [],
        "median": [],
        "modus": [],
        "const_unknown": [],
        "const_0": [],
    }
    return columns_nan_strategy


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

        if "features_info" in kwargs and check_type(
            kwargs["features_info"], FeaturesInfo
        ):
            kwargs["features_info"] = copy.deepcopy(kwargs["features_info"])

        if "cols_nan_strategy" in kwargs:
            kwargs["cols_nan_strategy"] = copy.deepcopy(kwargs["cols_nan_strategy"])

        if "idx_to_remove" in kwargs:
            kwargs["idx_to_remove"] = copy.deepcopy(kwargs["idx_to_remove"])

        result = func(*args, **kwargs)
        return result

    return wrapper
