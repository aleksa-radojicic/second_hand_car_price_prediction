import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from omegaconf.omegaconf import open_dict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.metrics import (make_scorer, mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils import json_object, pickle_object, unpickle_object


@dataclass
class BaseModelConfig:
    name: str
    type: str
    hyperparameters: dict[str, Any]
    random_seed: int = 0


def set_random_seed(model_configs, random_seed: int) -> None:
    """Sets random seed in every model configuration provided."""

    # NOTE: model_configs is of type list[DictConfig]
    with open_dict(model_configs):
        for model_config in model_configs:
            model_config.random_seed = random_seed


class Model(BaseEstimator, RegressorMixin):
    name: str
    base_model: Any  # TODO: Add correct type hints

    def __init__(self, name: str, base_model: Any):
        self.name = name
        self.base_model = base_model

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        return self.base_model.predict(X)

    def score(self, X, y):
        return self.base_model.score(X, y)


class MetricType(Enum):
    r2 = 0
    mae = 1
    rmse = 2


class Metric:
    type: MetricType
    score_func: Callable
    greater_is_better: bool

    def __init__(self, type: MetricType, score_func: Callable, greater_is_better: bool):
        self.type = type
        self.score_func = score_func
        self.greater_is_better = greater_is_better

    @property
    def scorer(self):
        scorer = make_scorer(
            score_func=self.compute,
            greater_is_better=self.greater_is_better,
        )
        return scorer

    def compute(self, y_true, y_pred):
        metric_value = self.score_func(y_true, y_pred)
        return metric_value

    @classmethod
    def from_name(cls, type: str) -> "Metric":
        METRICS: dict[str, Metric] = {
            MetricType.r2.name: cls(
                type=MetricType.r2, score_func=r2_score, greater_is_better=True
            ),
            MetricType.mae.name: cls(
                type=MetricType.mae,
                score_func=mean_absolute_error,
                greater_is_better=False,
            ),
            MetricType.rmse.name: cls(
                type=MetricType.rmse,
                score_func=root_mean_squared_error,
                greater_is_better=False,
            ),
        }
        return METRICS[type]


def serialize_base_models(dir: str):
    _rf_classifier = RandomForestRegressor()

    base_models: dict[str, Any] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(),
        "knn": KNeighborsRegressor(),
        "dt": DecisionTreeRegressor(),
        "ada": AdaBoostRegressor(),
        "rf": _rf_classifier,
        "gb": GradientBoostingRegressor(),
    }

    for name, model in base_models.items():
        file_path: str = os.path.join(dir, f"{name}.pkl")
        pickle_object(file_path, model)


def deserialize_base_model(file_path) -> Any:
    base_model: Any = unpickle_object(file_path)
    return base_model


def save_estimator(estimator: BaseEstimator, filepath):
    pickle_object(filepath, estimator)


def save_estimator_details(details: dict[str, Any], filepath):
    json_object(filepath, details)


def load_estimator(filepath) -> BaseEstimator:
    estimator = unpickle_object(filepath)
    return estimator


def main():
    serialize_base_models(dir=str(Path().absolute() / "models" / "base"))


if __name__ == "__main__":
    main()
