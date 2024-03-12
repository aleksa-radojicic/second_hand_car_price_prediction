import copy
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

import xgboost as xgb
from omegaconf.omegaconf import open_dict
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (make_scorer, mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.logger import logging
from src.utils import Dataset, pickle_object, unpickle_object


@dataclass
class BaseModelConfig:
    name: str
    type: str
    hyperparameters: Dict[str, Any]
    random_seed: int = 0


def set_random_seed(model_configs, random_seed: int) -> None:
    """Sets random seed in every model configuration provided."""
    
    # NOTE: model_configs is of type List[DictConfig]
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


class Evaluator:
    metric: Metric

    def __init__(self, metric: Metric):
        self.metric = metric

    def start(
        self,
        pipeline: Pipeline,
        X_train: Dataset,
        y_train: Dataset,
        X_test: Dataset,
        y_test: Dataset,
    ) -> Dict[str, float]:
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_score = self.metric.compute(y_train.values.ravel(), y_train_pred)
        test_score = self.metric.compute(y_test.values.ravel(), y_test_pred)

        scores: Dict[str, float] = {"train": train_score, "test": test_score}
        return scores


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: Dataset,
    y_train: Dataset,
    X_test: Dataset,
    y_test: Dataset,
    metric: Metric,
) -> Dict[str, float]:
    model: Model = pipeline[-1]

    # Train
    logging.info(f"Training predictor {model.name}...")
    pipeline.fit(X_train, y_train.values.ravel())
    logging.info(f"Trained predictor {model.name} successfully.")

    # Evaluate
    evaluator = Evaluator(metric)
    logging.info(f"Evaluating predictor {model.name}.")
    metrics: Dict[str, float] = evaluator.start(
        pipeline, X_train, y_train, X_test, y_test
    )
    logging.info(f"Evaluated predictor {model.name}.")

    return metrics


class Runner:
    # TODO: Add correct type hints
    models: list[Model]
    metric: Metric

    def __init__(self, models: list[Model], metric: Metric):
        self.models = models
        self.metric = metric

    def start(
        self,
        pipeline: Pipeline,
        X_train: Dataset,
        y_train: Dataset,
        X_test: Dataset,
        y_test: Dataset,
    ) -> Dict[str, float]:
        pipeline = copy.deepcopy(pipeline)

        results: Dict[str, float] = {}

        for model in self.models:
            main_pipe = copy.deepcopy(pipeline)
            main_pipe.steps.append(("predictor", model))
            scores = train_and_evaluate(
                main_pipe, X_train, y_train, X_test, y_test, self.metric
            )

            logging.info(f"Train score for {model.name}: {scores['train']}")
            logging.info(f"Test score for {model.name}: {scores['test']}")

            results[model.name] = scores["test"]
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
        return sorted_results


def serialize_base_models(dir: str) -> None:
    _rf_classifier = RandomForestRegressor()

    base_models: Dict[str, Any] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": Ridge(),
        # "svr": SVR(),
        "knn": KNeighborsRegressor(),
        "dt": DecisionTreeRegressor(),
        "ada": AdaBoostRegressor(),
        "rf": _rf_classifier,
        "xgb": xgb.XGBRegressor(),
    }

    for name, model in base_models.items():
        file_path: str = os.path.join(dir, f"{name}.pkl")
        pickle_object(file_path, model)


def deserialize_base_model(file_path: str) -> Any:
    base_model: Any = unpickle_object(file_path)
    return base_model
