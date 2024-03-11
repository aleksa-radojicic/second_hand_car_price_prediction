import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, Literal

import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (make_scorer, mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.logger import log_message, logging
from src.utils import Dataset, get_X_set, get_y_set


@dataclass
class ModelConfig:
    name: str
    type: str
    hyperparameters: Dict[str, Any]


class Model(BaseEstimator, RegressorMixin):
    name: str
    cfg: ModelConfig

    # TODO: Add correct typehints
    _all_base_models: Dict[str, Any]
    base_model: Any

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self._all_base_models = self._create_all_base_models()
        self.base_model = self.create_base_model()

    def _create_all_base_models(self) -> Dict[str, Any]:
        _rf_classifier = RandomForestRegressor(
            n_jobs=-1, random_state=config.RANDOM_SEED
        )
        all_base_models: Dict[str, Any] = {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "dummy_median": DummyRegressor(strategy="median"),
            "ridge": Ridge(random_state=config.RANDOM_SEED),
            # "svr": SVR(),
            "knn": KNeighborsRegressor(n_jobs=-1),
            "dt": DecisionTreeRegressor(random_state=config.RANDOM_SEED),
            "ada": AdaBoostRegressor(random_state=config.RANDOM_SEED),
            "rf": _rf_classifier,
            "xgb": xgb.XGBRegressor(),
        }
        return all_base_models

    def create_base_model(self) -> Any:
        base_model: Any = self._all_base_models[self.cfg.type]
        base_model.set_params(**self.cfg.hyperparameters)
        return base_model

    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        return self.base_model.predict(X)

    def score(self, X, y):
        return self.base_model.score(X, y)


class Metric:
    name: str
    score_func: Callable
    greater_is_better: bool

    def __init__(self, name: str, score_func: Callable, greater_is_better: bool):
        self.name = name
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
    def from_name(cls, name: Literal["r2", "mae", "rmse"]) -> "Metric":
        METRICS: dict[str, Metric] = {
            "r2": cls(name="r2", score_func=r2_score, greater_is_better=True),
            "mae": cls(
                name="mae", score_func=mean_absolute_error, greater_is_better=False
            ),
            "rmse": cls(
                name="rmse",
                score_func=root_mean_squared_error,
                greater_is_better=False,
            ),
        }
        return METRICS[name]


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
    pipeline: Pipeline, df_train: Dataset, df_test: Dataset, metric: Metric
) -> Dict[str, float]:
    X_train = get_X_set(df_train)
    X_test = get_X_set(df_test)
    y_train = get_y_set(df_train)
    y_test = get_y_set(df_test)

    model: Model = pipeline[-1]

    # Train
    logging.info(f"Training predictor {model.cfg.name}...")
    pipeline.fit(X_train, y_train.values.ravel())
    logging.info(f"Trained predictor {model.cfg.name} successfully.")

    # Evaluate
    evaluator = Evaluator(metric)
    logging.info(f"Evaluating predictor {model.cfg.name}.")
    metrics: Dict[str, float] = evaluator.start(
        pipeline, X_train, y_train, X_test, y_test
    )
    logging.info(f"Evaluated predictor {model.cfg.name}.")

    return metrics


class Runner:
    # TODO: Add correct typehints
    models: list[Model]
    metric: Metric

    def __init__(self, model_configs: list[ModelConfig], metric: Metric):
        self.models = self.create_models(model_configs)
        self.metric = metric

    def create_models(self, model_configs: list[ModelConfig]) -> list[Model]:
        models: list[Model] = [Model(model_config) for model_config in model_configs]
        return models

    def start(
        self, pipeline: Pipeline, df_train: Dataset, df_test: Dataset
    ) -> Dict[str, float]:
        pipeline = copy.deepcopy(pipeline)

        results: Dict[str, float] = {}

        for model in self.models:
            main_pipe = copy.deepcopy(pipeline)
            main_pipe.steps.append(("predictor", model))
            scores = train_and_evaluate(main_pipe, df_train, df_test, self.metric)

            logging.info(f"Train score for {model.cfg.name}: {scores['train']}")
            logging.info(f"Test score for {model.cfg.name}: {scores['test']}")

            results[model.cfg.name] = scores["test"]
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]))
        return sorted_results


class HPTunerType(Enum):
    GRID_SEARCH = auto()


class HyperparametersTuner:
    name: str
    metric: Metric
    verbose: int

    base_tuner: BaseSearchCV
    _tuner_map: Dict[HPTunerType, BaseSearchCV]

    def __init__(
        self,
        name: str,
        type: HPTunerType,
        estimator: BaseEstimator,
        param_grid,
        metric: Metric,
        verbose: int,
    ):
        self.name = name
        self.estimator = estimator
        self.metric = metric
        self.verbose = verbose
        self.param_grid = param_grid

        self._init_tuner_map()
        self.base_tuner = self._tuner_map[type]

    def _init_tuner_map(self):
        cv_no = 5
        cv = KFold(cv_no, shuffle=True, random_state=config.RANDOM_SEED)

        self._tuner_map = {
            HPTunerType.GRID_SEARCH: GridSearchCV(
                self.estimator,
                param_grid=self.param_grid,
                cv=cv,
                n_jobs=-1,
                return_train_score=True,
                verbose=self.verbose,
                # refit=True,
                scoring=self.metric.scorer,
                error_score="raise",
            )
        }

    def start(self, X: Dataset, y: Dataset) -> None:
        log_message(f"Tuning hyperparameters {self.name}...", self.verbose)
        self.base_tuner.fit(X=X, y=y)
        log_message(f"Tuned hyperparameters {self.name} successfully.", self.verbose)


def main():
    pass


if __name__ == "__main__":
    main()
