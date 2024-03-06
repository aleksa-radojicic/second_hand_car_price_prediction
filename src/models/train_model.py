import copy
from typing import Any, Callable, Dict

import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.logger import logging
from src.utils import Dataset, get_X_set, get_y_set


class Model(BaseEstimator, RegressorMixin):
    name: str
    # TODO: Add correct typehint
    base_model: Any

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


_rf_classifier = RandomForestRegressor(n_jobs=-1, random_state=config.RANDOM_SEED)
MODELS: Dict[str, Model] = {
    "dummy_mean": Model("dummy_mean", DummyRegressor(strategy="mean")),
    "dummy_median": Model("dummy_median", DummyRegressor(strategy="median")),
    "ridge": Model("ridge", Ridge(random_state=config.RANDOM_SEED)),
    # "svr": Model("svr", SVR()),
    "knn": Model("knn", KNeighborsRegressor(n_jobs=-1)),
    "dt": Model("dt", DecisionTreeRegressor(random_state=config.RANDOM_SEED)),
    "ada": Model("ada", AdaBoostRegressor(random_state=config.RANDOM_SEED)),
    "rf": Model("rf", _rf_classifier),
    "xgb": xgb.XGBRegressor(),
}
_rf_classifier.estimator_ = DecisionTreeRegressor(random_state=config.RANDOM_SEED)


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
            score_func=Metric.compute,
            greater_is_better=self.greater_is_better,
        )
        return scorer

    def compute(self, y_true, y_pred):
        metric_value = self.score_func(y_true, y_pred)
        return metric_value


METRICS: dict[str, Metric] = {
    "r2": Metric(name="r2", score_func=r2_score, greater_is_better=True)
}


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

        train_score = self.metric.compute(y_train.values, y_train_pred)
        test_score = self.metric.compute(y_test.values, y_test_pred)

        scores: Dict[str, float] = {"train": train_score, "test": test_score}
        return scores


def train_and_evaluate(
    pipeline: Pipeline, df_train: Dataset, df_test: Dataset
) -> Dict[str, float]:
    X_train = get_X_set(df_train)
    X_test = get_X_set(df_test)
    y_train = get_y_set(df_train)
    y_test = get_y_set(df_test)

    model: Model = pipeline[-1]

    # Train
    logging.info(f"Training predictor {model.name}...")
    pipeline.fit(X_train, y_train.values)
    logging.info(f"Trained predictor {model.name} successfully.")

    # Evaluate
    evaluator = Evaluator(METRICS["r2"])
    logging.info(f"Evaluating predictor {model.name}.")
    metrics = evaluator.start(pipeline, X_train, y_train, X_test, y_test)
    logging.info(f"Evaluated predictor {model.name}.")

    return metrics


class Runner:
    def __init__(self, models: list[Model]):
        self.models = models

    def start(self, pipeline: Pipeline, df_train: Dataset, df_test: Dataset):
        pipeline = copy.deepcopy(pipeline)

        for model in self.models:
            main_pipe = copy.deepcopy(pipeline)
            main_pipe.steps.append(("predictor", model))
            scores = train_and_evaluate(main_pipe, df_train, df_test)

            logging.info(f"Train score for {model.name}: {scores['train']}")
            logging.info(f"Test score for {model.name}: {scores['test']}")


def main():
    pass


if __name__ == "__main__":
    main()
