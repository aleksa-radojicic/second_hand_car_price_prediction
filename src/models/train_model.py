from typing import Callable, Dict

import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src import config
from src.utils import Dataset


class Model:
    _rf_classifier = RandomForestRegressor(n_jobs=-1, random_state=config.RANDOM_SEED)
    all_predictors: Dict[str, RegressorMixin] = {
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

    def __init__(self, name: str):
        self.name = name
        self.predictor = self.all_predictors[name]

        self._rf_classifier.estimator_ = DecisionTreeRegressor(
            random_state=config.RANDOM_SEED
        )


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
    ):
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_metric = self.metric.compute(y_train.values, y_train_pred)
        test_metric = self.metric.compute(y_test.values, y_test_pred)

        return {"train": train_metric, "test": test_metric}


def main():
    pass


if __name__ == "__main__":
    main()
