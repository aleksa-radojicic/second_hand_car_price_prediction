from typing import Dict

import xgboost as xgb
from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src import config


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


def main():
    pass


if __name__ == "__main__":
    main()
