import os
from dataclasses import dataclass
from typing import Any

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from src.logger import log_message
from src.models.train.train_model import Metric, Model, deserialize_base_model
from src.utils import Dataset


@dataclass
class HPTunerConfig:
    name: str
    param_grid: dict[str, Any]  # NOTE: Not used in HyperparametersTuner class
    cv_no: int
    random_seed: int
    n_jobs: int
    return_train_score: bool
    verbose: int
    refit: bool


class HyperparametersTuner:
    cfg: HPTunerConfig
    metric: Metric

    tuner: BaseSearchCV

    def __init__(self, cfg: HPTunerConfig, metric: Metric):
        self.cfg = cfg
        self.metric = metric

    def _create_tuner(
        self, estimator: Pipeline, param_grid: dict[str, Any]
    ) -> BaseSearchCV:
        cv = KFold(self.cfg.cv_no, shuffle=True, random_state=self.cfg.random_seed)

        tuner: BaseSearchCV = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            n_jobs=self.cfg.n_jobs,
            return_train_score=self.cfg.return_train_score,
            verbose=self.cfg.verbose,
            scoring=self.metric.scorer,
            error_score="raise",
        )
        return tuner

    def start(
        self, estimator: Pipeline, param_grid: dict[str, Any], X: Dataset, y: Dataset
    ):
        log_message(f"Tuning hyperparameters {self.cfg.name}...", self.cfg.verbose)
        tuner = self._create_tuner(estimator=estimator, param_grid=param_grid)
        tuner.fit(X=X, y=y)
        log_message(
            f"Tuned hyperparameters {self.cfg.name} successfully.", self.cfg.verbose
        )
        self.tuner = tuner


def get_base_model(model_type: str, model_dir: str) -> Model:
    """Retrieves base model deserialized from the model directory."""

    base_model: Any = deserialize_base_model(os.path.join(model_dir, model_type))

    model_name: str = model_type.split(".", 1)[0]
    model = Model(name=model_name, base_model=base_model)
    return model
