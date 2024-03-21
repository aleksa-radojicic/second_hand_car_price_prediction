import os
from dataclasses import dataclass
from typing import Any

from omegaconf.errors import ConfigKeyError
from omegaconf.omegaconf import DictConfig
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline

from src.features.build_features import FeaturesBuilder
from src.logger import log_message
from src.models.train.train_model import Metric, Model, deserialize_base_model
from src.utils import (Dataset, add_prefix, get_X_set, get_y_set, load_data,
                       train_test_split_custom)


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


@dataclass
class HPRunnerConfig:
    data_filepath: str
    base_model_filepath: str

    test_size: float
    random_seed: int
    label_col: str

    model_type: str
    metric: str

    pipeline_step_names: list[str]
    preprocessor_name: str
    predictor_name: str

    hyperparameter_tuning: HPTunerConfig


class HPRunner:
    """Responsible for loading processed data and further preparing it. Finally it performs
    tuning hyperparameters using sklearn's GridSearchCV.

    Attributes
    -------
    cfg
        Configuration containig all necessary informations for the HPRunner object.
    """

    cfg: HPRunnerConfig
    hyperparameter_tuner_: HyperparametersTuner

    def __init__(self, cfg: HPRunnerConfig):
        self.cfg = cfg

    def create_param_grid(self) -> dict[str, Any]:
        """Create parameter grid for hyperparameter tuner such
        that all hyperparameters have their appropriate prefix.

        Prefixes are added to hyperparameters for preprocessor pipeline
        containing various custom transformers and for the model (predictor).
        """

        hyperparams: DictConfig = self.cfg.hyperparameter_tuning.param_grid  # type: ignore
        separator = "__"

        # Preprocessor param grid
        preprocessor_param_grid: dict[str, Any] = {}

        for step_name in self.cfg.pipeline_step_names:
            try:
                step_name_prefix = (
                    f"{self.cfg.preprocessor_name}{separator}{step_name}{separator}"
                )
                step_param_grid = add_prefix(
                    prefix=step_name_prefix, **hyperparams[step_name]
                )
                preprocessor_param_grid = {**preprocessor_param_grid, **step_param_grid}
            # If certain hyperparameter is not set in the config file,
            # it won't raise an error.
            except ConfigKeyError as e:
                pass

        # Model param grid
        predictor_prefix = f"{self.cfg.predictor_name}{separator}"
        base_model_prefix = f"base_model{separator}"
        model_prefix = f"{predictor_prefix}{base_model_prefix}"

        model_param_grid = add_prefix(prefix=model_prefix, **dict(hyperparams.model))

        param_grid = {**preprocessor_param_grid, **model_param_grid}
        return param_grid

    def start(self):
        cfg = self.cfg
        df_processed, metadata_processed = load_data(cfg.data_filepath)

        df_train, _ = train_test_split_custom(
            df=df_processed, test_size=cfg.test_size, random_seed=cfg.random_seed
        )
        X_train = get_X_set(df_train, label_col=cfg.label_col)
        y_train = get_y_set(df_train, label_col=cfg.label_col)

        hp_tuning_cfg = cfg.hyperparameter_tuning
        metric = Metric.from_name(cfg.metric)

        features_builder = FeaturesBuilder()
        preprocess_pipe = features_builder.make_pipeline(
            cfg.pipeline_step_names, metadata_processed
        )
        model = get_base_model(
            model_type=cfg.model_type, model_dir=cfg.base_model_filepath
        )

        pipeline = Pipeline(
            [
                (cfg.preprocessor_name, preprocess_pipe),
                (cfg.predictor_name, model),
            ]
        )

        param_grid = self.create_param_grid()

        hyperparameter_tuner: HyperparametersTuner = HyperparametersTuner(
            cfg=hp_tuning_cfg, metric=metric
        )
        hyperparameter_tuner.start(
            estimator=pipeline, param_grid=param_grid, X=X_train, y=y_train
        )
        self.hyperparameter_tuner_ = hyperparameter_tuner
