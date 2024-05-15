import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from omegaconf.errors import ConfigKeyError
from omegaconf.omegaconf import DictConfig
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.features.build_features import FeaturesBuilder
from src.logger import log_message
from src.models.train.train_model import Metric, Model, deserialize_base_model
from src.utils import (Dataset, GeneralConfig, add_prefix, get_X_set,
                       get_y_set, load_data, load_general_cfg,
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

    tuner: GridSearchCV

    def __init__(self, cfg: HPTunerConfig, metric: Metric):
        self.cfg = cfg
        self.metric = metric

    def _create_tuner(
        self, estimator: Pipeline, param_grid: dict[str, Any]
    ) -> GridSearchCV:
        cv = KFold(self.cfg.cv_no, shuffle=True, random_state=self.cfg.random_seed)
        tuner = GridSearchCV(
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
        tuner = self._create_tuner(estimator=estimator, param_grid=param_grid)  # type: ignore

        tuner.fit(X=X, y=y.values.ravel())
        log_message(
            f"Tuned hyperparameters {self.cfg.name} successfully.", self.cfg.verbose
        )
        self.tuner = tuner


def get_base_model(
    model_type: str, model_dir: str, model_parameters: dict, random_state: int
) -> Model:
    """Retrieves base model deserialized from the model directory."""

    base_model = deserialize_base_model(os.path.join(model_dir, model_type))

    model_name: str = model_type.split(".", 1)[0]
    base_model.set_params(**model_parameters, random_state=random_state)
    # model = Model(name=model_name, base_model=base_model)
    model = Model(
        name=model_name,
        base_model=TransformedTargetRegressor(
            base_model, func=np.log1p, inverse_func=np.expm1
        ),
    )
    return model


@dataclass
class HPRunnerConfig:
    data_filepath: str
    base_model_filepath: str

    model_type: str
    metric: str

    model_parameters: dict

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
    general_cfg: GeneralConfig
    hyperparameter_tuner_: HyperparametersTuner

    def __init__(self, cfg: HPRunnerConfig):
        self.cfg = cfg
        self.general_cfg = load_general_cfg()

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
        model_prefix = f"{predictor_prefix}{base_model_prefix}regressor{separator}"

        model_param_grid = add_prefix(
            prefix=model_prefix,
            **dict(hyperparams.model),
        )

        param_grid = {**preprocessor_param_grid, **model_param_grid}
        return param_grid

    def start(self):
        cfg = self.cfg
        general_cfg = self.general_cfg

        df_processed, metadata_processed = load_data(cfg.data_filepath)

        df_train, _ = train_test_split_custom(
            df=df_processed,
            test_size=general_cfg.test_size,
            random_seed=general_cfg.random_seed,
        )
        df_train_cv, df_test_cv = train_test_split_custom(
            df=df_train,
            test_size=general_cfg.test_size,
            random_seed=general_cfg.random_seed,
        )

        X_train = get_X_set(df_train, label_col=general_cfg.label_col)
        y_train = get_y_set(df_train, label_col=general_cfg.label_col)

        hp_tuning_cfg = cfg.hyperparameter_tuning
        metric = Metric.from_name(cfg.metric)

        features_builder = FeaturesBuilder()
        preprocess_pipe = features_builder.make_pipeline(
            cfg.pipeline_step_names, metadata_processed
        )
        model = get_base_model(
            model_type=cfg.model_type,
            model_dir=cfg.base_model_filepath,
            model_parameters=cfg.model_parameters,
            random_state=general_cfg.random_seed,
        )

        pipeline = Pipeline(
            [
                (cfg.preprocessor_name, preprocess_pipe),
                (cfg.predictor_name, model),
            ]
        )
        param_grid = self.create_param_grid()

        hyperparameter_tuner = HyperparametersTuner(cfg=hp_tuning_cfg, metric=metric)
        hyperparameter_tuner.start(
            estimator=pipeline, param_grid=param_grid, X=X_train, y=y_train
        )
        self.hyperparameter_tuner_ = hyperparameter_tuner
        self.train_data_dimension = df_train_cv.shape
        self.test_data_dimension = df_test_cv.shape
