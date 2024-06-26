import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.features.build_features import FeaturesBuilder, FeaturesBuilderConfig
from src.logger import logging
from src.models.models import (BaseModelConfig, Metric, Model,
                               deserialize_base_model, set_random_seed)
from src.utils import (Dataset, GeneralConfig, get_X_set, get_y_set, load_data,
                       load_general_cfg, pickle_object,
                       train_test_split_custom)


def _get_model(estimator) -> Model:
    """Return model from the provided estimator.

    Estimator should either be a Model or a Pipeline containing a Model."""

    model: Model

    if isinstance(estimator, Model):
        model = estimator
    # Estimator is a Pipeline
    elif isinstance(estimator[-1], Model):
        model = estimator[-1]
    else:
        raise ValueError(
            f"Provided estimator {estimator} doesn't contain Model object."
        )

    return model


class Trainer:
    """Responsible for training an estimator containing Model instance."""

    def __init__(self, estimator):
        self.estimator = estimator

    def start(self, X_train: Dataset, y_train: Dataset):
        model = _get_model(self.estimator)

        logging.info(f"Training predictor {model.name}...")
        self.estimator.fit(X_train, y_train.values.ravel())
        logging.info(f"Trained predictor {model.name} successfully.")


class Evaluator:
    """Responsible for evaluating an estimator containing Model instance using
    the provided Metric object."""

    metric: Metric
    scores_: dict[str, float]

    def __init__(self, estimator, metric: Metric):
        self.estimator = estimator
        self.metric = metric

    def start(
        self,
        X_train: Dataset,
        y_train: Dataset,
        X_test: Dataset,
        y_test: Dataset,
    ):
        model = _get_model(self.estimator)
        logging.info(f"Evaluating predictor {model.name}...")

        y_train_pred: np.ndarray = self.estimator.predict(X_train)
        y_test_pred: np.ndarray = self.estimator.predict(X_test)

        train_score: float = self.metric.compute(y_train.values.ravel(), y_train_pred)
        test_score: float = self.metric.compute(y_test.values.ravel(), y_test_pred)
        logging.info(f"Evaluated predictor {model.name}.")

        self.scores_ = {"train": train_score, "test": test_score}


class ModelsTrainerEvaluator:
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
    ):
        pipeline = copy.deepcopy(pipeline)

        scores: dict[str, float] = {}

        for i, model in enumerate(self.models):
            main_pipe = copy.deepcopy(pipeline)
            main_pipe.steps.append(("predictor", model))

            if i == 0:
                self.pipeline = main_pipe

            trainer = Trainer(main_pipe)
            trainer.start(X_train, y_train)

            evaluator = Evaluator(main_pipe, self.metric)
            evaluator.start(X_train, y_train, X_test, y_test)

            current_scores = evaluator.scores_

            logging.info(f"Train score for {model.name}: {current_scores['train']}")
            logging.info(f"Test score for {model.name}: {current_scores['test']}")

            scores[model.name] = current_scores["test"]

        # Sort scores
        self.scores_ = dict(sorted(scores.items(), key=lambda x: x[1]))


@dataclass
class TrainRunnerConfig:
    data_filepath: str
    base_model_filepath: str

    models: list[BaseModelConfig]
    metric: str  # NOTE: Hydra DictConfig doesn't support Literal type hint

    pipeline_step_names: list[str]
    preprocessor_name: str
    predictor_name: str

    features_builder: FeaturesBuilderConfig


class TrainRunner:
    """Responsible for loading processed data and further preparing it. After all it trains
    provided list of models and evaluates them using the provided metric."""

    cfg: TrainRunnerConfig
    general_cfg: GeneralConfig
    scores_: dict[str, float]

    def __init__(self, cfg: TrainRunnerConfig):
        self.cfg = cfg
        self.general_cfg = load_general_cfg()

    def start(self):
        cfg = self.cfg
        general_cfg = self.general_cfg

        df_processed, metadata_processed = load_data(filepath=cfg.data_filepath)

        df_train, df_test = train_test_split_custom(
            df=df_processed,
            test_size=general_cfg.test_size,
            random_seed=general_cfg.random_seed,
        )

        X_train = get_X_set(df_train, label_col=general_cfg.label_col)
        y_train = get_y_set(df_train, label_col=general_cfg.label_col)

        X_test = get_X_set(df_test, label_col=general_cfg.label_col)
        y_test = get_y_set(df_test, label_col=general_cfg.label_col)

        filepath_pickle = Path().absolute() / "data" / "processed"
        pickle_object(filepath_pickle / "X_train_prep.pkl", X_train)
        pickle_object(filepath_pickle / "X_test_prep.pkl", X_test)
        pickle_object(filepath_pickle / "y_train_prep.pkl", y_train)
        pickle_object(filepath_pickle / "y_test_prep.pkl", y_test)

        features_builder = FeaturesBuilder(cfg.features_builder)
        preprocess_pipe = features_builder.make_pipeline(
            self.cfg.pipeline_step_names,
            metadata_processed,
            cfg.features_builder.verbose,
        )

        model_configs = cfg.models
        set_random_seed(
            model_configs=model_configs, random_seed=general_cfg.random_seed
        )
        metric = Metric.from_name(cfg.metric)
        models = setup_models(
            model_configs=model_configs, model_dir=self.cfg.base_model_filepath
        )

        pipeline = Pipeline([(self.cfg.preprocessor_name, preprocess_pipe)])

        trainer_evaluator = ModelsTrainerEvaluator(models=models, metric=metric)
        trainer_evaluator.start(
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        self.scores_ = trainer_evaluator.scores_
        self.models_ = models
        self.X_train_ = X_train
        self.y_train_ = y_train
        self.X_test_ = X_test
        self.y_test_ = y_test

        self.pipeline = trainer_evaluator.pipeline


def setup_models(model_configs: list[BaseModelConfig], model_dir: str) -> list[Model]:
    """Does setup of models using base models deserialized from the model directory and
    adds provided configs in them."""

    models: list[Model] = []

    for model_cfg in model_configs:
        base_model = deserialize_base_model(os.path.join(model_dir, model_cfg.type))
        # Setup hyperparameters
        base_model.random_state = model_cfg.random_seed
        try:
            base_model.set_params(**model_cfg.hyperparameters)
        except Exception as e:  # ConfigAttributeError
            # When hyperparameters config is not set
            pass

        model = Model(
            name=model_cfg.name,
            base_model=TransformedTargetRegressor(
                base_model, func=np.log1p, inverse_func=np.expm1
            ),
        )
        models.append(model)

    return models
