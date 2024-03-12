import copy
import os
from typing import Any, Dict

from sklearn.pipeline import Pipeline

from src.logger import logging
from src.models.models import (BaseModelConfig, Metric, Model,
                               deserialize_base_model)
from src.utils import Dataset


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


def setup_models(model_configs: list[BaseModelConfig], model_dir: str) -> list[Model]:
    """Does setup of models using base models deserialized from the model directory and
    adds provided configs in them."""

    models: list[Model] = []

    for model_cfg in model_configs:
        base_model: Any = deserialize_base_model(
            os.path.join(model_dir, model_cfg.type)
        )
        # Setup hyperparameters
        base_model.set_params(**model_cfg.hyperparameters)

        model = Model(name=model_cfg.name, base_model=base_model)
        models.append(model)

    return models
