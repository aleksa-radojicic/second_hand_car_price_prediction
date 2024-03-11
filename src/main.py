from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from src.models.train_model import (HPTunerType, HyperparametersTuner, Metric,
                                    ModelConfig, Runner, set_random_seed)
from src.utils import Dataset, get_X_set, get_y_set, train_test_split_custom


@dataclass
class Config:
    test_size: float
    random_seed: int
    label_col: str

    initial_build_verbose: int
    features_builder_verbose: int

    models: list[ModelConfig]
    metric: str  # NOTE: Hydra DictConfig doesn't support Literal type hint


cs: ConfigStore = ConfigStore.instance()
cs.store(name="scraping", node=Config)

CONFIG_PATH: str = str(Path().absolute() / "config")


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3.1")
def main(cfg: Config):
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    df_interim, metadata_interim = FeaturesBuilder().initial_build(
        df=df_raw, metadata=metadata_raw, verbose=cfg.initial_build_verbose
    )

    df_train, df_test = train_test_split_custom(
        df=df_interim, test_size=cfg.test_size, random_seed=cfg.random_seed
    )

    preprocess_pipe = FeaturesBuilder.make_pipeline(
        metadata_interim, verbose=cfg.features_builder_verbose
    )

    df_train_prep: Dataset = pd.DataFrame(preprocess_pipe.fit_transform(df_train))
    df_test_prep: Dataset = pd.DataFrame(preprocess_pipe.transform(df_test))

    X_train_prep: Dataset = get_X_set(df_train_prep, label_col=cfg.label_col)
    y_train_prep: Dataset = get_y_set(df_train_prep, label_col=cfg.label_col)

    X_test_prep: Dataset = get_X_set(df_test_prep, label_col=cfg.label_col)
    y_test_prep: Dataset = get_y_set(df_test_prep, label_col=cfg.label_col)

    # # pipeline = Pipeline([("predictor", model)])
    pipeline = Pipeline([])
    # # pipeline = Pipeline([("preprocessor", preprocess_pipe)])

    model_configs: List[ModelConfig] = cfg.models
    set_random_seed(model_configs=model_configs, random_seed=cfg.random_seed)

    results: Dict[str, float] = Runner(
        model_configs=model_configs, metric=Metric.from_name(cfg.metric)
    ).start(
        pipeline=pipeline,
        X_train=X_train_prep,
        y_train=y_train_prep,
        X_test=X_test_prep,
        y_test=y_test_prep,
    )
    print(results)

    # param_grid = {"predictor__base_model__alpha": np.linspace(0, 1, 100)}

    # hyperparameter_tuner = HyperparametersTuner(
    #     name="Tuning 1",
    #     type=HPTunerType.GRID_SEARCH,
    #     estimator=pipeline,
    #     param_grid=param_grid,
    #     verbose=2,
    # )
    # hyperparameter_tuner.start(X_train_prep, y_train_prep)

    # cv_results = hyperparameter_tuner.base_tuner.cv_results_ # type: ignore

    # print(cv_results)
    # print(hyperparameter_tuner.base_tuner.best_score_) # type: ignore


if __name__ == "__main__":
    main()
