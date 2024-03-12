from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import open_dict
from sklearn.pipeline import Pipeline

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from src.models.hp_tunning.hyperparameter_tuning import (HPTunerConfig,
                                                         HyperparametersTuner,
                                                         get_base_model)
from src.models.train.train_model import Metric, Model
from src.utils import (Dataset, add_prefix, get_X_set, get_y_set,
                       train_test_split_custom)

# NOTE: Could be used as arguments in main and parsed
CONFIG_PATH: str = str(Path().absolute() / "config" / "hyperparameters_tuning")
BASE_MODEL_PATH: str = str(Path().absolute() / "models" / "base")
CONFIG_FILE_NAME: str = "hyperparameters_tuning"

HYDRA_VERSION_BASE: str = "1.3.1"  # NOTE: Could be in config


@dataclass
class HPConfig:
    test_size: float
    random_seed: int
    label_col: str
    initial_build_verbose: int
    features_builder_verbose: int

    model_type: str
    metric: str

    hyperparameter_tuning: HPTunerConfig


cs: ConfigStore = ConfigStore.instance()
cs.store(name="hp_tuning", node=HPConfig)


def setup_metric(cfg):
    # NOTE: cfg is of type DictConfig
    with open_dict(cfg):
        metric_name: str = cfg.metric
        metric: Metric = Metric.from_name(metric_name)
        cfg.metric = metric


def create_param_grid(predictor_name: str, model_hyperparams: Any) -> Dict[str, Any]:
    param_grid: Dict[str, Any] = {}
    separator = "__"

    # Model param grid
    predictor_prefix: str = f"{predictor_name}{separator}"
    base_model_prefix: str = f"base_model{separator}"
    model_prefix: str = f"{predictor_prefix}{base_model_prefix}"

    model_param_grid: Dict[str, Any] = add_prefix(
        prefix=model_prefix, **dict(model_hyperparams.model)
    )
    param_grid = {**param_grid, **model_param_grid}
    return param_grid


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_FILE_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(cfg: HPConfig):
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    features_builder: FeaturesBuilder = FeaturesBuilder()

    df_interim, metadata_interim = features_builder.initial_build(
        df=df_raw, metadata=metadata_raw, verbose=cfg.initial_build_verbose
    )

    df_train, df_test = train_test_split_custom(
        df=df_interim, test_size=cfg.test_size, random_seed=cfg.random_seed
    )

    preprocess_pipe = features_builder.make_pipeline(metadata_interim)

    df_train_prep: Dataset = pd.DataFrame(preprocess_pipe.fit_transform(df_train))

    X_train_prep: Dataset = get_X_set(df_train_prep, label_col=cfg.label_col)
    y_train_prep: Dataset = get_y_set(df_train_prep, label_col=cfg.label_col)

    hp_tuning_cfg: HPTunerConfig = cfg.hyperparameter_tuning
    metric: Metric = Metric.from_name(cfg.metric)

    model: Model = get_base_model(model_type=cfg.model_type, model_dir=BASE_MODEL_PATH)

    predictor_name: str = "predictor"
    pipeline = Pipeline([(predictor_name, model)])

    param_grid: Dict[str, Any] = create_param_grid(
        predictor_name, cfg.hyperparameter_tuning.param_grid
    )

    hyperparameter_tuner: HyperparametersTuner = HyperparametersTuner(
        cfg=hp_tuning_cfg, metric=metric
    )
    hyperparameter_tuner.start(
        estimator=pipeline, param_grid=param_grid, X=X_train_prep, y=y_train_prep
    )

    cv_results: Dict[str, Any] = hyperparameter_tuner.tuner.cv_results_  # type: ignore
    print(cv_results["mean_train_score"])
    print(cv_results["mean_test_score"])

    print(hyperparameter_tuner.tuner.best_score_)  # type: ignore
    print(hyperparameter_tuner.tuner.best_params_)  # type: ignore


if __name__ == "__main__":
    main()
