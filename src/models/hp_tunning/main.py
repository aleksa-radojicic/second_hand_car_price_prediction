from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import open_dict
from omegaconf.errors import ConfigKeyError
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
PIPELINE_STEP_NAMES: list[str] = [
    "ua_clean",
    "ma_clean",
    "col_dropper",
    "cat_handler",
    "nan_handler",
    "final_ct",
]
PREDICTOR_NAME: str = "predictor"


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


def create_param_grid(
    predictor_name: str,
    pipe_step_names: list[str],
    hyperparams: Any,
) -> dict[str, Any]:
    separator = "__"

    # Premodel param grid (steps in pipeline before predictor)
    premodel_param_grid: dict[str, Any] = {}

    for step_name in pipe_step_names:
        try:
            step_name_prefix: str = f"{step_name}{separator}"
            step_param_grid: dict[str, Any] = add_prefix(
                prefix=step_name_prefix, **hyperparams[step_name]
            )
            premodel_param_grid = {**premodel_param_grid, **step_param_grid}
        # If certain hyperparameter is not set in the config file,
        # it won't raise an error.
        except ConfigKeyError as e:
            pass

    # Model param grid
    predictor_prefix: str = f"{predictor_name}{separator}"
    base_model_prefix: str = f"base_model{separator}"
    model_prefix: str = f"{predictor_prefix}{base_model_prefix}"

    model_param_grid: dict[str, Any] = add_prefix(
        prefix=model_prefix, **dict(hyperparams.model)
    )

    param_grid: dict[str, Any] = {
        **premodel_param_grid,
        **model_param_grid,
    }
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

    df_train, _ = train_test_split_custom(
        df=df_interim, test_size=cfg.test_size, random_seed=cfg.random_seed
    )
    X_train: Dataset = get_X_set(df_train, label_col=cfg.label_col)
    y_train: Dataset = get_y_set(df_train, label_col=cfg.label_col)

    hp_tuning_cfg: HPTunerConfig = cfg.hyperparameter_tuning
    metric: Metric = Metric.from_name(cfg.metric)

    preprocess_pipe: Pipeline = features_builder.make_pipeline(
        PIPELINE_STEP_NAMES, metadata_interim
    )
    model: Model = get_base_model(model_type=cfg.model_type, model_dir=BASE_MODEL_PATH)

    pipeline = Pipeline([*preprocess_pipe.steps, (PREDICTOR_NAME, model)])

    param_grid: dict[str, Any] = create_param_grid(
        PREDICTOR_NAME, PIPELINE_STEP_NAMES, cfg.hyperparameter_tuning.param_grid
    )

    hyperparameter_tuner: HyperparametersTuner = HyperparametersTuner(
        cfg=hp_tuning_cfg, metric=metric
    )
    hyperparameter_tuner.start(
        estimator=pipeline, param_grid=param_grid, X=X_train, y=y_train
    )

    cv_results: dict[str, Any] = hyperparameter_tuner.tuner.cv_results_  # type: ignore
    print(cv_results["mean_train_score"])
    print(cv_results["mean_test_score"])

    print(hyperparameter_tuner.tuner.best_score_)  # type: ignore
    print(hyperparameter_tuner.tuner.best_params_)  # type: ignore


if __name__ == "__main__":
    main()
