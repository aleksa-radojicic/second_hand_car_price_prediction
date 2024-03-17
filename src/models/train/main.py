from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from sklearn.pipeline import Pipeline

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder, FeaturesBuilderConfig
from src.models.models import BaseModelConfig, set_random_seed
from src.models.train.train_model import Metric, Model, Runner, setup_models
from src.utils import (Dataset, get_X_set, get_y_set,
                       train_test_split_custom)

CONFIG_PATH: str = str(Path().absolute() / "config" / "train")
BASE_MODEL_PATH: str = str(Path().absolute() / "models" / "base")
CONFIG_FILE_NAME: str = "train"

HYDRA_VERSION_BASE: str = "1.3.1"  # NOTE: Could be in config


@dataclass
class TrainConfig:
    test_size: float
    random_seed: int
    label_col: str

    initial_build_verbose: int

    models: list[BaseModelConfig]
    metric: str  # NOTE: Hydra DictConfig doesn't support Literal type hint

    features_builder: FeaturesBuilderConfig


cs: ConfigStore = ConfigStore.instance()
cs.store(name="training", node=TrainConfig)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_FILE_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(cfg: TrainConfig):
    # serialize_base_models(os.path.join("models", "base"))
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    features_builder: FeaturesBuilder = FeaturesBuilder(cfg.features_builder)

    df_interim, metadata_interim = features_builder.initial_build(
        df=df_raw,
        metadata=metadata_raw,
        verbose=cfg.initial_build_verbose,
    )

    df_train, df_test = train_test_split_custom(
        df=df_interim, test_size=cfg.test_size, random_seed=cfg.random_seed
    )

    preprocess_pipe: Pipeline = features_builder.make_pipeline(
        metadata_interim, cfg.features_builder.verbose
    )

    df_train_prep: Dataset = pd.DataFrame(preprocess_pipe.fit_transform(df_train))
    df_test_prep: Dataset = pd.DataFrame(preprocess_pipe.transform(df_test))

    X_train_prep: Dataset = get_X_set(df_train_prep, label_col=cfg.label_col)
    y_train_prep: Dataset = get_y_set(df_train_prep, label_col=cfg.label_col)

    X_test_prep: Dataset = get_X_set(df_test_prep, label_col=cfg.label_col)
    y_test_prep: Dataset = get_y_set(df_test_prep, label_col=cfg.label_col)

    model_configs: List[BaseModelConfig] = cfg.models
    set_random_seed(model_configs=model_configs, random_seed=cfg.random_seed)
    metric: Metric = Metric.from_name(cfg.metric)
    models: List[Model] = setup_models(
        model_configs=model_configs, model_dir=BASE_MODEL_PATH
    )

    pipeline = Pipeline([])

    runner: Runner = Runner(models=models, metric=metric)
    results: Dict[str, float] = runner.start(
        pipeline=pipeline,
        X_train=X_train_prep,
        y_train=y_train_prep,
        X_test=X_test_prep,
        y_test=y_test_prep,
    )
    print(results)

if __name__ == "__main__":
    main()
