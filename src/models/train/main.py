from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from sklearn.pipeline import Pipeline

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder, FeaturesBuilderConfig
from src.models.models import BaseModelConfig, set_random_seed
from src.models.train.train_model import Metric, Runner, setup_models
from src.utils import Dataset, get_X_set, get_y_set, train_test_split_custom

CONFIG_PATH: str = str(Path().absolute() / "config" / "train")
BASE_MODEL_PATH: str = str(Path().absolute() / "models" / "base")
CONFIG_FILE_NAME: str = "train"
HYDRA_VERSION_BASE: str = "1.3.1"  # NOTE: Could be in config
PIPELINE_STEP_NAMES: list[str] = [
    "ua_clean",
    "ma_clean",
    "col_dropper",
    "cat_handler",
    "nan_handler",
    "final_ct",
]
PREPROCESSOR_NAME: str = "preprocessor"


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

    features_builder = FeaturesBuilder(cfg.features_builder)

    df_interim, metadata_interim = features_builder.initial_build(
        df=df_raw,
        metadata=metadata_raw,
        verbose=cfg.initial_build_verbose,
    )

    df_train, df_test = train_test_split_custom(
        df=df_interim, test_size=cfg.test_size, random_seed=cfg.random_seed
    )

    X_train: Dataset = get_X_set(df_train, label_col=cfg.label_col)
    y_train: Dataset = get_y_set(df_train, label_col=cfg.label_col)

    X_test: Dataset = get_X_set(df_test, label_col=cfg.label_col)
    y_test: Dataset = get_y_set(df_test, label_col=cfg.label_col)

    preprocess_pipe = features_builder.make_pipeline(
        PIPELINE_STEP_NAMES, metadata_interim, cfg.features_builder.verbose
    )

    model_configs = cfg.models
    set_random_seed(model_configs=model_configs, random_seed=cfg.random_seed)
    metric = Metric.from_name(cfg.metric)
    models = setup_models(model_configs=model_configs, model_dir=BASE_MODEL_PATH)

    pipeline = Pipeline([(PREPROCESSOR_NAME, preprocess_pipe)])

    runner = Runner(models=models, metric=metric)
    results = runner.start(
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print(results)


if __name__ == "__main__":
    main()
