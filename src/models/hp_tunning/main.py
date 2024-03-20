from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore

from src.models.hp_tunning.hyperparameter_tuning import (HPRunner,
                                                         HPRunnerConfig)

# NOTE: Could be used as arguments in main and parsed
CONFIG_PATH: str = str(Path().absolute() / "config" / "hyperparameters_tuning")
CONFIG_FILE_NAME: str = "hyperparameters_tuning"
HYDRA_VERSION_BASE: str = "1.3.1"  # NOTE: Could be in config

cs: ConfigStore = ConfigStore.instance()
cs.store(name="hp_tuning", node=HPRunnerConfig)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_FILE_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(cfg: HPRunnerConfig):
    hp_runner = HPRunner(cfg)
    hyperparameter_tuner = hp_runner.start()

    cv_results: dict[str, Any] = hyperparameter_tuner.tuner.cv_results_  # type: ignore
    print(cv_results["mean_train_score"])
    print(cv_results["mean_test_score"])

    print(hyperparameter_tuner.tuner.best_score_)  # type: ignore
    print(hyperparameter_tuner.tuner.best_params_)  # type: ignore


if __name__ == "__main__":
    main()
