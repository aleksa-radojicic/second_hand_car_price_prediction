from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from src.logger import logging
from src.models.hp_tunning.hyperparameter_tuning import (HPRunner,
                                                         HPRunnerConfig)
from src.models.models import save_estimator, save_estimator_details
from src.utils import pickle_object

# NOTE: Could be used as arguments in main and parsed
CONFIG_PATH = str(Path().absolute() / "config" / "hyperparameters_tuning")
CONFIG_FILE_NAME = "hyperparameters_tuning"
HYDRA_VERSION_BASE = "1.3.1"

ESTIMATOR_PATH = Path().absolute() / "models" / "optimized"


cs = ConfigStore.instance()
cs.store(name="hp_tuning", node=HPRunnerConfig)


def create_pipeline_details(cfg: HPRunnerConfig, hp_runner: HPRunner):
    tuner = hp_runner.hyperparameter_tuner_.tuner

    best_score_idx = tuner.cv_results_["rank_test_score"][0] - 1
    best_train_score = tuner.cv_results_["mean_train_score"][best_score_idx]
    best_train_score_std = tuner.cv_results_["std_train_score"][best_score_idx]
    best_test_score_std = tuner.cv_results_["std_test_score"][best_score_idx]

    pipeline_details = {
        "predictor": cfg.hyperparameter_tuning.name,
        "metric": cfg.metric,
        "train data dimension": hp_runner.train_data_dimension,
        "test data dimension": hp_runner.test_data_dimension,
        "hyperparameters": tuner.best_params_,
        "performances on GridSearchCV": {
            "train_score": best_train_score,
            "train_score_std": best_train_score_std,
            "test_score": tuner.best_score_,
            "test_score_std": best_test_score_std,
        },
    }
    return pipeline_details


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_FILE_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(cfg: HPRunnerConfig):
    hp_runner = HPRunner(cfg)
    hp_runner.start()

    # Refit optimized pipeline on training data
    pipeline = hp_runner.hyperparameter_tuner_.tuner.best_estimator_
    pipeline_details = create_pipeline_details(cfg, hp_runner)

    estimator_filepath = ESTIMATOR_PATH / f"{cfg.hyperparameter_tuning.name}.pkl"
    estimator_details_filepath = (
        ESTIMATOR_PATH / f"{cfg.hyperparameter_tuning.name}_details.json"
    )
    cv_results_filepath = (
        ESTIMATOR_PATH / f"{cfg.hyperparameter_tuning.name}_cv_results.pickle"
    )

    print(hp_runner.hyperparameter_tuner_.tuner.best_score_)
    print(hp_runner.hyperparameter_tuner_.tuner.best_params_)
    save_estimator(pipeline, estimator_filepath)
    logging.info(f"Saved best estimator in '{estimator_filepath}'.")
    save_estimator_details(pipeline_details, estimator_details_filepath)
    logging.info(f"Saved best estimator details in '{estimator_details_filepath}'.")
    pickle_object(cv_results_filepath, hp_runner.hyperparameter_tuner_.tuner.cv_results_)


if __name__ == "__main__":
    main()
