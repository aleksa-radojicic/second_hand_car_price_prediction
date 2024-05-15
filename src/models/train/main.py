from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from src.models.models import save_estimator
from src.models.train.train_model import TrainRunner, TrainRunnerConfig

CONFIG_PATH = str(Path().absolute() / "config" / "train")
CONFIG_FILE_NAME = "train"
HYDRA_VERSION_BASE = "1.3.1"


cs = ConfigStore.instance()
cs.store(name="training", node=TrainRunnerConfig)


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_FILE_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(cfg: TrainRunnerConfig):
    train_runner = TrainRunner(cfg)
    train_runner.start()
    scores = train_runner.scores_
    print(scores)

    # model = train_runner.pipeline
    # save_estimator(model, Path().absolute() / "models" / "optimized" / "gb opt trained 0.05 opt.pkl")


if __name__ == "__main__":
    main()
