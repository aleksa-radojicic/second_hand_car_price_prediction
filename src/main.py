import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src import config
from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from src.models.train_model import (METRICS, MODELS, HPTunerType,
                                    HyperparametersTuner, Model, Runner)
from src.utils import Dataset, get_X_set, get_y_set, train_test_split_custom


def main():
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    df_interim, metadata_interim = FeaturesBuilder(verbose=2).initial_build(
        df=df_raw, metadata=metadata_raw
    )

    df_train, df_test = train_test_split_custom(df_interim)

    preprocess_pipe = FeaturesBuilder.make_pipeline(metadata_interim, verbose=2)

    df_train_prep: Dataset = pd.DataFrame(preprocess_pipe.fit_transform(df_train))
    df_test_prep: Dataset = pd.DataFrame(preprocess_pipe.transform(df_test))

    # X_train_prep = get_X_set(df_train_prep)
    # y_train_prep = get_y_set(df_train_prep)

    model: Model = MODELS["ridge"]
    # pipeline = Pipeline([("predictor", model)])
    pipeline = Pipeline([])
    # pipeline = Pipeline([("preprocessor", preprocess_pipe)])

    # model_names = ["ridge", "dt", "dummy_mean", "dummy_median"]
    model_names = list(MODELS.keys())
    models: list[Model] = [
        model for model_name, model in MODELS.items() if model_name in model_names
    ]
    results = Runner(models, METRICS["mae"]).start(
        pipeline, df_train_prep, df_test_prep
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
