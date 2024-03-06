import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from src.logger import logging
from src.models.train_model import METRICS, Evaluator, Model
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

    X_train = get_X_set(df_train_prep)
    X_test = get_X_set(df_test_prep)
    y_train = get_y_set(df_train_prep)
    y_test = get_y_set(df_test_prep)

    model = Model("ridge")
    pipeline = Pipeline([("predictor", model.predictor)])

    logging.info(f"Training predictor {model.name}...")
    pipeline.fit(X_train, y_train.values)
    logging.info(f"Trained predictor {model.name} successfully.")

    evaluator = Evaluator(METRICS["r2"])
    logging.info(f"Evaluating predictor {model.name}.")
    metrics = evaluator.start(pipeline, X_train, y_train, X_test, y_test)
    logging.info(f"Evaluated predictor {model.name}.")

    print(f"train evaluation metric: {metrics['train']}")
    print(f"test evaluation metric: {metrics['test']}")


if __name__ == "__main__":
    main()
