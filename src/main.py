import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.make_dataset import DatasetMaker
from src.features.build_features import FeaturesBuilder
from src.models.train_model import MODELS, Model, Runner
from src.utils import Dataset, train_test_split_custom


def main():
    df_raw, metadata_raw = DatasetMaker("data/raw").start()

    df_interim, metadata_interim = FeaturesBuilder(verbose=2).initial_build(
        df=df_raw, metadata=metadata_raw
    )

    df_train, df_test = train_test_split_custom(df_interim)

    preprocess_pipe = FeaturesBuilder.make_pipeline(metadata_interim, verbose=2)

    df_train_prep: Dataset = pd.DataFrame(preprocess_pipe.fit_transform(df_train))
    df_test_prep: Dataset = pd.DataFrame(preprocess_pipe.transform(df_test))

    pipeline = Pipeline([])

    model_names = ["ridge", "dt", "dummy_mean", "dummy_median"]
    models: list[Model] = [model for model_name, model in MODELS.items() if model_name in model_names]
    Runner(models).start(pipeline, df_train_prep, df_test_prep)


if __name__ == "__main__":
    main()
