# -*- coding: utf-8 -*-
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src import config
from src.db.broker import DbBroker
from src.logger import logging
from src.utils import initialize_features_info, json_object, pickle_object


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logging.info("Making raw data from the MySQL database...")

    dataset: pd.DataFrame = get_dataset_from_db()
    metadata: config.FeaturesInfo = get_metadata()

    output_filename = "data.pickle"
    output_meta_filename = "metadata.json"

    pickle_object(f"{output_filepath}/{output_filename}", dataset)
    json_object(f"{output_filepath}/meta/{output_meta_filename}", metadata)

    logging.info("Successfully made raw data from the MySQL database.")


def get_dataset_from_db() -> pd.DataFrame:
    db_broker = DbBroker()
    df = pd.read_sql(
        db_broker.get_all_listings_statement(),
        db_broker.engine,
        dtype_backend=config.DTYPE_BACKEND,
        index_col=config.INDEX,
    )
    df = df.rename(str, axis="columns")
    db_broker.engine.dispose()
    return df


def get_metadata() -> config.FeaturesInfo:
    features_info = initialize_features_info()
    return features_info


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
