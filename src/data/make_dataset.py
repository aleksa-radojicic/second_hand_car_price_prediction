# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src import config
from src.db.broker import DbBroker
from src.logger import logging
from src.utils import Dataset, Metadata, save_dataset, save_metadata


class DatasetMaker:
    def __init__(self, output_filepath):
        self.output_filepath = output_filepath

    def start(self) -> Tuple[Dataset, Metadata]:
        logging.info("Making raw data from the MySQL database...")

        dataset: pd.DataFrame = get_dataset_from_db()
        metadata = Metadata()

        output_filename = "raw"

        save_dataset(output_filename, self.output_filepath, dataset)
        save_metadata(output_filename, self.output_filepath, metadata)

        logging.info("Successfully made raw data from the MySQL database.")

        return dataset, metadata


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(output_filepath):
    dataset_maker = DatasetMaker(output_filepath)
    dataset_maker.start()


def get_dataset_from_db() -> Dataset:
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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
