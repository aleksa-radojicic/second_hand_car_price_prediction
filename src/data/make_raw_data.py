# -*- coding: utf-8 -*-
import click
import pandas as pd

from src import config
from src.db.broker import DbBroker
from src.logger import logging
from src.utils import Dataset, Metadata, save_data


class RawDataMaker:
    """Class responsible for creation of raw dataset by loading from SQL table and
    for creation of raw (empty) metadata.

    It assumes dataset is named as "dataset.pickle" and metadata as "metadata.json".

    Attributes
    -------
    output_filepath : str
        Output filepath folder to save raw dataset and metadata.
    """

    output_filepath: str

    def __init__(self, output_filepath: str):
        self.output_filepath = output_filepath

    def start(self):
        logging.info(
            f"Making raw data from the MySQL database to output filepath '{self.output_filepath}'..."
        )

        dataset = get_dataset_from_db()
        metadata = Metadata()
        save_data(self.output_filepath, dataset, metadata)
        logging.info(
            "Successfully made raw data from the MySQL database and initialized metadata."
        )


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(output_filepath: str):
    dataset_maker = RawDataMaker(output_filepath)
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
    main()
