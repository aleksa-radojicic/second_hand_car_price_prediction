# -*- coding: utf-8 -*-
import click

from src.features.build_features import FeaturesBuilder
from src.logger import logging
from src.utils import load_data, save_data


class ProcessedDataMaker:
    """Class responsible for creation of processed dataset and metadata using
    raw dataset and metadata.

    It assumes dataset is named as "dataset.pickle" and metadata as "metadata.json".

    Attributes
    -------
    input_filepath : str
        Input filepath folder where dataset and metadata live.
    output_filepath : str
        Output filepath folder to save processed dataset and metadata.
    """

    input_filepath: str
    output_filepath: str

    def __init__(self, input_filepath: str, output_filepath: str):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath

    def start(self):
        logging.info(
            f"Making processed data from raw data '{self.input_filepath}' to '{self.output_filepath}'..."
        )

        dataset_raw, metadata_raw = load_data(self.input_filepath)

        features_builder = FeaturesBuilder()

        dataset_processed, metadata_processed = features_builder.initial_build(
            df=dataset_raw, metadata=metadata_raw
        )

        save_data(self.output_filepath, dataset_processed, metadata_processed)
        logging.info("Successfully made processed data from raw data.")


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    dataset_maker = ProcessedDataMaker(input_filepath, output_filepath)
    dataset_maker.start()


if __name__ == "__main__":
    main()
