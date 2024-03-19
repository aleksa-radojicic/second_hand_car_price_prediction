import multiprocessing
import time
from dataclasses import dataclass
from multiprocessing.sharedctypes import SynchronizedBase
from pathlib import Path
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore

from src.logger import logging
from src.scraping.scraper_process import ScraperProcess, ScraperProcessConfig
from src.tor_manager import TorManagerConfig


def print_total_cars_scraped(processes, cars_scraped_total_no) -> None:
    while any(process.is_alive() for process in processes):
        time.sleep(60)
        with cars_scraped_total_no.get_lock():
            print(f"Total cars scraped: {cars_scraped_total_no.value}")


@dataclass
class ScrapeConfig:
    sp_offset: int
    scraper_processes: list[ScraperProcessConfig]
    tor: TorManagerConfig


cs: ConfigStore = ConfigStore.instance()
cs.store(name="scraping", node=ScrapeConfig)

CONFIG_PATH: str = str(Path().absolute() / "config" / "scrape")


@hydra.main(config_path=CONFIG_PATH, config_name="scrape", version_base="1.3.1")
def main(cfg: ScrapeConfig):
    scraper_processes_configs: list[ScraperProcessConfig] = cfg.scraper_processes
    process_no: int = len(scraper_processes_configs)

    cars_scraped_total_no: SynchronizedBase[Any] = multiprocessing.Value("i", 0)
    scraper_processes: list[ScraperProcess] = []

    for i, scraper_process_config in enumerate(scraper_processes_configs, start=1):
        start_search_page: int = cfg.sp_offset + i
        scraper_process = ScraperProcess(
            name=f"Process_{i}",
            cfg=scraper_process_config,
            tor_cfg=cfg.tor,
            sp_no=start_search_page,
            sp_incrementer=process_no,
            cars_scraped_total_no=cars_scraped_total_no,
        )
        scraper_processes.append(scraper_process)
        scraper_process.start()

    print_total_cars_scraped(
        processes=scraper_processes, cars_scraped_total_no=cars_scraped_total_no
    )
    logging.info("Main process has finished.")


if __name__ == "__main__":
    main()
