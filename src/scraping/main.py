import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore

from src.logger import logging
from src.scraping.scraper_process import ScraperProcess, ScraperProcessConfig
from src.tor_manager import TorManagerConfig


def print_total_cars_scraped(
    processes: list[ScraperProcess], cars_scraped_total_no
) -> None:
    while any(process.is_alive() for process in processes):
        time.sleep(60)
        with cars_scraped_total_no.get_lock():
            print(f"Total cars scraped: {cars_scraped_total_no.value}")


@dataclass
class ScrapeConfig:
    index_page_url: str
    sp_offset: int
    scraper_processes: list[ScraperProcessConfig]
    tor: TorManagerConfig


cs = ConfigStore.instance()
cs.store(name="scraping", node=ScrapeConfig)

CONFIG_PATH = str(Path().absolute() / "config" / "scrape")
CONFIG_FILE_NAME = "scrape"
HYDRA_VERSION_BASE = "1.3.1"


@hydra.main(
    config_path=CONFIG_PATH,
    config_name=CONFIG_FILE_NAME,
    version_base=HYDRA_VERSION_BASE,
)
def main(cfg: ScrapeConfig):
    process_no = len(cfg.scraper_processes)

    cars_scraped_total_no = multiprocessing.Value("i", 0)
    all_scraper_processes: list[ScraperProcess] = []

    for i, scraper_process_cfg in enumerate(cfg.scraper_processes, start=1):
        start_search_page = cfg.sp_offset + i
        scraper_process = ScraperProcess(
            name=f"Process_{i}",
            cfg=scraper_process_cfg,
            tor_cfg=cfg.tor,
            index_page_url=cfg.index_page_url,
            sp_no=start_search_page,
            sp_incrementer=process_no,
            cars_scraped_total_no=cars_scraped_total_no,
        )
        all_scraper_processes.append(scraper_process)
        scraper_process.start()

    print_total_cars_scraped(all_scraper_processes, cars_scraped_total_no)
    logging.info("Main process has finished.")


if __name__ == "__main__":
    main()
