import multiprocessing
import time
from typing import List

from selenium.webdriver.firefox.options import Options as FirefoxOptions

from src.logger import logging
from src.scraping.scraper_process import ScraperProcess
from src.tor_manager import TorManagerConfig


def print_total_cars_scraped(processes, cars_scraped_total_no):
    while any(process.is_alive() for process in processes):
        time.sleep(60)
        with cars_scraped_total_no.get_lock():
            print(f"Total cars scraped: {cars_scraped_total_no.value}")


def main():
    logging.info("Main process has started.")

    SOCKSPorts = [9250, 9350, 9450]
    process_no = len(SOCKSPorts)
    sp_offset = 1800

    TorManagerConfig.HEADLESS_MODE = True

    options = FirefoxOptions()
    options.set_preference("permissions.default.image", 2)
    options.set_preference("permissions.default.stylesheet", 2)

    cars_scraped_total_no = multiprocessing.Value("i", 0)

    scraper_processes: List[ScraperProcess] = []
    for i, SOCKSPort in enumerate(SOCKSPorts, start=1):
        start_sp_no = sp_offset + i
        scraper_process = ScraperProcess(
            f"Process_{i}",
            options,
            SOCKSPort,
            start_sp_no,
            process_no,
            cars_scraped_total_no,
        )
        scraper_processes.append(scraper_process)
        scraper_process.start()

    print_total_cars_scraped(scraper_processes, cars_scraped_total_no)
    logging.info("Main process has finished.")


if __name__ == "__main__":
    main()
