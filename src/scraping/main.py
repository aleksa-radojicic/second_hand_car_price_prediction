import multiprocessing
import re
import tempfile
import time
from multiprocessing.sharedctypes import SynchronizedBase

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from sqlalchemy.exc import DatabaseError, IntegrityError

from src.config import INDEX_PAGE_URL
from src.db.broker import DbBroker
from src.exception import (IPAddressBlockedException, LabelNotGivenException,
                           ScrapingException)
from src.logger import log_by_severity, log_detailed_error, logging
from src.scraping.web_scraper import Scraper, get_listing_urls_from_page
from src.tor_manager import TorManager, TorManagerConfig

BASE_URL_SP = (
    lambda x: f"{INDEX_PAGE_URL}/auto-oglasi/pretraga?page={x}&sort=renewDate_desc&city_distance=0&showOldNew=all&with_images=1"
)


def handle_integrity_error(e: IntegrityError, error_msg):
    error_message = str(e.orig)
    match = re.search(r"'(\d+)' for key 'PRIMARY'", error_message)
    if match:
        duplicate_entry = match.group(1)
        logging.warning(f"Duplicate entry: {duplicate_entry}")
    else:
        logging.warning(error_msg)


def scraper_process(
    options: FirefoxOptions,
    SOCKSPort: int,
    start_sp_no: int,
    sp_incrementer: int,
    cars_scraped_total_no: SynchronizedBase,
):
    process_name = multiprocessing.current_process().name
    torcc = {
        "ControlPort": str(SOCKSPort + 1),
        "SOCKSPort": str(SOCKSPort),
        "DataDirectory": tempfile.mkdtemp(),
    }
    finished_flag = False
    url_sp = ""
    logging.info(f"Process {process_name} has started.")
    try:
        with TorManager(options, torcc).manage() as tor_manager:
            while not finished_flag:
                url_sp = BASE_URL_SP(start_sp_no)
                exception_msg_sp = lambda e: f"{str(e)} for sp {url_sp}"
                try:
                    DbBroker()  # Mainly for checking connection with the db
                    tor_manager.load_url(url_sp, TorManagerConfig.URL_SP_LOAD_TIMEOUT)
                    logging.info("Loaded sp successfully.")
                    cars_scraped_per_sp_no = 0

                    for url_listing in get_listing_urls_from_page(tor_manager.driver):
                        exception_msg_listing = (
                            lambda e: f"{str(e)} for listing {url_listing}"
                        )
                        try:
                            tor_manager.load_url(
                                url_listing, TorManagerConfig.URL_LISTING_LOAD_TIMEOUT
                            )
                            listing = Scraper(tor_manager.driver).scrape_listing()
                            DbBroker().save_listing(listing)
                            cars_scraped_per_sp_no += 1
                            with cars_scraped_total_no.get_lock():
                                cars_scraped_total_no.value += 1
                        except IntegrityError as e:
                            handle_integrity_error(e, exception_msg_listing(e))
                        except (ScrapingException, LabelNotGivenException) as e:
                            log_by_severity(e, exception_msg_listing(e))
                        except IPAddressBlockedException as e:
                            logging.warning(exception_msg_listing(e))
                            tor_manager.get_new_tor_circuit()
                        except Exception as e:
                            logging.error(exception_msg_listing(e))
                            break
                    start_sp_no += sp_incrementer
                    logging.info(f"Scraping from sp {url_sp} completed.")
                    logging.info(f"Cars on sp scraped: {cars_scraped_per_sp_no}.")
                except IPAddressBlockedException as e:
                    logging.warning(exception_msg_sp(e))
                    tor_manager.get_new_tor_circuit()
                except (ConnectionError, WebDriverException) as e:
                    finished_flag = True
                    logging.warning(exception_msg_sp(e))
                except DatabaseError as e:
                    finished_flag = True
                    logging.warning(str(e))
                except Exception as e:
                    log_detailed_error(e, exception_msg_sp(e))
    except (OSError, Exception) as e:
        log_detailed_error(e, str(e))

    logging.info(f"Process {process_name} has finished.")


def print_total_cars_scraped(processes, cars_scraped_total_no):
    while any(process.is_alive() for process in processes):
        time.sleep(60)
        with cars_scraped_total_no.get_lock():
            print(f"Total cars scraped: {cars_scraped_total_no.value}")


def main():
    logging.info("Main process has started.")

    SOCKSPorts = [9250, 9350, 9450]
    process_no = len(SOCKSPorts)
    sp_offset = 0

    TorManagerConfig.HEADLESS_MODE = True

    options = FirefoxOptions()
    options.set_preference("permissions.default.image", 2)
    options.set_preference("permissions.default.stylesheet", 2)

    cars_scraped_total_no = multiprocessing.Value("i", 0)

    processes = []
    for i, SOCKSPort in enumerate(SOCKSPorts, start=1):
        start_sp_no = sp_offset + i
        process = multiprocessing.Process(
            target=scraper_process,
            args=(options, SOCKSPort, start_sp_no, process_no, cars_scraped_total_no),
            name=f"Process_{i}",
        )
        processes.append(process)
        process.start()

    print_total_cars_scraped(processes, cars_scraped_total_no)
    logging.info("Main process has finished.")


if __name__ == "__main__":
    main()
