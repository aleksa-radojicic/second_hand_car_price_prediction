import multiprocessing
import re
import tempfile
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
    lambda x: f"{INDEX_PAGE_URL}/auto-oglasi/pretraga?page={x}sort=basic&city_distance=0&showOldNew=all"
)


class ScraperProcess(multiprocessing.Process):
    def __init__(
        self,
        name: str,
        options: FirefoxOptions,
        SOCKSPort: int,
        sp_no: int,
        sp_incrementer: int,
        cars_scraped_total_no: SynchronizedBase,
    ):
        super(ScraperProcess, self).__init__()
        self.name = name
        self.options = options
        self.SOCKSPort = SOCKSPort
        self.sp_no = sp_no
        self.sp_incrementer = sp_incrementer
        self.cars_scraped_total_no = cars_scraped_total_no
        self.cars_scraped_per_sp_no = 0

    def _handle_integrity_error(self, e: IntegrityError, error_msg):
        error_message = str(e.orig)
        match = re.search(r"'(\d+)' for key 'PRIMARY'", error_message)
        if match:
            duplicate_entry = match.group(1)
            logging.warning(f"Duplicate entry: {duplicate_entry}")
        else:
            logging.warning(error_msg)

    def scrape_and_save_listing(self, url_listing, tor_manager, exception_msg_sp):
        exception_msg_listing = lambda e: f"{str(e)} for listing {url_listing}"
        try:
            tor_manager.load_url(
                url_listing,
                TorManagerConfig.URL_LISTING_LOAD_TIMEOUT,
            )
            listing = Scraper(tor_manager.driver).scrape_listing()
            DbBroker().save_listing(listing)
            self.cars_scraped_per_sp_no += 1
            with self.cars_scraped_total_no.get_lock():
                self.cars_scraped_total_no.value += 1  # type: ignore
        except IntegrityError as e:
            self._handle_integrity_error(e, exception_msg_listing(e))
        except (ScrapingException, LabelNotGivenException) as e:
            log_by_severity(e, exception_msg_listing(e))
        except IPAddressBlockedException as e:
            logging.warning(exception_msg_listing(e))
            tor_manager.get_new_tor_circuit()
        # Page couldn't load
        except TypeError as e:
            pass
        except Exception as e:
            log_detailed_error(e, exception_msg_sp(e))
            return "break"

    def _scrape_and_save_listings_from_sp(self, tor_manager, exception_msg_sp):
        self.cars_scraped_per_sp_no = 0

        for url_listing in get_listing_urls_from_page(tor_manager.driver):
            signal = self.scrape_and_save_listing(
                url_listing, tor_manager, exception_msg_sp
            )
            if signal:
                break

        self.sp_no += self.sp_incrementer

    def run(self):
        torcc = {
            "ControlPort": str(self.SOCKSPort + 1),
            "SOCKSPort": str(self.SOCKSPort),
            "DataDirectory": tempfile.mkdtemp(),
        }
        finished_flag = False
        url_sp = ""
        logging.info(f"Process {self.name} has started.")
        try:
            DbBroker()  # Mainly for checking connection with the db
            with TorManager(self.options, torcc).manage() as tor_manager:
                while not finished_flag:
                    url_sp = BASE_URL_SP(self.sp_no)
                    exception_msg_sp = lambda e: f"{str(e)} for sp {url_sp}"
                    try:
                        tor_manager.load_url(
                            url_sp, TorManagerConfig.URL_SP_LOAD_TIMEOUT
                        )
                        logging.info("Loaded sp successfully.")

                        self._scrape_and_save_listings_from_sp(
                            tor_manager, exception_msg_sp
                        )
                        logging.info(f"Scraping from sp {url_sp} completed.")
                        logging.info(
                            f"Cars on sp scraped: {self.cars_scraped_per_sp_no}."
                        )
                    except IPAddressBlockedException as e:
                        logging.warning(exception_msg_sp(e))
                        tor_manager.get_new_tor_circuit()
                    except (ConnectionError, WebDriverException, DatabaseError) as e:
                        finished_flag = True
                        logging.warning(exception_msg_sp(e))
                    except Exception as e:
                        log_detailed_error(e, exception_msg_sp(e))
        except (OSError, DatabaseError, Exception) as e:
            log_detailed_error(e, str(e))

        logging.info(f"Process {self.name} has finished.")
