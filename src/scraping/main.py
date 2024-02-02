import os
import time
from typing import List, Tuple

import tbselenium.common as cm
from selenium.common.exceptions import TimeoutException, WebDriverException
from stem import Signal
from stem.control import Controller
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import launch_tbb_tor_with_stem

from src.config import INDEX_PAGE_URL, PROJECT_DIR
from src.db.broker import DbBroker
from src.exception import (IPAddressBlockedException, LabelNotGivenException,
                           ScrapingException)
from src.logger import log_by_severity, log_detailed_error, logging
from src.scraping.web_scraper import Scraper, create_soup

TBB_PATH = os.path.join(PROJECT_DIR, "tor-browser")
TOR_CONTROL_PORT = 9251
TOR_CIRCUIT_WAIT_TIME = 10
URL_LISTING_LOAD_TIMEOUT = 11
URL_SP_LOAD_TIMEOUT = 15
HEADLESS_MODE = True
BASE_URL_SP = (
    lambda x: f"{INDEX_PAGE_URL}/auto-oglasi/pretraga?page={x}&sort=renewDate_desc&city_distance=0&showOldNew=all&with_images=1"
)

def get_new_tor_circuit(controller):
    controller.signal(Signal.NEWNYM)  # type: ignore
    time.sleep(TOR_CIRCUIT_WAIT_TIME)


def check_for_blocked_ip(driver: TorBrowserDriver):
    soup = create_soup(driver.page_source)
    img_elements = soup.find_all("img")

    if len(img_elements) == 4:
        raise IPAddressBlockedException("IP address is blocked")


def get_listing_urls_and_ids_from_page(
    driver: TorBrowserDriver,
) -> Tuple[List[str], List[str]]:
    soup = create_soup(driver.page_source)
    anchors = soup.find_all("a", class_="ga-title")
    listing_urls = [f"{INDEX_PAGE_URL}{anchor.get('href')}" for anchor in anchors]
    car_ids = [anchor.get("href").split("/")[2] for anchor in anchors]
    return listing_urls, car_ids


def load_url(driver: TorBrowserDriver, url: str, timeout: float):
    try:
        driver.set_page_load_timeout(timeout)
        driver.load_url(url)
        check_for_blocked_ip(driver)
    except TimeoutException:
        pass


def main():
    finished_flag = False
    page_no = 1
    url_sp = ""
    cars_scraped_total_no = 0
    
    tor_process = launch_tbb_tor_with_stem(TBB_PATH)

    with TorBrowserDriver(
        TBB_PATH, tor_cfg=cm.USE_STEM, headless=HEADLESS_MODE
    ) as driver:
        with Controller.from_port(port=TOR_CONTROL_PORT) as controller:  # type: ignore
            controller.authenticate()

            while not finished_flag:
                url_sp = BASE_URL_SP(page_no)

                try:
                    load_url(driver, url_sp, URL_SP_LOAD_TIMEOUT)
                    logging.info("Loaded sp successfully.")
                    cars_scraped_per_sp_no = 0
                    listing_urls, car_ids = get_listing_urls_and_ids_from_page(driver)

                    for url_listing, car_id in zip(listing_urls, car_ids):
                        try:
                            load_url(driver, url_listing, URL_LISTING_LOAD_TIMEOUT)
                            listing = Scraper(driver, car_id).scrape_listing()
                            DbBroker().save_listing(listing)
                            cars_scraped_per_sp_no += 1
                        except (ScrapingException, LabelNotGivenException) as e:
                            log_by_severity(e, str(e) + " for url " + url_listing)
                        except IPAddressBlockedException as e:
                            logging.warning(str(e) + " for url " + url_listing)
                            get_new_tor_circuit(controller)
                        except Exception as e:
                            logging.error(str(e) + " for url " + url_listing)
                            break
                    cars_scraped_total_no += cars_scraped_per_sp_no
                    page_no += 1
                    logging.info(f"Scraping from sp {url_sp} completed.")
                    logging.info(f"Cars on sp scraped: {cars_scraped_per_sp_no}.")
                    logging.info(f"Total cars scraped: {cars_scraped_total_no}")
                except IPAddressBlockedException as e:
                    logging.warning(str(e) + " for sp " + url_sp)
                    get_new_tor_circuit(controller)
                except (ConnectionError, WebDriverException) as e:
                    finished_flag = True
                    logging.warning(str(e) + " for sp " + url_sp)
                except Exception as e:
                    log_detailed_error(e, str(e))

    tor_process.kill()


if __name__ == "__main__":
    main()
