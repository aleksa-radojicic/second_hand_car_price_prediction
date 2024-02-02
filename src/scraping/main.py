from selenium.common.exceptions import WebDriverException

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


def main():
    finished_flag = False
    page_no = 1
    url_sp = ""
    cars_scraped_total_no = 0

    with TorManager().manage_tor_browser() as tor_manager:
        while not finished_flag:
            url_sp = BASE_URL_SP(page_no)
            exception_msg_sp = lambda e: f"{str(e)} for sp {url_sp}"
            try:
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
                    except (ScrapingException, LabelNotGivenException) as e:
                        log_by_severity(e, exception_msg_listing(e))
                    except IPAddressBlockedException as e:
                        logging.warning(exception_msg_listing(e))
                        tor_manager.get_new_tor_circuit()
                    except Exception as e:
                        logging.error(exception_msg_listing(e))
                        break
                cars_scraped_total_no += cars_scraped_per_sp_no
                page_no += 1
                logging.info(f"Scraping from sp {url_sp} completed.")
                logging.info(f"Cars on sp scraped: {cars_scraped_per_sp_no}.")
                logging.info(f"Total cars scraped: {cars_scraped_total_no}")
            except IPAddressBlockedException as e:
                logging.warning(exception_msg_sp(e))
                tor_manager.get_new_tor_circuit()
            except (ConnectionError, WebDriverException) as e:
                finished_flag = True
                logging.warning(exception_msg_sp(e))
            except Exception as e:
                log_detailed_error(e, exception_msg_sp(e))


if __name__ == "__main__":
    main()
