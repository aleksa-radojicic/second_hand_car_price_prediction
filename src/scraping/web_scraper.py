import os
import time
from typing import List, Tuple

import tbselenium.common as cm
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from stem import Signal
from stem.control import Controller
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import launch_tbb_tor_with_stem

from src.db.broker import DbBroker
from src.domain.domain import (AdditionalInformation, EquipmentInformation,
                               GeneralInformation, Listing, OtherInformation,
                               SafetyInformation)
from src.exception import (IPAddressBlockedException, LabelNotGivenException,
                           ScrapingException)
from src.logger import log_by_severity, log_detailed_error, logging


def get_new_tor_circuit(controller):
    controller.signal(Signal.NEWNYM)  # type: ignore
    time.sleep(5)


def check_for_blocked_ip(soup: BeautifulSoup):
    img_elements = soup.find_all("img")

    if len(img_elements) == 0:
        raise IPAddressBlockedException("IP address blocked")


def scrape_listing(soup: BeautifulSoup, car_id) -> Listing:
    try:
        listing = Listing(id=car_id)

        listing.name = soup.find("h1").contents[0].get_text(strip=True)  # type: ignore
        listing.price = soup.find("span", "priceClassified").get_text(strip=True)  # type: ignore

        if listing.price == "Po dogovoru":
            raise LabelNotGivenException("Price is not set")

        listing.listing_followers_no = soup.find("span", "classified-liked prati-oglas-like").get_text(strip=True)  # type: ignore

        if soup.find("div", class_="address"):
            location = soup.find("div", class_="address").find_parent("div").contents[0].get_text(strip=True)  # type: ignore
        else:
            location = (
                soup.find("div", class_="js-tutorial-contact")
                .findChild("div", class_="uk-width-1-2")
                .get_text(strip=True)
            )

        listing.location = location
        listing.images_no = soup.find("div", class_="js-gallery-numbers image-counter").get_text(strip=True).split("/")[1]  # type: ignore

        listing.general_information = scrape_general_information(soup)
        listing.additional_information = scrape_additional_information(soup)
        listing.equipment_information = scrape_equipment_information(soup)
        listing.safety_information = scrape_safety_information(soup)
        listing.other_information = scrape_other_information(soup)

        # logging.info("Successfully scraped listing")
        return listing
    except LabelNotGivenException as e:
        raise e
    except Exception as e:
        log_detailed_error(e, str(e))
        raise ScrapingException(str(e))


def scrape_general_information(soup: BeautifulSoup):
    general_information = GeneralInformation()

    return general_information


def scrape_additional_information(soup: BeautifulSoup):
    additional_information = AdditionalInformation()

    return additional_information


def scrape_equipment_information(soup: BeautifulSoup):
    equipment_information = EquipmentInformation()

    return equipment_information


def scrape_safety_information(soup: BeautifulSoup):
    safety_information = SafetyInformation()

    return safety_information


def scrape_other_information(soup: BeautifulSoup):
    other_information = OtherInformation()

    return other_information


def get_listing_urls_and_ids_from_page(
    search_page_soup: BeautifulSoup,
) -> Tuple[List[str], List[str]]:
    anchors = search_page_soup.find_all("a", class_="ga-title")
    listing_urls = [f"{index_page_url}{anchor.get('href')}" for anchor in anchors]
    car_ids = [anchor.get("href").split("/")[2] for anchor in anchors]
    return listing_urls, car_ids


def create_soup(page_source: str):
    return BeautifulSoup(page_source, "lxml")


def check_if_site_fully_loaded(url, page_source: str) -> BeautifulSoup:
    count_not_fully_loaded = 0

    while True:
        soup = create_soup(page_source)
        if count_not_fully_loaded == 10:
            logging.warning("Couldn't find gallery image counter for url " + url)
            raise ValueError()

        if not soup.find("div", class_="js-gallery-numbers image-counter"):
            time.sleep(5)
            count_not_fully_loaded += 1
        else:
            if count_not_fully_loaded != 0:
                logging.info("Found gallery image counter for url " + url)
            return soup


PROJECT_DIR = os.getcwd()
TBB_PATH = os.path.join(PROJECT_DIR, "tor-browser")
PORT = 9251
index_page_url = "https://www.polovniautomobili.com"

if __name__ == "__main__":
    tor_process = launch_tbb_tor_with_stem(TBB_PATH)

    with TorBrowserDriver(TBB_PATH, tor_cfg=cm.USE_STEM, headless=True) as driver:
        with Controller.from_port(port=PORT) as controller:  # type: ignore
            controller.authenticate()

            all_pages_scraped_flag = False
            page_no = 1
            base_url_search_page = (
                lambda x: f"{index_page_url}/auto-oglasi/pretraga?page={x}&sort=renewDate_asc&city_distance=0&showOldNew=all&without_price=1"
            )
            logging.info("Authenticated")
            url_search_page = ""

            two_time_failed_listing = 0
            two_time_failed_page = 0

            no_cars_scraped_total = 0

            while not all_pages_scraped_flag:
                try:
                    url_search_page = base_url_search_page(page_no)
                    all_listings_in_page_scraped_flag = False

                    driver.set_page_load_timeout(60)
                    driver.load_url(url_search_page)
                    logging.info("Loaded search page successfully.")
                    search_page_soup = create_soup(driver.page_source)

                    check_for_blocked_ip(search_page_soup)

                    no_cars_scraped_per_search_page = 0
                    listing_urls, car_ids = get_listing_urls_and_ids_from_page(
                        search_page_soup
                    )

                    for listing_url, car_id in zip(listing_urls, car_ids):
                        try:
                            driver.set_page_load_timeout(60)
                            driver.load_url(listing_url)
                            listing_page_soup = check_if_site_fully_loaded(
                                listing_url, driver.page_source
                            )
                            check_for_blocked_ip(listing_page_soup)

                            listing = scrape_listing(listing_page_soup, car_id)
                            DbBroker().save_listing(listing)
                            logging.info(str(listing))
                            page_no += 1
                            no_cars_scraped_per_search_page += 1
                        except (ScrapingException, LabelNotGivenException) as e:
                            log_by_severity(e, str(e) + " for url " + listing_url)
                        except IPAddressBlockedException as e:
                            logging.warning(str(e) + " for url " + listing_url)
                            get_new_tor_circuit(controller)
                        except TimeoutException as e:
                            logging.warning(
                                "Message: Navigation timed out for url " + listing_url
                            )

                            two_time_failed_listing += 1

                            if two_time_failed_listing == 2:
                                get_new_tor_circuit(controller)
                                two_time_failed_listing = 0
                        except ValueError as e:
                            logging.error(str(e) + " for url " + listing_url)
                        except Exception as e:
                            logging.error(str(e) + " for url " + listing_url)
                            break
                    no_cars_scraped_total += no_cars_scraped_per_search_page
                    logging.info(
                        f"Scraping from search page {url_search_page} completed, cars scraped: {no_cars_scraped_per_search_page}."
                    )
                    logging.info(f"Total cars scraped: {no_cars_scraped_total}")
                except ValueError as e:
                    all_pages_scraped_flag = True
                    logging.error(str(e) + " for search page " + url_search_page)
                except IPAddressBlockedException as e:
                    logging.warning(str(e) + " for search page " + url_search_page)
                    get_new_tor_circuit(controller)
                except TimeoutException as e:
                    logging.warning(
                        "Message: Navigation timed out for search page "
                        + url_search_page
                    )

                    two_time_failed_page += 1

                    if two_time_failed_page == 2:
                        get_new_tor_circuit(controller)
                        two_time_failed_page = 0
                except Exception as e:
                    log_detailed_error(e, str(e))
                    raise ScrapingException(str(e))

    tor_process.kill()
