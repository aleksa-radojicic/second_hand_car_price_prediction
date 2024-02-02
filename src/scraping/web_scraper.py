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
from src.domain.domain import (SECTION_NAMES_SRB_MAP, AdditionalInformation,
                               EquipmentInformation, GeneralInformation,
                               Listing, OtherInformation, SafetyInformation)
from src.exception import (IPAddressBlockedException, LabelNotGivenException,
                           ScrapingException)
from src.logger import log_by_severity, log_detailed_error, logging


def get_new_tor_circuit(controller):
    controller.signal(Signal.NEWNYM)  # type: ignore
    time.sleep(5)


def check_for_blocked_ip(driver: TorBrowserDriver):
    soup = create_soup(driver.page_source)
    img_elements = soup.find_all("img")

    if len(img_elements) == 0:
        raise IPAddressBlockedException("IP address blocked")


class Scraper:
    def __init__(self, driver: TorBrowserDriver, car_id):
        self.soup = create_soup(driver.page_source)
        self.car_id = car_id

    def scrape_listing(self) -> Listing:
        try:
            listing = Listing(id=self.car_id)

            listing.name = self.soup.find("h1").contents[0].get_text(strip=True)  # type: ignore
            listing.price = self.soup.find("span", "priceClassified").get_text(strip=True)  # type: ignore

            if listing.price == "Po dogovoru":
                raise LabelNotGivenException("Price is not set")

            listing.listing_followers_no = self.soup.find("span", "classified-liked prati-oglas-like").get_text(strip=True)  # type: ignore

            if self.soup.find("div", class_="address"):
                location = self.soup.find("div", class_="address").find_parent("div").contents[0].get_text(strip=True)  # type: ignore
            else:
                location = (
                    self.soup.find("div", class_="js-tutorial-contact")
                    .findChild("div", class_="uk-width-1-2")
                    .get_text(strip=True)
                )

            listing.location = location
            listing.images_no = self.soup.find("div", class_="js-gallery-numbers image-counter").get_text(strip=True).split("/")[1]  # type: ignore

            listing.general_information = self._scrape_gi_and_ai("GeneralInformation")
            listing.additional_information = self._scrape_gi_and_ai(
                "AdditionalInformation"
            )
            listing.equipment_information = self._scrape_equipment_information()
            listing.safety_information = self._scrape_safety_information()
            listing.other_information = self._scrape_other_information()

            # logging.info("Successfully scraped listing")
            return listing
        except LabelNotGivenException as e:
            raise e
        except Exception as e:
            log_detailed_error(e, str(e))
            raise ScrapingException(str(e))

    def _scrape_gi_and_ai(self, class_name: str):
        domain_instance = None
        if class_name == "GeneralInformation":
            domain_instance = GeneralInformation()
        elif class_name == "AdditionalInformation":
            domain_instance = AdditionalInformation()

        h2s = self.soup.find_all("h2", class_="classified-title")
        main_h2 = next(
            (
                h2
                for h2 in h2s
                if h2.get_text(strip=True) == SECTION_NAMES_SRB_MAP[class_name]
            ),
            None,
        )
        main_div = main_h2.find_next_sibling("div")
        divider_els = main_div.find_all("div", class_="divider")
        property_value_pairs = [
            el.find_all(class_="uk-width-1-2") for el in divider_els
        ]

        for property_in_srb, value in property_value_pairs:
            try:
                property_in_srb_txt = property_in_srb.get_text(strip=True).strip(":")

                if property_in_srb_txt in [
                    "Broj oglasa",
                    "Broj šasije",
                    "Datum postavke",
                    "Datum obnove",
                    "U ponudi od",
                ]:
                    continue

                value_txt = value.get_text(strip=True)
                attribute_name = domain_instance.SRB_NAMES_TO_ATTRS_MAP[
                    property_in_srb_txt
                ]

                # if class_name == "AdditionalInformation":
                #     logging.info(
                #         f"{property_in_srb_txt} -> {attribute_name} -> {value_txt}"
                #     )

                domain_instance.__setattr__(attribute_name, value_txt)
            except Exception as e:
                logging.error(f"{str(e)}")

        return domain_instance

    def _scrape_equipment_information(self):
        equipment_information = EquipmentInformation()

        return equipment_information

    def _scrape_safety_information(self):
        safety_information = SafetyInformation()

        return safety_information

    def _scrape_other_information(self):
        other_information = OtherInformation()

        return other_information


def get_listing_urls_and_ids_from_page(
    driver: TorBrowserDriver,
) -> Tuple[List[str], List[str]]:
    soup = create_soup(driver.page_source)
    anchors = soup.find_all("a", class_="ga-title")
    listing_urls = [f"{index_page_url}{anchor.get('href')}" for anchor in anchors]
    car_ids = [anchor.get("href").split("/")[2] for anchor in anchors]
    return listing_urls, car_ids


def create_soup(page_source: str):
    return BeautifulSoup(page_source, "lxml")


def check_if_site_fully_loaded(url, driver: TorBrowserDriver) -> BeautifulSoup:
    count_not_fully_loaded = 0

    while True:
        soup = create_soup(driver.page_source)
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
                    check_for_blocked_ip(driver)
                    logging.info("Loaded search page successfully.")

                    no_cars_scraped_per_search_page = 0
                    listing_urls, car_ids = get_listing_urls_and_ids_from_page(driver)

                    for listing_url, car_id in zip(listing_urls, car_ids):
                        try:
                            driver.set_page_load_timeout(60)
                            driver.load_url(listing_url)
                            check_if_site_fully_loaded(listing_url, driver)
                            check_for_blocked_ip(driver)

                            listing = Scraper(driver, car_id).scrape_listing()
                            DbBroker().save_listing(listing)
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
