from typing import List

from bs4 import BeautifulSoup
from tbselenium.tbdriver import TorBrowserDriver

from src.config import INDEX_PAGE_URL
from src.domain.domain import (SECTION_NAMES_SRB_MAP, AdditionalInformation,
                               GeneralInformation, Listing)
from src.exception import LabelNotGivenException, ScrapingException
from src.logger import log_detailed_error, logging


class Scraper:
    def __init__(self, driver: TorBrowserDriver):
        self.soup = create_soup(driver.page_source)
        self.url = driver.current_url

    def scrape_listing(self) -> Listing:
        try:
            listing = Listing()
            listing.id = self.url.split("/")[-2]
            listing.general_information = self._scrape_kv_information(
                GeneralInformation
            )
            listing.additional_information = self._scrape_kv_information(
                AdditionalInformation
            )

            listing.name = self.soup.find(class_="table js-tutorial-all").find("h1").contents[0].get_text(strip=True)  # type: ignore
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

            listing.safety = self._scrape_value_information("SafetyInformation")
            listing.equipment = self._scrape_value_information("EquipmentInformation")
            listing.other = self._scrape_value_information("OtherInformation")
            listing.description = self._scrape_description()

            # logging.info("Successfully scraped listing")
            return listing
        except LabelNotGivenException as e:
            raise e
        except Exception as e:
            log_detailed_error(e, str(e))
            raise ScrapingException(str(e))

    def _scrape_kv_information(self, class_type: type) -> object:
        domain_instance = class_type()
        class_name = type(domain_instance).__name__

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

                domain_instance.__setattr__(attribute_name, value_txt)
            except Exception as e:
                logging.error(f"{str(e)}")

        return domain_instance

    def _scrape_value_information(self, section_name: str) -> str:
        section_name_srb = SECTION_NAMES_SRB_MAP[section_name]

        h2s = self.soup.find_all("h2", class_="classified-title")
        property_values_str = ""

        try:
            main_h2 = next(
                (h2 for h2 in h2s if h2.get_text(strip=True) == section_name_srb),
                None,
            )
            main_div = main_h2.find_next_sibling("div")

            divs_to_iterate = main_div.find_all(
                name="div",
                class_="uk-width-medium-1-4 uk-width-1-2 uk-margin-small-bottom",
            )
            property_values_list = [
                div_el.get_text(strip=True) for div_el in divs_to_iterate
            ]
            property_values_str = ",".join(property_values_list)

        except Exception as e:
            pass

        return property_values_str

    def _scrape_description(self) -> str:
        description = ""

        try:
            description_elements = self.soup.find("div", class_="description-wrapper").contents
            description_texts = [el.get_text(strip=True, separator="") for el in description_elements if el.name != 'br']
            description = "\n".join(description_texts)
        except Exception as e:
            pass
        return description

def create_soup(page_source: str):
    return BeautifulSoup(page_source, "lxml")


def get_listing_urls_from_page(
    driver: TorBrowserDriver,
) -> List[str]:
    soup = create_soup(driver.page_source)
    anchors = soup.find_all("a", class_="ga-title")
    listing_urls = [f"{INDEX_PAGE_URL}{anchor.get('href')}" for anchor in anchors]
    return listing_urls
