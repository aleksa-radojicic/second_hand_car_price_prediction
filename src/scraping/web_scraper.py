from bs4 import BeautifulSoup
from tbselenium.tbdriver import TorBrowserDriver

from src.domain.domain import (SECTION_NAMES_SRB_MAP, AdditionalInformation,
                               EquipmentInformation, GeneralInformation,
                               Listing, OtherInformation, SafetyInformation)
from src.exception import LabelNotGivenException, ScrapingException
from src.logger import log_detailed_error, logging


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


def create_soup(page_source: str):
    return BeautifulSoup(page_source, "lxml")
