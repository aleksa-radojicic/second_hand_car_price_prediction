import os
import subprocess
import time
from contextlib import contextmanager

import tbselenium.common as cm
from selenium.common.exceptions import TimeoutException
from stem import Signal
from stem.control import Controller
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import launch_tbb_tor_with_stem

from src.config import PROJECT_DIR
from src.exception import IPAddressBlockedException
from src.logger import logging
from src.scraping.web_scraper import create_soup


class TorManagerConfig:
    TBB_PATH = os.path.join(PROJECT_DIR, "tor-browser")
    TOR_CONTROL_PORT = 9251
    TOR_CIRCUIT_WAIT_TIME = 10
    URL_LISTING_LOAD_TIMEOUT = 11
    URL_SP_LOAD_TIMEOUT = 15
    HEADLESS_MODE = True


class TorManager:
    def __init__(self):
        self.tor_process: subprocess.Popen
        self.driver: TorBrowserDriver
        self.controller: Controller

    @contextmanager
    def manage_tor_browser(self):
        self.tor_process = launch_tbb_tor_with_stem(TorManagerConfig.TBB_PATH)

        try:
            with TorBrowserDriver(
                TorManagerConfig.TBB_PATH,
                tor_cfg=cm.USE_STEM,
                headless=TorManagerConfig.HEADLESS_MODE,
            ) as self.driver:
                with Controller.from_port(port=TorManagerConfig.TOR_CONTROL_PORT) as self.controller:  # type: ignore
                    self.controller.authenticate()
                    logging.info(f"Tor is running with PID={self.controller.get_pid()}")
                    yield self
        finally:
            self.tor_process.kill()

    def get_new_tor_circuit(self):
        self.controller.signal(Signal.NEWNYM)  # type: ignore
        logging.info("Got new tor circuit")
        time.sleep(TorManagerConfig.TOR_CIRCUIT_WAIT_TIME)

    def check_for_blocked_ip(self):
        soup = create_soup(self.driver.page_source)
        img_elements = soup.find_all("img")

        if len(img_elements) < 4:
            raise IPAddressBlockedException("IP address is blocked")

    def load_url(self, url: str, timeout: float):
        try:
            self.driver.set_page_load_timeout(timeout)
            self.driver.load_url(url)
        except TimeoutException:
            pass
        self.check_for_blocked_ip()