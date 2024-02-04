import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from os.path import dirname, isfile, join

import tbselenium.common as cm
from selenium.common.exceptions import TimeoutException
from stem import Signal
from stem.control import Controller
from stem.process import launch_tor_with_config
from tbselenium.exceptions import StemLaunchError
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import prepend_to_env_var

from src.config import PROJECT_DIR
from src.exception import IPAddressBlockedException
from src.logger import logging
from src.scraping.web_scraper import create_soup


class TorManagerConfig:
    TBB_PATH = os.path.join(PROJECT_DIR, "tor-browser")
    TOR_CONTROL_PORT = 9251
    TOR_CIRCUIT_WAIT_TIME = 10
    URL_LISTING_LOAD_TIMEOUT = 9
    URL_SP_LOAD_TIMEOUT = 15
    HEADLESS_MODE = True
    TIMEOUT = 120


def launch_tbb_tor_with_stem_expanded(
    tbb_path=None,
    torrc=None,
    tor_binary=None,
    timeout: int = 90,
):
    """Based on tbselenium.utils.launch_tbb_tor_with_stem, expanded with additional timeout parameter."""
    if not (tor_binary or tbb_path):
        raise StemLaunchError("Either pass tbb_path or tor_binary")

    if not tor_binary and tbb_path:
        tor_binary = join(tbb_path, cm.DEFAULT_TOR_BINARY_PATH)

    if not isfile(tor_binary):
        raise StemLaunchError("Invalid Tor binary")

    prepend_to_env_var("LD_LIBRARY_PATH", dirname(tor_binary))
    if torrc is None:
        torrc = {
            "ControlPort": str(cm.STEM_CONTROL_PORT),
            "SOCKSPort": str(cm.STEM_SOCKS_PORT),
            "DataDirectory": tempfile.mkdtemp(),
        }

    return launch_tor_with_config(config=torrc, tor_cmd=tor_binary, timeout=timeout)


class TorManager:
    def __init__(self, options, torcc):
        self.tor_process: subprocess.Popen
        self.driver: TorBrowserDriver
        self.controller: Controller
        self.options = options
        self.torcc = torcc

    @contextmanager
    def manage(self):
        self.tor_process = launch_tbb_tor_with_stem_expanded(
            TorManagerConfig.TBB_PATH,
            torrc=self.torcc,
            timeout=TorManagerConfig.TIMEOUT,
        )

        try:
            with TorBrowserDriver(
                TorManagerConfig.TBB_PATH,
                tor_cfg=cm.USE_STEM,
                headless=TorManagerConfig.HEADLESS_MODE,
                options=self.options,
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
