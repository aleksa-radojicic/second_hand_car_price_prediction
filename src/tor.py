import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from os.path import dirname, isfile, join
from typing import Any, Literal

import tbselenium.common as cm
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from stem import Signal
from stem.control import Controller
from stem.process import launch_tor_with_config
from tbselenium.exceptions import StemLaunchError
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import prepend_to_env_var

from src.exception import IPAddressBlockedException
from src.logger import logging
from src.scraping.web_scraper import create_soup


@dataclass
class TorConfig:
    tbb_path: str
    circuit_wait_time: int
    url_listing_load_timeout: int
    url_sp_load_timeout: int
    headless_mode: bool
    start_timeout: int


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


class Tor:
    tor_process: subprocess.Popen
    driver: TorBrowserDriver
    controller: Controller

    cfg: TorConfig
    options: FirefoxOptions
    torcc: dict[str, Any]

    def __init__(
        self, cfg: TorConfig, options: FirefoxOptions, torcc: dict[str, Any]
    ):
        self.cfg = cfg
        self.options = options
        self.torcc = torcc

    @contextmanager
    def manage(self):
        self.tor_process = launch_tbb_tor_with_stem_expanded(
            self.cfg.tbb_path,
            torrc=self.torcc,
            timeout=self.cfg.start_timeout,
        )

        try:
            with TorBrowserDriver(
                self.cfg.tbb_path,
                tor_cfg=cm.USE_STEM,
                headless=self.cfg.headless_mode,
                options=self.options,
            ) as self.driver:
                with Controller.from_port(port=int(self.torcc["ControlPort"])) as self.controller:  # type: ignore
                    self.controller.authenticate()
                    logging.info(f"Tor is running with PID={self.controller.get_pid()}")
                    yield self
        finally:
            self.tor_process.kill()

    def get_new_tor_circuit(self):
        self.controller.signal(Signal.NEWNYM)  # type: ignore
        logging.info("Got new tor circuit")
        time.sleep(self.cfg.circuit_wait_time)

    def check_for_blocked_ip(self):
        soup = create_soup(self.driver.page_source)
        img_elements = soup.find_all("img")

        if len(img_elements) < 4:
            raise IPAddressBlockedException("IP address is blocked")

    def load_url(self, url: str, type: Literal["listing", "sp"]):
        try:
            timeout: int

            if type == "listing":
                timeout = self.cfg.url_listing_load_timeout
            elif type == "sp":
                timeout = self.cfg.url_sp_load_timeout

            self.driver.set_page_load_timeout(timeout)
            self.driver.load_url(url)
        except TimeoutException:
            pass
        self.check_for_blocked_ip()
