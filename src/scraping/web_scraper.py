import os

import tbselenium.common as cm
from bs4 import BeautifulSoup
from tbselenium.tbdriver import TorBrowserDriver
from tbselenium.utils import launch_tbb_tor_with_stem

PROJECT_DIR = os.getcwd()
TBB_PATH = os.path.join(PROJECT_DIR, "tor-browser")

url = "https://www.polovniautomobili.com/auto-oglasi/23092655/nissan-qashqai-15-dci-n-connecta?attp=p19_pv0_pc1_pl10_plv0"

tor_process = launch_tbb_tor_with_stem(TBB_PATH)

with TorBrowserDriver(TBB_PATH, tor_cfg=cm.USE_STEM, headless=True) as driver:
    driver.load_url(url)
    soup = BeautifulSoup(driver.page_source, "lxml")

    img_elements = soup.find_all("img")

    for img in img_elements:
        img_src = img.get("src")
        img_alt = img.get("alt")

        print(f"Image Source: {img_src}")
        print(f"Alt Text: {img_alt}")
        print("-----")
tor_process.kill()
