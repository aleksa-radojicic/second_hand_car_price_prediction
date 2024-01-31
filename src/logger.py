import logging
import os
from datetime import datetime

LOG_DATETIME_FORMAT = "%Y_%m_%d_%H_%M_%S"
TODAY_DATE = datetime.now().strftime(LOG_DATETIME_FORMAT)

LOG_FILE = f"{TODAY_DATE}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
LOG_ENDING = 5 * "*"

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)