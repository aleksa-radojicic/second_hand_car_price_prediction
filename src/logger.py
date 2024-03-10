from enum import IntEnum
import logging
import os
import sys
from datetime import datetime


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.exception import SeverityException

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


class SeverityMode(IntEnum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def log_detailed_error(e: Exception, message):
    _, e_object, e_traceback = sys.exc_info()

    e_filename = os.path.split(e_traceback.tb_frame.f_code.co_filename)[1]
    e_line_number = e_traceback.tb_lineno

    logging.error(message)
    logging.error(f"File: {e_filename}, Line: {e_line_number}")


def log_by_severity(e: "SeverityException", message):
    severity = e.severity
    log_function_name = logging.getLevelName(severity).lower()
    log_function = getattr(logging, log_function_name, logging.error)
    log_function(message)


def get_function_to_log(severity: SeverityMode):
    log_function_name = logging.getLevelName(severity).lower()
    function_to_log = getattr(logging, log_function_name, logging.error)
    return function_to_log


def log_message(message, verbose=1):
    if verbose > 0:
        logging.info(message)
