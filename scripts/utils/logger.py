from datetime import datetime
import logging
from scripts.config.constants import LOGS_DIR, LOG_FORMAT, DATE_FORMAT, LOG_LEVEL


def setup_logger(name, rq, command):
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    log_dir = LOGS_DIR / rq
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{command}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
