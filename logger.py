import logging
import os
from config import LOG_DIR

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to both console and a log file.
    Usage:  from logger import get_logger
            log = get_logger(__name__)
            log.info("Something happened")
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ── Console handler (INFO and above) ──
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler (DEBUG and above) ──
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(LOG_DIR, 'app.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger