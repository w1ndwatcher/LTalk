# backend/logging_config.py

import logging
import sys
from pythonjsonlogger import jsonlogger

from config import config


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level"},
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.DEBUG if not config.is_production else logging.INFO)

    # Quiet down noisy third-party loggers so they don't drown out app logs.
    for noisy in ("azure", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)