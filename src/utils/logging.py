"""Central logging setup.

Per PROJECT_SPEC.md: log the model/provider used for every response, and
never log secrets (API keys, raw patient identifiers).
"""
import logging
import sys

from src.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
