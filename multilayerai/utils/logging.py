import logging
import sys
from typing import Union, TextIO


def create_logger(level: Union[str, int] = logging.INFO, stream: TextIO = sys.stderr) -> logging.Logger:
    logger = logging.getLogger()
    logging.basicConfig(stream=stream)
    logger.setLevel(level)
    return logger
