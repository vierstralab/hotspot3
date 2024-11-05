import logging
import sys
from hotspot3.models import ProcessorConfig


def setup_logger(name='hotspot3', level=None, outstream=None) -> logging.Logger:
    if outstream is None:
        outstream = sys.stderr

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler(outstream)
            handler.setLevel(level)
            formatter = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


class WithLogger:
    def __init__(self, logger=None, config=None, name=None):
        if logger is None:
            logger = setup_logger()
        self.logger = logger

        if config is None:
            config = ProcessorConfig()
        self.config = config

        if name is None:
            name = self.__class__.__name__
        self.name = name

