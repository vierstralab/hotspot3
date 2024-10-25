import logging
import sys


def setup_logger(name=None, level=None, outstream=None) -> logging.Logger:
    if name is None:
        name = __name__
    
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


