import logging
import sys

from genome_tools.genomic_interval import GenomicInterval

from hotspot3.config import ProcessorConfig


class WithLogger:
    def __init__(self, config=None, logger=None, name=None):
        if logger is None:
            logger = setup_logger()
        self.logger = logger

        if config is None:
            config = ProcessorConfig()
        self.config = config

        if name is None:
            name = self.__class__.__name__
        self.name = name


class WithLoggerAndInterval(WithLogger): # TODO: Add method to quickly inherit for a subclass
    def __init__(self, genomic_interval: GenomicInterval, config: ProcessorConfig=None, logger=None):
        super().__init__(logger=logger, config=config, name=genomic_interval.to_ucsc())
        self.genomic_interval = genomic_interval


def setup_logger(name='hotspot3', level=None, outstream=None) -> logging.Logger:
    if outstream is None:
        outstream = sys.stdout

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
