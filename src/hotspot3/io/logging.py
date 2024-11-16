import logging
import sys
from typing import Type, TypeVar

from genome_tools import GenomicInterval

from hotspot3.config import ProcessorConfig


T = TypeVar('T', bound='WithLogger')

class WithLogger:
    def __init__(self, config=None, logger=None, name=None):
        if config is None:
            config = ProcessorConfig()
        self.config = config

        self._logger = logger if logger is not None else setup_logger(level=self.config.logger_level)

        if name is None:
            name = self.__class__.__name__
        self.name = name
    
    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @logger.setter
    def logger(self, logger: logging.Logger):
        # Setter allows updating the logger
        self._logger = logger
    
    def copy_with_params(self, cls: Type[T]) -> T:
        """
        Creates a new instance of the specified class `cls` with the same initialization parameters.
        """
        assert issubclass(cls, WithLogger), "cls should be a subclass of WithLogger"
        return cls(
            name=self.name,
            config=self.config,
            logger=self.logger
        )


class WithLoggerAndInterval(WithLogger):
    def __init__(self, genomic_interval: GenomicInterval, config: ProcessorConfig=None, logger=None):
        super().__init__(logger=logger, config=config, name=genomic_interval.to_ucsc())
        self.genomic_interval = genomic_interval
    
    def copy_with_params(self, cls: Type[T], **kwargs) -> T:
        """
        Creates a new instance of the specified class `cls` with the same initialization parameters.
        """
        if issubclass(cls, WithLoggerAndInterval):
            class_fields = dict(
                genomic_interval=self.genomic_interval,
                config=self.config,
                logger=self.logger
            )
        elif issubclass(cls, WithLogger):
            class_fields = dict(
                config=self.config,
                logger=self.logger,
                name=self.genomic_interval.to_ucsc()
            )
        else:
            raise ValueError(f"cls should be a subclass of WithLogger or WithLoggerAndInterval, got {cls}")
        class_fields.update(kwargs)
        return cls(**class_fields)


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
