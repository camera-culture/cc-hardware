import logging
import logging.config


class FileHandler(logging.FileHandler):
    """A file handler which creates the directory if it doesn't exist."""

    def __init__(self, filename, *args, **kwargs):
        # Create the file before calling the super constructor
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        super().__init__(filename, *args, **kwargs)


class TqdmStreamHandler(logging.StreamHandler):
    """A handler that uses tqdm.write to log messages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            from tqdm import tqdm  # noqa
        except ImportError:
            raise ImportError("tqdm is required for TqdmStreamHandler")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            from tqdm import tqdm

            tqdm.write(msg, end=self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


class LoggerMaxLevelFilter(logging.Filter):
    """This filter sets a maximum level."""

    def __init__(self, max_level: int | str):
        if isinstance(max_level, str):
            max_level = getattr(logging, max_level.upper())
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= self._max_level


LOGGING_CONFIG = {
    "version": 1,
    "filters": {
        "max_level": {
            "()": LoggerMaxLevelFilter,
            "max_level": logging.INFO,
        }
    },
    "handlers": {
        "stdout": {
            "class": TqdmStreamHandler,
            "formatter": "simple",
            "stream": "ext://sys.stdout",
            "level": logging.DEBUG,
        },
    },
    "loggers": {
        "cc_hardware": {
            "level": logging.INFO,
            "handlers": ["stdout"],
            # Set propagate to false to avoid double logging
            # If it were true, all logs applied to the logger would continue down to
            # the root.
            "propagate": False,
        },
    },
    "formatters": {
        "simple": {"format": "%(levelname)-8s | %(module)s.%(funcName)s :: %(message)s"}
    },
}


def get_logger(name: str = "cc_hardware", *, overrides: dict = {}) -> logging.Logger:
    LOGGING_CONFIG.update(overrides)
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger(name)
