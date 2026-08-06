"""
Shared logging base for the package.

Every message the package emits goes through the standard :mod:`logging`
module, under one logger per processor class beneath the ``neurofunctionx``
package logger. A single :meth:`BaseProcessor.configure_logging` call -- or any
``logging`` setup of your own, e.g. ``logging.basicConfig`` or a file handler in
a batch job -- therefore controls all of it, including level, format and where
the output goes.

Subclasses keep using the familiar instance helpers::

    self._log("...")        self._log_warn("...")
    self._log_error("...")  self._log_verbose("...")   # debug level

Module-level code that has no class to hang the call off uses the same helpers
as classmethods::

    BaseProcessor.log("...")        BaseProcessor.log_warn("...")
"""
import logging
import sys
from abc import ABC
from datetime import datetime
from typing import Optional, TextIO

PACKAGE_LOGGER_NAME = "neurofunctionx"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


class BaseProcessor(ABC):
    """
    Base logging functionality
    """
    time_log = None
    verbose = False

    # ----------------------------------------------------------------- #
    # Configuration
    # ----------------------------------------------------------------- #

    @classmethod
    def configure_logging(
            cls,
            level: int = logging.INFO,
            stream: Optional[TextIO] = None,
            fmt: str = LOG_FORMAT,
            datefmt: str = LOG_DATE_FORMAT,
    ) -> logging.Logger:
        """
        Attach a stream handler to the package logger and return it.

        Call this once at the start of a script or batch job. Pass
        ``level=logging.DEBUG`` to also get the ``_log_verbose`` messages.
        Replaces any handler a previous call installed, so it is safe to call
        again to change level or destination.
        """
        logger = logging.getLogger(PACKAGE_LOGGER_NAME)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        handler = logging.StreamHandler(stream or sys.stdout)
        handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(handler)
        logger.setLevel(level)
        # Ours is the only handler that should print these; without this the
        # root logger would emit a duplicate line once the application calls
        # basicConfig().
        logger.propagate = False
        return logger

    @classmethod
    def _logger(cls) -> logging.Logger:
        """Logger for this class: ``neurofunctionx.<ClassName>``."""
        package = logging.getLogger(PACKAGE_LOGGER_NAME)
        if not package.handlers and not logging.getLogger().handlers:
            # These helpers used to print unconditionally. Without a fallback
            # handler every info message would silently vanish in scripts that
            # never configure logging, which is a nasty surprise mid-run.
            cls.configure_logging()
        if cls is BaseProcessor:
            return package
        return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{cls.__name__}")

    # ----------------------------------------------------------------- #
    # Logging helpers
    # ----------------------------------------------------------------- #

    @classmethod
    def log(cls, message, *args):
        cls._logger().info(message, *args)

    @classmethod
    def log_verbose(cls, message, *args):
        cls._logger().debug(message, *args)

    @classmethod
    def log_warn(cls, message, *args):
        cls._logger().warning(message, *args)

    @classmethod
    def log_error(cls, message, *args):
        cls._logger().error(message, *args)

    # Instance-style aliases, unchanged for existing subclasses.
    _log = log
    _log_verbose = log_verbose
    _log_warn = log_warn
    _log_error = log_error

    def _log_time(self, silent=False):
        now = datetime.now()
        if self.time_log is None:
            if not silent:
                self._log("Time log started")
            self.time_log = now
            return None
        else:
            delta = now - self.time_log
            minutes = delta.total_seconds() / 60
            self.time_log = None
            if not silent:
                self._log(f"Time log stopped. Time Delta: {minutes:.2f} minutes")
            return minutes