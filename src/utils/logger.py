"""
Centralized logging configuration for the Spark-Kafka ML Training Pipeline.

Provides structured logging with support for console and file handlers,
log rotation, and integration with Spark's internal logging system.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LOG_DIR = Path("logs")
_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
_BACKUP_COUNT = 5


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = False,
    log_dir: Optional[Path] = None,
    rotating: bool = True,
) -> logging.Logger:
    """
    Create or retrieve a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Logging level (default: INFO).
        log_to_file: Whether to add a file handler.
        log_dir: Directory for log files (default: ./logs).
        rotating: Use rotating file handler if True, timed rotation otherwise.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        _log_dir = log_dir or _DEFAULT_LOG_DIR
        _log_dir.mkdir(parents=True, exist_ok=True)
        log_file = _log_dir / f"{name.replace('.', '_')}.log"

        if rotating:
            file_handler = RotatingFileHandler(
                filename=str(log_file),
                maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )
        else:
            file_handler = TimedRotatingFileHandler(
                filename=str(log_file),
                when="midnight",
                interval=1,
                backupCount=_BACKUP_COUNT,
                encoding="utf-8",
            )

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_spark_logging(level: str = "WARN") -> None:
    """
    Suppress verbose Spark/Hadoop logging to keep pipeline logs readable.

    Args:
        level: Java log level string (ERROR, WARN, INFO, DEBUG).
    """
    spark_loggers = [
        "org.apache.spark",
        "org.apache.hadoop",
        "org.apache.kafka",
        "io.delta",
    ]
    log4j_level_map = {
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    py_level = log4j_level_map.get(level.upper(), logging.WARNING)

    for spark_logger_name in spark_loggers:
        spark_logger = logging.getLogger(spark_logger_name)
        spark_logger.setLevel(py_level)


class PipelineLogContext:
    """Context manager for structured pipeline step logging."""

    def __init__(self, logger: logging.Logger, step_name: str, **kwargs):
        self.logger = logger
        self.step_name = step_name
        self.metadata = kwargs

    def __enter__(self):
        meta_str = " | ".join(f"{k}={v}" for k, v in self.metadata.items())
        self.logger.info(f"[START] {self.step_name} | {meta_str}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"[FAILED] {self.step_name} | error={exc_type.__name__}: {exc_val}"
            )
            return False
        self.logger.info(f"[COMPLETED] {self.step_name}")
        return False
