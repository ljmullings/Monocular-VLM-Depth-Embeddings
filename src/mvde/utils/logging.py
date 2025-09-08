"""Logging utilities for MVDE."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = (
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Module-level logger
logger = get_logger(__name__)
