"""
Logging configuration for ETL pipeline.
Uses structlog for structured logging with proper formatting.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging() -> structlog.BoundLogger:
    """
    Setup structured logging with file and console output.
    Falls back to console-only logging if config is not available.
    
    Returns:
        structlog.BoundLogger: Configured logger instance
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Try to get log file path from environment, fall back to console-only
    log_file_path = os.getenv('LOG_FILE_PATH', 'logs/etl_pipeline.log')
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Configure standard library logging
    import logging
    
    # Create logs directory if it doesn't exist and we're using file logging
    if log_file_path != 'logs/etl_pipeline.log' or os.path.exists('.env'):
        try:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                format="%(message)s",
                level=getattr(logging, log_level),
                handlers=[
                    logging.FileHandler(log_file_path),
                    logging.StreamHandler(sys.stdout)
                ]
            )
        except Exception:
            # Fall back to console-only logging if file logging fails
            logging.basicConfig(
                format="%(message)s",
                level=getattr(logging, log_level),
                handlers=[logging.StreamHandler(sys.stdout)]
            )
    else:
        # Console-only logging for tests
        logging.basicConfig(
            format="%(message)s",
            level=getattr(logging, log_level),
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    return structlog.get_logger()


def get_logger(name: str = "etl_pipeline") -> structlog.BoundLogger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        structlog.BoundLogger: Logger instance
    """
    return structlog.get_logger(name)


# Global logger instance
logger = setup_logging() 