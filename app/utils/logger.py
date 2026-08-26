"""
Structured JSON logging utilities.
"""
import logging
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Custom logging formatter to output logs in JSON format.
    """
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.
        
        Args:
            record (logging.LogRecord): The log record to format.
            
        Returns:
            str: The formatted JSON string.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "correlation_id"):
            log_data["correlation_id"] = record.correlation_id # type: ignore
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter to add a correlation ID to log records if available.
    """
    def __init__(self, correlation_id: str = ""):
        """
        Initialize the filter.
        
        Args:
            correlation_id (str): The initial correlation ID.
        """
        super().__init__()
        self.correlation_id = correlation_id

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add the correlation ID to the record.
        
        Args:
            record (logging.LogRecord): The log record.
            
        Returns:
            bool: Always True to include the record.
        """
        record.correlation_id = self.correlation_id # type: ignore
        return True


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up and return a structured JSON logger.
    
    Args:
        name (str): The name of the logger.
        level (str): The logging level.
        
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times if the logger is already set up
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger
