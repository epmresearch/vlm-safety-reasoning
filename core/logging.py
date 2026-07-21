"""
Shared logger. Every module does:
    from core.logging import get_logger
    logger = get_logger(__name__)
"""
import sys
from loguru import logger as _loguru_logger

_configured = False


def _configure_once():
    global _configured
    if _configured:
        return
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    _configured = True


def get_logger(name: str = "vlm-project1"):
    _configure_once()
    return _loguru_logger.bind(name=name)

_file_handler_id = None

def attach_file_logger(filepath: str):
    """Dynamically attaches a file sink to save logs for a specific run."""
    global _file_handler_id
    if _file_handler_id is not None:
        return  # Already attached
    
    _file_handler_id = _loguru_logger.add(
        filepath,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        enqueue=True,
    )