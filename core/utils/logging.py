"""
Logging Setup - Unified logging configuration
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "chess_ai",
    level: int = logging.INFO,
    log_file: str = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with consistent formatting
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "chess_ai") -> logging.Logger:
    """Get existing logger or create default one"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


class TrainingLogger:
    """
    Training-specific logger with metrics tracking
    
    Usage:
        logger = TrainingLogger('sft_training', log_dir='logs')
        logger.log_epoch(epoch=1, loss=2.5, accuracy=0.85)
    """
    
    def __init__(self, name: str, log_dir: str = 'logs'):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f"{name}_{timestamp}.log"
        
        self.logger = setup_logger(
            name=name,
            level=logging.INFO,
            log_file=str(log_file),
            console=True
        )
        
        self.logger.info(f"Training logger initialized: {name}")
        self.logger.info(f"Log file: {log_file}")
    
    def log_epoch(self, epoch: int, **metrics):
        """Log epoch metrics"""
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metric_str}")
    
    def log_iteration(self, iteration: int, **metrics):
        """Log iteration metrics"""
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in metrics.items()])
        self.logger.info(f"Iteration {iteration}: {metric_str}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
