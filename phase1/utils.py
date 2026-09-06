import os
import logging

def setup_logger(name: str = "SIH_Phase1") -> logging.Logger:
    """
    Configures and returns a structured logger.
    Only logs essential pipeline events, avoiding screen clutter.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter format: [INFO] 12:34:56 - Message
        formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s', datefmt='%H:%M:%S')
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
    return logger

def ensure_dir(dir_path: str):
    """
    Ensures that a directory exists by creating it if it doesn't.
    """
    if dir_path:
        os.makedirs(os.path.abspath(dir_path), exist_ok=True)
