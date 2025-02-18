import os
import logging
from varseek.utils import set_up_logger

logger = logging.getLogger(__name__)

def another_func():
    logger.info('another_func message')

def build():
    global logger
    logger = set_up_logger(logger, logging_level=20, save_logs=True, log_dir="/Users/joeyrich/Desktop/local/varseek/logs")
    another_func()
    logger.info("build message")

if __name__ == "__main__":
    build()