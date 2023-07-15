from loguru import logger

def create_logger(log_path):
    logger.add(log_path, level='DEBUG', format="{time}:{module}:{level}:{message}")
    return logger