import logging
import os
import shutil

def get_logger(name=__name__, level=logging.INFO, log_file='program.log', log_dir='logs'):
    """
    This function configures a logger with the specified name, log level, and log file.
    You can see the log files in the specified logging logs/program.log

    Return:
        logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # Remove and recreate the log directory for each run
    os.makedirs(log_dir, exist_ok=True)
    # Remove all handlers associated with the logger
    logger.handlers.clear()

    # StreamHandler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler for file output
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger