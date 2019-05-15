'''
    The logging system for easy hyper-parameter recording.
'''

import logging

def init_logging():
    '''
        Initialize the logger and return it. The log file will be at "log\trainLog.txt".
    '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(r"log\trainLog.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
