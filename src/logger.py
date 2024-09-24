import logging
import os
from datetime import datetime

# specify the log file name
LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'

# create a logs directory if it does not exist
logs_paths = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_paths, exist_ok=True)

# specify the log file path
LOG_FILE_PATH = os.path.join(logs_paths, LOG_FILE)

# configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(levelno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
