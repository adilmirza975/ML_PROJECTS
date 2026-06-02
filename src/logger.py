# this logger is for the purpose that if any execution that happens it should be able to log the file

import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%_M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)



logging.basicConfig(
  filename=LOG_FILE_PATH,
  format="[%(asctime)s %(lineno)d - %(levelname)s - %(message)s ]",
  level=logging.INFO,
  
)
