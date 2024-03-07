import json
from preprocessing import Preprocessor
from training import RegTreeTrainer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__=="__main__":
    config_path = 'config.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    current_time = Preprocessor(config)()
    RegTreeTrainer(current_time,config)()