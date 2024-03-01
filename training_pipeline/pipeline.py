import json
from preprocessing import Preprocessor
from training import Trainer

if __name__=="__main__":
    config_path = 'config.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    current_time = Preprocessor(config)()
    Trainer(current_time)()