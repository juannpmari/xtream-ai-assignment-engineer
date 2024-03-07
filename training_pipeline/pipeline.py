import json
from preprocessing import Preprocessor
from training import RegTreeTrainer
import logging
import schedule
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

processed_datasets = []
new_data = None

def get_latest_dataset(config):
    directory = config['preprocessing']['data_path']
    files = os.listdir(directory)
    filtered_files = [f for f in files if f.startswith('diamonds_')]
    dates = [f.split('_')[-1] for f in filtered_files]
    sorted_dates = sorted(dates, reverse=True)
    most_recent_date = sorted_dates[0]
    most_recent_filename = f"diamonds_{most_recent_date}"
    return most_recent_filename

def run_pipeline(config,latest_dataset=None):
    latest_dataset = get_latest_dataset(config) if latest_dataset is None else latest_dataset
    logging.info(f"Launching new training pipeline run with dataset {latest_dataset}")
    date = latest_dataset.split('_')[-1].split('.')[0]
    Preprocessor(config,date)()
    RegTreeTrainer(date,config)()
    processed_datasets.append(latest_dataset)
    logging.info(f"Added {latest_dataset} to processed datasets")
    global new_data
    new_data = None

def check_for_new_data(config):
    latest_dataset = get_latest_dataset(config)
    global new_data
    if latest_dataset not in processed_datasets:
        logging.info(f"Found new dataset: {str(latest_dataset)}")
        new_data = latest_dataset
    else:
        logging.info("No new datasets found")
        new_data = None


if __name__=="__main__":
    config_path = 'config.json'
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    run_pipeline(config)
    schedule.every(50).minutes.do(check_for_new_data,config=config)
    
    while True:
        schedule.run_pending()
        if new_data is not None:
            run_pipeline(config,new_data)
        time.sleep(1)