from datetime import datetime
from fastapi import FastAPI
from model import RegressionTreeModel
import torch
import os
from pydantic import BaseModel
import logging
import numpy as np
import pandas as pd
from typing import List
from preprocessing import Preprocessor
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


config_path = 'config.json'
with open(config_path, 'r') as file:
    config = json.load(file)

app = FastAPI()

def get_latest_model():
    logging.info("Getting latest model")
    directory = config['training']['model_output_path']
    models = os.listdir(directory)
    filtered_files = [f for f in models if f.startswith('tree_')]
    dates = [f.split('_')[-1] for f in filtered_files]
    sorted_dates = sorted(dates, reverse=True)
    most_recent_date = sorted_dates[0]
    latest_model = f"tree_reg_model_{most_recent_date}"
    logging.info(f"Loaded model {latest_model}")
    global model
    model = RegressionTreeModel(latest_model,directory)

latest_model = get_latest_model()

class Diamond(BaseModel):
    carat:float
    cut:str
    color:str     
    clarity:str  
    depth:float  
    table:float   
    x:float    
    y:float    
    z:float 

@app.post("/predict")
def predict(diamonds:List[Diamond]):
    logging.info("Running prediction...")
    diamond_dict_list=[]
    for diamond in diamonds:
        diamond_dict_list.append(diamond.dict())
    diamonds_df = pd.DataFrame(diamond_dict_list)
    
    try:
        diamonds_df_proc = Preprocessor(config,diamonds_df)()
        predictions = model(diamonds_df_proc)
        return {
            "msg": "Diamond price predicted succesfully",
            "pred_prices": str([pred.item() for pred in predictions])
        }
    except Exception as e:
        logging.error("Error making predictions")
        return {
            "msg": f"An error ocurred: {e}",
            "pred_prices": "[]"
        }