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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create an instance of the FastAPI class
app = FastAPI()

def get_latest_model():
    logging.info("Getting latest model")
    
    # models = os.listdir('/model_registry')
    # models_dates = [datetime.strptime(model.split('_')[-1], '%Y-%m-%d-%H-%M-%S') for model in models]
    # latest_model_index = models_dates.index(max(models_dates))
    # latest_model = models[latest_model_index] if models else None
    # logging.info(f"Using model {latest_model}")

    directory = '/model_registry'#config['preprocessing']['data_path']
    models = os.listdir(directory)
    filtered_files = [f for f in models if f.startswith('tree_')]
    dates = [f.split('_')[-1] for f in filtered_files]
    sorted_dates = sorted(dates, reverse=True)
    most_recent_date = sorted_dates[0]
    latest_model = f"tree_reg_model_{most_recent_date}"
    logging.info(f"Loaded model {latest_model}")
    global model
    model = RegressionTreeModel(latest_model)

latest_model = get_latest_model()

class Diamond(BaseModel):
    carat:float
    cut:int
    color:int     
    clarity:int  
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
    predictions = model(diamonds_df)
  
    return {
        "msg": "Diamond price predicted succesfully",
        "pred_prices": str([pred.item() for pred in predictions])
    }

# uvicorn app:app --host 0.0.0.0 --port 8000
