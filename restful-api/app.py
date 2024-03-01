from datetime import datetime
from fastapi import FastAPI
from model import RegressionTreeModel
import torch
import os 

# Create an instance of the FastAPI class
app = FastAPI()

def get_latest_model():
    models = os.listdir('model_registry')
    models_dates = [datetime.strptime(model.split('_')[-1], '%Y-%m-%d-%H-%M-%S') for model in models]
    latest_model_index = models_dates.index(max(models_dates))
    latest_model = models[latest_model_index] if models else None
    return latest_model

# Define the first endpoint
@app.post("/predict")
def predict(X):
    print(X)
    # Instantiate the model
    
    # latest_model = get_latest_model()
    # model = RegressionTreeModel(latest_model)

    # # Use the model for inference
    # # Assuming X_test is your test data
    # predictions = model(torch.tensor(X_test))
    # return {"price": predictions}

# Define the second endpoint
@app.get("/predict_batch")
def predict_batch(X_df):
    return {"message": "This is endpoint 2"}