from fastapi import FastAPI
from model import RegressionTreeModel
import torch

# Create an instance of the FastAPI class
app = FastAPI()

# Define the first endpoint
@app.post("/predict")
def read_endpoint1(X):
    print(X)
    # Instantiate the model
    # model = RegressionTreeModel()

    # # Use the model for inference
    # # Assuming X_test is your test data
    # predictions = model(torch.tensor(X_test))
    # return {"price": predictions}

# Define the second endpoint
@app.get("/endpoint2")
def read_endpoint2():
    return {"message": "This is endpoint 2"}
