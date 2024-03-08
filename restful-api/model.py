import torch
import joblib
from pathlib import Path

# Load the serialized model using PyTorch
class RegressionTreeModel(torch.nn.Module):
    def __init__(self,model_name):
        super(RegressionTreeModel, self).__init__()
        self.tree = joblib.load(Path('/model_registry',model_name))

    def forward(self, x):
        return torch.tensor(self.tree.predict(x))


