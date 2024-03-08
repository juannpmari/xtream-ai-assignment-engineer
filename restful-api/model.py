import torch
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load the serialized model using PyTorch
class RegressionTreeModel(torch.nn.Module):
    def __init__(self,model_name,directory):
        super(RegressionTreeModel, self).__init__()
        self.tree = joblib.load(Path(directory,model_name))

    def forward(self, x):
        return torch.tensor(self.tree.predict(x))


