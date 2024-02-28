from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import joblib
import logging

from sklearn.metrics import mean_squared_error,mean_absolute_error

#Pensar cómo hacerlo más genérico por si quiero cambiar de modelo

class Trainer():
   
    def __init__(self):
        self.data_path = './datasets/diamonds/processed_data'
        self.tree_reg = DecisionTreeRegressor()  # You can adjust the max_depth parameter to control the tree depth

    def load_data(self):
        self.X_train = pd.read_csv(Path(self.data_path,'X_train.csv'))
        self.y_train = pd.read_csv(Path(self.data_path,'y_train.csv'))
        # X_test = 
        # y_test =

    # Initialize the DecisionTreeRegressor

    # ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    # cols = ['carat', 'cut', 'color','clarity','depth', 'table', 'x', 'y', 'z']
    def train(self):
        ''' Train the model'''
        self.tree_reg.fit(self.X_train, self.y_train)
        joblib.dump(self.tree_reg, Path('./model_registry','tree_reg_model.pt'))

    def evaluate(self):
        # Predict on the test set
        y_pred = self.tree_reg.predict(self.X_test)

        # Evaluate the model
        mse = mean_absolute_error(self.y_test, y_pred)

        with open('metrics.txt','w+') as file:
            file.write(mse)
        logging.info("Mean absolute Error:", mse)

    def __call__(self):
        self.load_data()
        self.train()
        self.evaluate()


if __name__ == "__main__":
    Trainer()