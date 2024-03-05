from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import joblib
import logging
import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score


#Pensar cómo hacerlo más genérico por si quiero cambiar de modelo. Puede ser clase Trainer() y de ahí heredar RegTreeTrainer()

class Trainer():
   
    def __init__(self,current_time,config):
        self.data_path = Path(config['preprocessing']['output_path'],f"data_{current_time}")
        self.model_output_path = os.get(config['training']['model_output_path'],'/model_registry')
        self.metrics_output_path = os.get(config['training']['metrics_output_path'],'/metrics')
        os.makedirs(self.model_output_path,exist_ok=True)
        os.makedirs(self.metrics_output_path,exist_ok=True)
        self.current_time = current_time

        #Initialize DecisionTreeRegressor
        max_depth = os.get(config['training']['reg_tree_hyperparameters']['max_depth'],None)
        min_samples_leaf = os.get(config['training']['reg_tree_hyperparameters']['min_samples_leaf'],4)
        min_samples_split = os.get(config['training']['reg_tree_hyperparameters']['min_samples_split'],10)
        self.tree_reg = DecisionTreeRegressor(max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)
        
        

    def load_data(self):
        logging.info("Loading data")
        self.X_train = pd.read_csv(Path(self.data_path,f'X_train.csv'))
        self.y_train = pd.read_csv(Path(self.data_path,f'y_train.csv'))
        self.X_test = pd.read_csv(Path(self.data_path,f'X_test.csv'))
        self.y_test = pd.read_csv(Path(self.data_path,f'y_test.csv'))

    def train(self):
        ''' Train the model'''
        logging.info("Training model")
        self.tree_reg.fit(self.X_train, self.y_train)
        joblib.dump(self.tree_reg, Path(self.model_output_path,f'tree_reg_model_{self.current_time}.pt'))

    def evaluate(self):
        '''Evaluate the model on train set (using k-fold cross validation) and on test set'''

        logging.info("Performing offline evaluation")

        cv_maes = cross_val_score(self.tree_reg, self.X_train, self.y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_maes = -cv_maes
        cv_mean_mae = cv_maes.mean()

        logging.info("Mean CV MAE:", cv_mean_mae)

        y_pred_test = self.tree_reg.predict(self.X_test)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
    
        logging.info("Test set MAE: ", mae_test)



        # Predict on the test set
        # y_pred = self.tree_reg.predict(self.X_test)

        # Evaluate the model
        # mse = mean_absolute_error(self.y_test, y_pred)

        with open(Path(self.metrics_output_path,f'metrics_{self.current_time}.txt'),'w+') as file:
            file.write(f"Train set Mean CV MAE: {cv_mean_mae}")
            file.write(f"Test set MAE: {mae_test}")
        

    def __call__(self):
        self.load_data()
        self.train()
        self.evaluate()


# if __name__ == "__main__":
#     Trainer()()