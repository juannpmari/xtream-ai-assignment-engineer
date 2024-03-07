from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import joblib
import logging
import os
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Trainer(ABC):
    '''
    This abstraction allows to easily add trainer classes for new models in the future, as long as they implement train() and evaluate() methods.
    '''
    def __init__(self,date,config):
        logging.info(f"Initializing model trainer for dataset data_{date}")
        self.data_path = Path(config['preprocessing']['output_path'],f"data_{date}")
        self.model_output_path = config['training'].get('model_output_path','/model_registry')
        self.metrics_output_path = config['training'].get('metrics_output_path','/metrics')
        os.makedirs(self.model_output_path,exist_ok=True)
        os.makedirs(self.metrics_output_path,exist_ok=True)
        self.date = date
    
    def load_data(self):
        '''Load train and test data from .csv files'''
        logging.info("Loading data")
        try:
            self.X_train = pd.read_csv(Path(self.data_path,f'X_train.csv'))
            self.y_train = pd.read_csv(Path(self.data_path,f'y_train.csv'))
            self.X_test = pd.read_csv(Path(self.data_path,f'X_test.csv'))
            self.y_test = pd.read_csv(Path(self.data_path,f'y_test.csv'))
            logging.info(f"Loaded X:{len(self.X_train)}, y:{len(self.y_train)} samples for train set")
            logging.info(f"Loaded X:{len(self.X_test)}, y:{len(self.y_test)} samples for test set")
        except Exception as e:
            logging.error("Error loading data: ",e)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

class RegTreeTrainer(Trainer):
    def __init__(self,date,config):
        super().__init__(date,config)
        #Initialize DecisionTreeRegressor
        max_depth = config['training']['reg_tree_hyperparameters'].get('max_depth',None)
        min_samples_leaf = config['training']['reg_tree_hyperparameters'].get('min_samples_leaf',4)
        min_samples_split = config['training']['reg_tree_hyperparameters'].get('min_samples_split',10)
        self.tree_reg = DecisionTreeRegressor(max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split)

    def train(self):
        ''' Train the model'''
        logging.info("Training model")
        try:
            self.tree_reg.fit(self.X_train, self.y_train)
        except Exception as e:
            logging.error("Error training model: ",e)
        joblib.dump(self.tree_reg, Path(self.model_output_path,f'tree_reg_model_{self.date}.pt'))

    def evaluate(self):
        '''Evaluate the model on train set (using k-fold cross validation) and on test set'''
        logging.info("Performing offline evaluation")
        try:
            cv_maes = cross_val_score(self.tree_reg, self.X_train, self.y_train, cv=5, scoring='neg_mean_absolute_error')
            cv_maes = -cv_maes
            cv_mean_mae = cv_maes.mean()
            y_pred_test = self.tree_reg.predict(self.X_test)
            mae_test = mean_absolute_error(self.y_test, y_pred_test)
        except Exception as e:
            logging.error("Error evaluating model: ",e)

        try:
            with open(Path(self.metrics_output_path,f'metrics_{self.date}.txt'),'w+') as file:
                file.write(f"Train set Mean CV MAE: {cv_mean_mae}\n")
                file.write(f"Test set MAE: {str(mae_test)}")
            logging.info("Pipeline finished succesfully")
        except Exception as e:
            logging.error("Error saving model metrics: ",e)

    def __call__(self):
        self.load_data()
        self.train()
        self.evaluate()