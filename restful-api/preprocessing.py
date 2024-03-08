import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import os
import logging


logger = logging.getLogger(__name__)

class Preprocessor():
    
    def __init__(self,config,diamonds_df):
        logging.info("Initializing preprocessor")
        self.diamonds_df = diamonds_df
        self.categorical_variables = config['preprocessing']['categorical_variables']
        self.scaler_mean = config['deployment']['scaler_mean']
        self.scaler_std = config['deployment']['scaler_std']

    def data_validation(self):
        '''Checks that all rows have valid values. Numerical magnitudes represent the price and physical properties that must be >0'''
        numeric_columns = self.diamonds_df.select_dtypes(include='number')
        self.diamonds_df = self.diamonds_df[(numeric_columns > 0).all(axis=1)]

    def encode_data(self):
        '''Encode ordinal categorical features'''
        try:
            for k,v in self.categorical_variables.items():
                encoder = OrdinalEncoder(categories=v)
                self.diamonds_df[k]  = encoder.fit_transform(self.diamonds_df[k].values.reshape(-1, 1))
        except Exception as e:
            logging.error("Error encoding data: ",e)

    def scale_data(self):
        '''Scale all features (except target) using z-score scaling'''
        try:
            columns_to_scale = self.diamonds_df.columns
            scaler = StandardScaler()
            scaler.mean_ = self.scaler_mean
            scaler.scale_ = self.scaler_std
            scaled_data = scaler.transform(self.diamonds_df[columns_to_scale])
            scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
            self.scaled_df = scaled_df
        except Exception as e:
            logging.error("Error scaling data: ",e)
 

    def __call__(self):
        self.data_validation()
        self.encode_data()
        self.scale_data()
        return self.scaled_df