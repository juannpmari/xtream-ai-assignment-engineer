import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import os
import logging


logger = logging.getLogger(__name__)

class Preprocessor():
    
    def __init__(self,config):
        logging.info("Initializing preprocessor")
        self.config = config
        self.data_path = Path(config['preprocessing']['data_path'],'diamonds.csv')
        self.current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.output_path = f"{config['preprocessing']['output_path']}/data_{self.current_time}"
        os.makedirs(self.output_path,exist_ok=True)
        self.categorical_variables = config['preprocessing']['categorical_variables']
    
    def load_data(self):
        '''Load raw data'''
        try:
            self.diamonds_df = pd.read_csv(self.data_path)
        except Exception as e:
            logging.error("Error loading raw data: ",e)

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
            columns_to_scale = self.diamonds_df.columns[self.diamonds_df.columns != 'price']
            scaler = StandardScaler()
            scaler.fit(self.diamonds_df[columns_to_scale])
            scaled_data = scaler.transform(self.diamonds_df[columns_to_scale])
            scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)
            self.scaled_df_x = scaled_df
        except Exception as e:
            logging.error("Error scaling data: ",e)
 
    def split_data(self):
        '''Split processed data into train and test sets, and them to output_path'''
        try:
            X = self.scaled_df_x
            y = self.diamonds_df['price']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train.to_csv(Path(self.output_path,f'X_train.csv'), index=False)
            X_test.to_csv(Path(self.output_path,f'X_test.csv'), index=False)
            y_train.to_csv(Path(self.output_path,f'y_train.csv'), index=False)
            y_test.to_csv(Path(self.output_path,f'y_test.csv'), index=False)
            logging.info("Finished preprocessing data")
            return self.current_time
        except Exception as e:
            logging.error("Error performing train/test split: ",e)

    def __call__(self):
        self.load_data()
        self.data_validation()
        self.encode_data()
        self.scale_data()
        current_time = self.split_data()
        return current_time