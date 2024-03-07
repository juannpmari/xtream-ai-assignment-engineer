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

        #Could be static @property.
        self.categorical_variables = config['preprocessing']['categorical_variables']
    
    
    def load_data(self):
        '''Load raw data'''
        self.diamonds_df = pd.read_csv(self.data_path)

    def data_validation(self):
        '''Checks that all rows have valid values. Numerical magnitudes represent the price and physical properties that must be >0'''
        numeric_columns = self.diamonds_df.select_dtypes(include='number')
        self.diamonds_df = self.diamonds_df[(numeric_columns > 0).all(axis=1)]

    def encode_data(self):
        '''Encode ordinal categorical features'''
        for k,v in self.categorical_variables.items():
            encoder = OrdinalEncoder(categories=v)
            self.diamonds_df[k]  = encoder.fit_transform(self.diamonds_df[k].values.reshape(-1, 1))


    def scale_data(self):
        '''Scale all features (except target) using z-score scaling'''
        columns_to_scale = self.diamonds_df.columns[self.diamonds_df.columns != 'price']
        scaler = StandardScaler()
        scaler.fit(self.diamonds_df[columns_to_scale])
        scaled_data = scaler.transform(self.diamonds_df[columns_to_scale])
        scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

        # scaled_df['price'] = self.diamonds_df['price']
        
        # self.diamonds_df['price'].to_csv(Path(self.config['preprocessing']['data_path'],f'df_orig.csv'), index=False)
        # scaled_df.to_csv(Path(self.config['preprocessing']['data_path'],f'df_asig.csv'), index=False)

        self.scaled_df_x = scaled_df
 

    def split_data(self):
        '''Split processed data into train and test sets, and them to output_path'''
        X = self.scaled_df_x #self.scaled_df.drop(columns=['price'])
        y = self.diamonds_df['price'] #self.scaled_df['price']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train.to_csv(Path(self.output_path,f'X_train.csv'), index=False)
        X_test.to_csv(Path(self.output_path,f'X_test.csv'), index=False)
        y_train.to_csv(Path(self.output_path,f'y_train.csv'), index=False)
        y_test.to_csv(Path(self.output_path,f'y_test.csv'), index=False)
        return self.current_time
        logging.info("Finished preprocessing data")

    def __call__(self):
        self.load_data()
        self.data_validation()
        self.encode_data()
        self.scale_data()
        current_time = self.split_data()
        return current_time

# if __name__ == "__main__":
#     Preprocessor()()



# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split

# # Sample data
# X = ...
# y = ...

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define your custom function for imputation
# def custom_func(data):
#     # Your custom imputation logic here
#     return data.fillna(data.mean())  # Example: filling missing values with mean

# # Define your custom preprocessing steps
# numeric_features = ['numerical_feature_1', 'numerical_feature_2']
# categorical_features = ['categorical_feature_1', 'categorical_feature_2']

# numeric_transformer = Pipeline(steps=[
#     ('imputer', FunctionTransformer(func=custom_func)),
#     ('scaler', StandardScaler())
# ])

# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

# # Combine your custom preprocessing steps for numerical and categorical features
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)
#     ])

# # Define the pipeline with your custom preprocessing steps
# custom_pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor)
# ])

# # Fit the pipeline to your data
# custom_pipeline.fit(X_train)

# # Transform your data using the fitted pipeline
# X_train_transformed = custom_pipeline.transform(X_train)
# X_test_transformed = custom_pipeline.transform(X_test)

# # Now you can use X_train_transformed and X_test_transformed for further analysis or modeling
