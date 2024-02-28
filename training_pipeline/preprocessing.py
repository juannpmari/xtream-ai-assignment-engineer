# from typing import Any
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import os



class Preprocessor():
    
    def __init__(self) -> None:
        self.data_path = '/datasets/diamonds/diamonds.csv'
        self.output_path = '/datasets/diamonds/processed_data'

        #Could be static @property. Se podría leer desde un config.json para que sea más dinámico
        self.categorical_variables = {
            'cut':[['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']],
            'color':[['D','E','F','G','H','I','J']],
            'clarity':[['IF','VVS1', 'VVS2','VS1','VS2','SI1','SI2','I1']]
        }
    
    def load_data(self):
        self.diamonds_df = pd.read_csv(self.data_path)

    def data_validation(self):
        '''All numerical magnitudes represent physical properties of the diamond, so must be > 0'''
        numeric_columns = self.diamonds_df.select_dtypes(include='number')
        self.diamonds_df = self.diamonds_df[(numeric_columns > 0).all(axis=1)]

    def remove_outliers(self):
        pass

    def encode_data(self):
        for k,v in self.categorical_variables.items():
            encoder = OrdinalEncoder(categories=v)
            self.diamonds_df[k]  = encoder.fit_transform(self.diamonds_df[k].values.reshape(-1, 1))

    def scale_data(self):
        '''Not necessary for regression trees'''
        pass

    def split_data(self):
        '''Save data to self.output_path'''
        X = self.diamonds_df.drop(columns=['price'])  # Features
        y = self.diamonds_df['price']  # Target variable

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        os.makedirs(self.output_path,exist_ok=True)

        X_train.to_csv(Path(self.output_path,'X_train.csv'), index=False)
        X_test.to_csv(Path(self.output_path,'X_test.csv'), index=False)
        y_train.to_csv(Path(self.output_path,'y_train.csv'), index=False)
        y_test.to_csv(Path(self.output_path,'y_test.csv'), index=False)

    def __call__(self):
        self.load_data()
        self.data_validation()
        self.encode_data()
        self.split_data()
        # self.scale_data()

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
