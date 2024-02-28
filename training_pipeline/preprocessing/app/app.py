from typing import Any
import pandas as pd

class Preprocessing():
    
    def __init__(self) -> None:
        self.data_path = './datasets/diamonds/diamonds.csv'
        self.output_path = './datasets/diamonds/processed_data'
    
    def load_data(self):
        self.diamonds_df = pd.read_csv(self.data_path)

    def remove_outliers(self):
        pass

    def encode_data(self):
        pass

    def scale_data(self):
        pass

    def split_data(self):
        #save data to self.output_path
        pass

    def __call__(self):
        self.load_data()
        self.remove_outliers()
        self.encode_data()
        self.split_data()
        self.scale_data()

if __name__ == "__main__":
    Preprocessing()



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
