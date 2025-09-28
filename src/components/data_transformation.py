import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_df):
        """
        Creates preprocessing pipelines for numerical and categorical features.
        Only uses columns present in train_df and excludes the target column.
        """
        try:
            target_column_name = "Disease_NCD"

            numerical_columns_all = [
                'Year', 'Age', 'Obese(%)', 'Cholesterol(%)', 'PM2.5',
                'NO2', 'SO2', 'Population_Density', 'Sleep_Hours', 'NDVI'
            ]
            categorical_columns_all = [
                'District', 'Gender', 'Tobacco_Use', 'Alcohol_Use', 'Hypertension',
                'Diabetes', 'Aging_Population', 'SES_Income', 'Healthcare_Access',
                'Urban_or_Rural'
            ]  # target removed

            # Keep only columns present in the dataset
            numerical_columns = [col for col in numerical_columns_all if col in train_df.columns]
            categorical_columns = [col for col in categorical_columns_all if col in train_df.columns]

            if len(numerical_columns) + len(categorical_columns) == 0:
                raise ValueError("No valid input features found in train CSV.")

            logging.info(f"Numerical columns used: {numerical_columns}")
            logging.info(f"Categorical columns used: {categorical_columns}")

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ],
                remainder='drop'
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Strip spaces from column names
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(train_df)

            target_column_name = "Disease_NCD"

            # Check target existence
            if target_column_name not in train_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in train data")
            if target_column_name not in test_df.columns:
                raise ValueError(f"Target column '{target_column_name}' not found in test data")

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[target_column_name], errors='ignore')
            input_feature_test_df = test_df.drop(columns=[target_column_name], errors='ignore')

            target_feature_train_arr = train_df[target_column_name].to_numpy().reshape(-1)
            target_feature_test_arr = test_df[target_column_name].to_numpy().reshape(-1)

            # Apply preprocessing
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Ensure matching number of rows
            if input_feature_train_arr.shape[0] != target_feature_train_arr.shape[0]:
                raise ValueError(f"Number of rows in train features ({input_feature_train_arr.shape[0]}) "
                                 f"does not match target ({target_feature_train_arr.shape[0]})")
            if input_feature_test_arr.shape[0] != target_feature_test_arr.shape[0]:
                raise ValueError(f"Number of rows in test features ({input_feature_test_arr.shape[0]}) "
                                 f"does not match target ({target_feature_test_arr.shape[0]})")

            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
