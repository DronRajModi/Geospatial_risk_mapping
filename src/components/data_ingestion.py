import os
import sys
import pandas as pd
from src.exception import CustomException
from src.components import data_transformation, model_trainer


class DataIngestion:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir

    def _read_path_or_df(self, maybe_path):
        """If string and exists -> read as CSV/parquet, else if DataFrame, return it."""
        if isinstance(maybe_path, pd.DataFrame):
            return maybe_path
        if isinstance(maybe_path, str):
            if not os.path.exists(maybe_path):
                raise FileNotFoundError(f"File not found: {maybe_path}")
            ext = os.path.splitext(maybe_path)[1].lower()
            if ext == ".csv":
                return pd.read_csv(maybe_path)
            elif ext in [".parquet", ".pq"]:
                return pd.read_parquet(maybe_path)
            else:
                return pd.read_csv(maybe_path)
        raise ValueError("Input must be a file path or a pandas DataFrame.")

    def initiate_data_ingestion(self, train_path="artifacts/train.csv", test_path="artifacts/test.csv"):
        """
        Loads train/test (paths or dataframes), calls data_transformation and model training.
        Returns the model training result dict.
        """
        try:
            train_df = self._read_path_or_df(train_path)
            test_df = self._read_path_or_df(test_path)

            print(f"[DataIngestion] Loaded train rows: {train_df.shape[0]}, test rows: {test_df.shape[0]}")

            train_arr, test_arr, transformer = data_transformation.initiate_data_transformation(
                train_df, test_df, artifacts_dir=self.artifacts_dir
            )

            result = model_trainer.initiate_model_training(train_arr, test_arr, artifacts_dir=self.artifacts_dir)

            print("[DataIngestion] training result:", result)
            return result

        except Exception as e:
            raise CustomException(e, sys)
