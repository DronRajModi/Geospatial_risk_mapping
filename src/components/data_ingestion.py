# src/components/data_ingestion.py
import os
import sys
import argparse
import pandas as pd

from src.exception import CustomException
from src.components import data_transformation, model_trainer


def _read_path_or_df(maybe_path):
    """
    If string and exists -> read as CSV/parquet, else if DataFrame, return it.
    """
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


def initiate_data_ingestion(train_path, test_path, artifacts_dir="artifacts"):
    """
    Loads train/test (paths or dataframes), calls data_transformation and model training.
    Returns the model training result dict.
    """
    try:
        # Load raw
        train_df = _read_path_or_df(train_path)
        test_df = _read_path_or_df(test_path)

        print(f"[data_ingestion] Loaded train rows: {train_df.shape[0]}, test rows: {test_df.shape[0]}")

        # Call data transformation
        train_arr, test_arr, transformer = data_transformation.initiate_data_transformation(
            train_df, test_df, artifacts_dir=artifacts_dir
        )

        # Call model trainer
        result = model_trainer.initiate_model_training(train_arr, test_arr, artifacts_dir=artifacts_dir)

        print("[data_ingestion] training result:", result)
        return result

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    """
    Example usage:
      python src/components/data_ingestion.py --train data/train.csv --test data/test.csv --artifacts artifacts
    """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--train", type=str, default="artifacts/train.csv", help="Path to training CSV")
        parser.add_argument("--test", type=str, default="artifacts/test.csv", help="Path to test CSV")
        parser.add_argument("--artifacts", type=str, default="artifacts", help="Artifacts directory")

        args = parser.parse_args()

        initiate_data_ingestion(args.train, args.test, artifacts_dir=args.artifacts)

    except Exception as e:
        # If argparse or other top-level error
        raise CustomException(e, sys)