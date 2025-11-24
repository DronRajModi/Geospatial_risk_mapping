# src/components/data_transformation.py
import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException


def _load_df(maybe_df_or_path):
 
    if isinstance(maybe_df_or_path, pd.DataFrame):
        return maybe_df_or_path.copy()
    if isinstance(maybe_df_or_path, str):
        if not os.path.exists(maybe_df_or_path):
            raise CustomException(FileNotFoundError(f"File not found: {maybe_df_or_path}"), sys)
        ext = os.path.splitext(maybe_df_or_path)[1].lower()
        if ext in [".csv"]:
            return pd.read_csv(maybe_df_or_path)
        elif ext in [".parquet", ".pq"]:
            return pd.read_parquet(maybe_df_or_path)
        else:
         
            return pd.read_csv(maybe_df_or_path)
    raise CustomException(ValueError("Input must be a pandas DataFrame or a valid file path."), sys)


def _get_target_column(df: pd.DataFrame):
    """
    Return name of target column if present, otherwise last column name.
    Prioritize column named 'target', 'label', or 'y' if present.
    """
    for candidate in ["target", "label", "y"]:
        if candidate in df.columns:
            return candidate
  
    return df.columns[-1]


def build_transformer(df: pd.DataFrame):
    """
    Build and return a ColumnTransformer based on df's dtypes.
    Numeric: SimpleImputer(median) + StandardScaler
    Categorical: SimpleImputer(constant='missing') + OneHotEncoder(handle_unknown='ignore')
    """
    try:
        numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        target_col = _get_target_column(df)
        if target_col in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target_col]

        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if target_col in categorical_cols:
            categorical_cols = [c for c in categorical_cols if c != target_col]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        transformers = []
        if len(numeric_cols) > 0:
            transformers.append(("num", numeric_pipeline, numeric_cols))
        if len(categorical_cols) > 0:
            transformers.append(("cat", categorical_pipeline, categorical_cols))

        if len(transformers) == 0:
            raise ValueError("No numeric or categorical features detected to build transformer.")

        ct = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
        return ct, numeric_cols, categorical_cols

    except Exception as e:
        raise CustomException(e, sys)


def initiate_data_transformation(train_input, test_input, artifacts_dir="artifacts"):
  
    try:
        os.makedirs(artifacts_dir, exist_ok=True)

        # Load
        train_df = _load_df(train_input)
        test_df = _load_df(test_input)

        if train_df.shape[0] == 0:
            raise ValueError("Train dataframe is empty.")
        if test_df.shape[0] == 0:
            raise ValueError("Test dataframe is empty.")


        target_col = _get_target_column(train_df)

    
        train_missing = train_df[target_col].isna().sum()
        test_missing = test_df[target_col].isna().sum()
        if train_missing > 0 or test_missing > 0:
            train_df = train_df.dropna(subset=[target_col])
            test_df = test_df.dropna(subset=[target_col])
            print(f"[data_transformation] Dropped {train_missing} rows from train, {test_missing} rows from test due to missing target '{target_col}'.")

     
        combined_for_schema = pd.concat(
            [train_df.drop(columns=[target_col], errors="ignore"),
             test_df.drop(columns=[target_col], errors="ignore")],
            axis=0, ignore_index=True
        )

        transformer, numeric_cols, categorical_cols = build_transformer(combined_for_schema)

      
        X_train_df = train_df.drop(columns=[target_col], errors="ignore")
        X_test_df = test_df.drop(columns=[target_col], errors="ignore")

        transformer.fit(X_train_df)
        X_train_arr = transformer.transform(X_train_df)
        X_test_arr = transformer.transform(X_test_df)

       
        y_train = train_df[target_col].astype(str).values
        y_test = test_df[target_col].astype(str).values

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

       
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        with open(label_encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)
        print(f"[data_transformation] Encoded target classes: {list(label_encoder.classes_)}")

      
        train_arr = np.hstack([X_train_arr, y_train])
        test_arr = np.hstack([X_test_arr, y_test])

   
        transformer_path = os.path.join(artifacts_dir, "transformer.pkl")
        with open(transformer_path, "wb") as f:
            pickle.dump(transformer, f)

        meta = {
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "target_col": target_col,
            "feature_count": X_train_arr.shape[1],
        }
        meta_path = os.path.join(artifacts_dir, "transformer_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"[data_transformation] train_arr shape: {train_arr.shape}, test_arr shape: {test_arr.shape}")
        print(f"[data_transformation] transformer saved to: {transformer_path}")

        return train_arr, test_arr, transformer

    except Exception as e:
        raise CustomException(e, sys)


class DataTransformation:
    def __init__(self):
        pass

    def initiate_data_transformation(self, train_input, test_input, artifacts_dir="artifacts"):
        return initiate_data_transformation(train_input, test_input, artifacts_dir)
