import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC 

from src.exception import CustomException
from src.logger import logger as logging

def _load_df(maybe_df_or_path):
    if isinstance(maybe_df_or_path, pd.DataFrame):
        return maybe_df_or_path.copy()
    if isinstance(maybe_df_or_path, str):
        if not os.path.exists(maybe_df_or_path):
             raise CustomException(FileNotFoundError(f"File not found: {maybe_df_or_path}"), sys)
        ext = os.path.splitext(maybe_df_or_path)[1].lower()
        if ext == ".csv": return pd.read_csv(maybe_df_or_path)
        elif ext in [".parquet", ".pq"]: return pd.read_parquet(maybe_df_or_path)
        else: return pd.read_csv(maybe_df_or_path)
    raise CustomException(ValueError("Input must be a pandas DataFrame or a valid file path."), sys)

def _get_target_column(df: pd.DataFrame):
    target_candidates = ["Disease_NCD", "target", "label", "y"]
    for candidate in target_candidates:
        if candidate in df.columns: return candidate
    logging.warning("Standard target column name not found, falling back to the last column.")
    return df.columns[-1]

def build_preprocessing_pipeline(df: pd.DataFrame):
    try:
        target_col = _get_target_column(df)
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        if 'Year' in numeric_cols: numeric_cols.remove('Year')
            
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if target_col in categorical_cols: categorical_cols.remove(target_col)
        if 'District' in categorical_cols: categorical_cols.remove('District')

        logging.info(f"Numeric columns for preprocessing: {numeric_cols}")
        logging.info(f"Categorical columns for preprocessing: {categorical_cols}")

        categorical_features_indices = [
            df.columns.get_loc(c) for c in categorical_cols if c in df
        ]
        logging.info(f"Categorical feature indices found for SMOTE-NC: {categorical_features_indices}")

        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        transformers = []
        if numeric_cols: transformers.append(("num", numeric_pipeline, numeric_cols))
        if categorical_cols: transformers.append(("cat", categorical_pipeline, categorical_cols))

        if not transformers:
             if numeric_cols:
                 logging.info("Building preprocessor for numeric features only (imputation).")
                 return numeric_pipeline, numeric_cols, [], [] 
             else:
                raise ValueError("No numeric or categorical features detected to build transformer.")

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        
        return preprocessor, numeric_cols, categorical_cols, categorical_features_indices

    except Exception as e:
        raise CustomException(e, sys)

def initiate_data_transformation_smote(train_input, test_input, artifacts_dir="artifacts"):
    try:
        logging.info("Initiating Data Transformation Component (with SMOTE-NC)")
        os.makedirs(artifacts_dir, exist_ok=True)

        train_df = _load_df(train_input)
        test_df = _load_df(test_input)
        logging.info(f"Loaded train ({train_df.shape}) and test ({test_df.shape}) data.")

        target_col = _get_target_column(train_df)
        logging.info(f"Target column identified: {target_col}")

        train_df.dropna(subset=[target_col], inplace=True)
        test_df.dropna(subset=[target_col], inplace=True)
        logging.info(f"[data_transformation] Dropped {train_df[target_col].isnull().sum()} rows from train, {test_df[target_col].isnull().sum()} rows from test due to missing target '{target_col}'.")


        X_train_df = train_df.drop(columns=[target_col], errors='ignore')
        y_train_series = train_df[target_col]
        X_test_df = test_df.drop(columns=[target_col], errors='ignore')
        y_test_series = test_df[target_col]
        
        logging.info(f"Original training class distribution:\n{y_train_series.value_counts().to_string()}")

        preprocessor, numeric_cols, categorical_cols, categorical_features_indices = build_preprocessing_pipeline(X_train_df)
        
        logging.info("Applying SMOTE-NC for class imbalance...")
        
        min_class_count = y_train_series.value_counts().min()
        k_neighbors_smote = max(1, min_class_count - 1)
        
        if k_neighbors_smote < 1:
            logging.warning(f"Smallest class has only {min_class_count} sample(s). SMOTE cannot be applied. Using original data.")
            X_train_resampled = X_train_df
            y_train_resampled = y_train_series
        else:
            try:
                logging.info(f"Setting SMOTE-NC k_neighbors to: {k_neighbors_smote}")
                smote_nc = SMOTENC(
                    categorical_features=categorical_features_indices, 
                    random_state=42, 
                    k_neighbors=k_neighbors_smote
                ) 
                
                X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train_df, y_train_series)
                logging.info(f"SMOTE-NC finished. New training shape: {X_train_resampled.shape}")
                logging.info(f"New training class distribution:\n{pd.Series(y_train_resampled).value_counts().to_string()}")

            except Exception as e:
                logging.warning(f"SMOTE-NC failed: {e}. Using original training data.")
                X_train_resampled = X_train_df
                y_train_resampled = y_train_series

        logging.info("Applying preprocessing (imputer/encoder)...")
        X_train_processed = preprocessor.fit_transform(X_train_resampled)
        X_test_processed = preprocessor.transform(X_test_df)
        logging.info("Data preprocessing (imputation/encoding) complete.")

        logging.info("Applying StandardScaler.")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        logging.info("Data scaling complete.")

        logging.info("Encoding target variable.")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_resampled)
        y_test_encoded = label_encoder.transform(y_test_series)
        
        logging.info(f"[data_transformation] Encoded target classes: {list(label_encoder.classes_)}")

        y_train_encoded = y_train_encoded.reshape(-1, 1)
        y_test_encoded = y_test_encoded.reshape(-1, 1)

        train_arr_scaled_aug = np.hstack([X_train_scaled, y_train_encoded])
        test_arr_scaled = np.hstack([X_test_scaled, y_test_encoded])
        preprocessor_path = os.path.join(artifacts_dir, "transformer_smote.pkl") # New name
        scaler_path = os.path.join(artifacts_dir, "scaler_smote.pkl") # New name
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        
        with open(preprocessor_path, "wb") as f: pickle.dump(preprocessor, f)
        with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
        with open(label_encoder_path, "wb") as f: pickle.dump(label_encoder, f)

        try:
             feature_names_out = preprocessor.get_feature_names_out()
        except AttributeError:
             if isinstance(preprocessor, Pipeline):
                 feature_names_out = numeric_cols
             else:
                 cat_features_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols) if categorical_cols else []
                 feature_names_out = numeric_cols + list(cat_features_out)
        
        meta = {
            "feature_names_out": list(feature_names_out),
            "target_classes": list(label_encoder.classes_),
            "feature_count_processed": X_train_scaled.shape[1]
        }
        meta_path = os.path.join(artifacts_dir, "transformer_meta_smote.pkl") # New name
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
            
        logging.info(f"[data_transformation] AUGMENTED train_arr shape: {train_arr_scaled_aug.shape}, test_arr shape: {test_arr_scaled.shape}")
        logging.info(f"[data_transformation] SMOTE objects saved.")

        return train_arr_scaled_aug, test_arr_scaled, preprocessor_path, scaler_path

    except Exception as e:
        logging.error(f"Error in data transformation: {e}")
        raise CustomException(e, sys)