# src/utils.py
import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import accuracy_score, f1_score # Changed from r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger as logging

# --- Your Existing Functions (Modified) ---

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    # --- This function is for classification (accuracy/f1), not regression (r2) ---
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = f1_score(y_test, y_test_pred, average='weighted') # Use F1-score
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# --- NEW FUNCTION FOR PREDICTION PIPELINE (Modified) ---

def load_all_artifacts():
    """
    Loads all required artifacts for prediction into memory.
    """
    try:
        artifacts_dir = os.path.join('artifacts')
        
        preprocessor_path = os.path.join(artifacts_dir, 'final_preprocessor.pkl')
        scaler_path = os.path.join(artifacts_dir, 'final_scaler.pkl')
        model_path = os.path.join(artifacts_dir, 'model_final_spatial_rf.pkl')
        label_encoder_path = os.path.join(artifacts_dir, 'label_encoder.pkl')
        
        imputed_features_path = os.path.join(artifacts_dir, 'imputed_features.csv')
        gnn_embeddings_path = os.path.join(artifacts_dir, 'gnn_embeddings.csv')
        raw_data_path = os.path.join(artifacts_dir, 'data.csv')

        logging.info("Loading all artifacts...")
        
        preprocessor = load_object(preprocessor_path)
        scaler = load_object(scaler_path)
        model = load_object(model_path)
        label_encoder = load_object(label_encoder_path)
        
        df_imputed_num = pd.read_csv(imputed_features_path)
        df_gnn = pd.read_csv(gnn_embeddings_path)
        df_raw = pd.read_csv(raw_data_path)
        
        logging.info("Artifacts loaded. Pre-processing regional data...")

        df_imputed_num['District'] = df_imputed_num['District'].str.strip().str.title()
        df_gnn['District'] = df_gnn['District'].str.strip().str.title()
        df_raw['District'] = df_raw['District'].str.strip().str.title()
        
        # --- THIS IS THE FIX ---
        # Create the indexed GNN dataframe
        df_gnn_indexed = df_gnn.drop_duplicates(subset=['District']).set_index('District')
        df_imputed_num = df_imputed_num.drop_duplicates(subset=['District']).set_index('District')
        
        cat_cols = ['Aging_Population', 'SES_Income', 'Healthcare_Access']
        def get_mode(x):
            try: return x.mode().iloc[0]
            except IndexError: return np.nan
        
        df_agg_cat = df_raw.groupby('District').agg({col: get_mode for col in cat_cols if col in df_raw.columns})

        regional_data_lookup = pd.concat([df_imputed_num, df_gnn_indexed, df_agg_cat], axis=1)
        
        fallback_numeric = df_imputed_num.mean()
        fallback_gnn = df_gnn_indexed.mean()
        fallback_categorical = df_agg_cat.mode().iloc[0]
        fallback_data = pd.concat([fallback_numeric, fallback_gnn, fallback_categorical])

        logging.info("Regional data lookup table created.")
        
        # --- FIX: Return df_gnn_indexed as well ---
        return model, preprocessor, scaler, label_encoder, regional_data_lookup, fallback_data, df_gnn_indexed

    except Exception as e:
        raise CustomException(e, sys)