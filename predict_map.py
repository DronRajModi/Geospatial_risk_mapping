# predict_map.py
# (Corrected Version)

import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
import matplotlib.pyplot as plt

# --- 1. SETUP LOGGER ---
logging.basicConfig(
    format="[ %(asctime)s ] %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PredictionLogger")

# --- 2. DEFINE FILE PATHS TO ARTIFACTS ---
SHAPEFILE_PATH = os.path.join('notebook', 'india_districts.shp')
IMPUTED_FEATURES_PATH = os.path.join('artifacts', 'imputed_features.csv') # Numeric
GNN_EMBEDDINGS_PATH = os.path.join('artifacts', 'gnn_embeddings.csv')   # GNN
RAW_DATA_PATH = os.path.join('artifacts', 'data.csv')                     # Categorical

FINAL_PREPROCESSOR_PATH = os.path.join('artifacts', 'final_preprocessor.pkl')
FINAL_SCALER_PATH = os.path.join('artifacts', 'final_scaler.pkl')
FINAL_MODEL_PATH = os.path.join('artifacts', 'model_final_spatial_rf.pkl')
LABEL_ENCODER_PATH = os.path.join('artifacts', 'label_encoder.pkl')

# --- 3. MAIN PREDICTION FUNCTION ---
def generate_prediction_map():
    try:
        logger.info("Starting final prediction pipeline...")

        # --- 1. Load All Artifacts ---
        logger.info("Loading map, features, embeddings, and raw data...")
        gdf_map = gpd.read_file(SHAPEFILE_PATH)
        df_numeric_features = pd.read_csv(IMPUTED_FEATURES_PATH)
        df_gnn = pd.read_csv(GNN_EMBEDDINGS_PATH)
        df_raw = pd.read_csv(RAW_DATA_PATH) # <-- NEW: Load raw data

        logger.info("Loading saved model and processors...")
        with open(FINAL_PREPROCESSOR_PATH, 'rb') as f:
            preprocessor = pickle.load(f)
        with open(FINAL_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FINAL_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        
        logger.info("All artifacts loaded successfully.")

        # --- 2. Prepare the Full Prediction Dataset ---
        
        # Get the feature lists from the preprocessor
        try:
            num_cols = preprocessor.named_transformers_['num'].feature_names_in_
            cat_cols = preprocessor.named_transformers_['cat'].feature_names_in_
            gnn_cols = preprocessor.named_transformers_['gnn'].feature_names_in_
            all_features = list(num_cols) + list(cat_cols) + list(gnn_cols)
        except Exception:
            # Fallback based on your logs if preprocessor is complex
            num_cols = ['Age', 'Obese(%)', 'Cholesterol(%)', 'PM2.5', 'NO2', 'SO2', 'Population_Density', 'Sleep_Hours', 'NDVI']
            cat_cols = ['Gender', 'Tobacco_Use', 'Alcohol_Use', 'Hypertension', 'Diabetes', 'Aging_Population', 'SES_Income', 'Healthcare_Access', 'Urban_or_Rural']
            gnn_cols = [f'gnn_embed_{i}' for i in range(64)]
            all_features = num_cols + cat_cols + gnn_cols

        # Standardize district names
        gdf_map["District"] = gdf_map["NAME_2"].str.strip().str.title()
        df_numeric_features["District"] = df_numeric_features["District"].str.strip().str.title()
        df_gnn["District"] = df_gnn["District"].str.strip().str.title()
        df_raw["District"] = df_raw["District"].str.strip().str.title()

        # --- NEW: Aggregate categorical features ---
        logger.info("Aggregating categorical features (using mode)...")
        # Define the mode function
        def get_mode(x):
            try:
                return x.mode().iloc[0]
            except IndexError:
                return np.nan # Return NaN if the group is empty
        
        # Create an aggregation dictionary for just the categorical columns
        agg_funcs = {col: get_mode for col in cat_cols}
        df_cat_features = df_raw.groupby('District').agg(agg_funcs).reset_index()

        # --- Merge all three data sources ---
        logger.info("Merging all data sources for final prediction...")
        # 1. Start with the master map (all 676+ districts)
        merged_data = gdf_map
        # 2. Merge imputed numeric features (from Stage A)
        merged_data = merged_data.merge(df_numeric_features, on="District", how="left")
        # 3. Merge GNN features (from Stage C)
        merged_data = merged_data.merge(df_gnn, on="District", how="left")
        # 4. Merge aggregated categorical features
        merged_data = merged_data.merge(df_cat_features, on="District", how="left")

        # Fill any NaNs that may have resulted from the merge
        merged_data[gnn_cols] = merged_data[gnn_cols].fillna(0)
        # We don't need to fill NaNs for cat_cols, the preprocessor will do it.
        
        # Ensure the final dataframe only has the features the model expects
        X_predict_df = merged_data[all_features].copy()

        # --- 3. Preprocess and Scale the Full Dataset ---
        logger.info("Preprocessing and scaling the full dataset...")
        # The preprocessor will impute missing categorical features with "missing"
        X_processed = preprocessor.transform(X_predict_df)
        X_scaled = scaler.transform(X_processed)
        
        logger.info(f"Final data shape for prediction: {X_scaled.shape}")

        # --- 4. Make Predictions ---
        logger.info("Making predictions for all districts...")
        predictions_numeric = model.predict(X_scaled)
        
        # --- 5. Decode Predictions ---
        predictions_named = label_encoder.inverse_transform(predictions_numeric)
        
        # Add the final predictions back to our map
        merged_data['Predicted_NCD'] = predictions_named
        
        logger.info("Predictions complete.")

        # --- 6. Plot the Final Map ---
        logger.info("Generating final risk map...")
        plt.figure(figsize=(15, 10))
        merged_data.plot(
            column="Predicted_NCD",
            categorical=True, 
            cmap="Set3",      
            legend=True,
            edgecolor="black",
            linewidth=0.1,
            legend_kwds={'bbox_to_anchor': (1.1, 1)} # Move legend to the side
        )
        plt.title("Final Predicted NCD Risk Map for All Districts", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        
        map_save_path = os.path.join('artifacts', 'final_prediction_map.png')
        plt.savefig(map_save_path)
        logger.info(f"Final map saved to {map_save_path}")
        plt.show()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

# --- 7. RUN THE SCRIPT ---
if __name__ == "__main__":
    generate_prediction_map()