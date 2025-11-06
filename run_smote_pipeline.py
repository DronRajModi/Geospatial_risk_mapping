# run_smote_pipeline.py

import os
import sys
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

#  SETUP LOGGER   
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
logging.basicConfig(
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SMOTEPipelineLogger")

#  DEFINE EXCEPTION 
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        self.error_message = f"Error in {file_name} at line {exc_tb.tb_lineno}: {error_message}"
    def __str__(self): return self.error_message

# HELPER FUNCTIONS 
def _load_df(maybe_df_or_path):
    if isinstance(maybe_df_or_path, pd.DataFrame):
        return maybe_df_or_path.copy()
    if isinstance(maybe_df_or_path, str):
        if not os.path.exists(maybe_df_or_path):
             raise CustomException(FileNotFoundError(f"File not found: {maybe_df_or_path}"), sys)
        ext = os.path.splitext(maybe_df_or_path)[1].lower()
        if ext == ".csv": return pd.read_csv(maybe_df_or_path)
        else: return pd.read_csv(maybe_df_or_path)
    raise CustomException(ValueError("Input must be a pandas DataFrame or a valid file path."), sys)

def _get_target_column(df: pd.DataFrame):
    target_candidates = ["Disease_NCD", "target", "label", "y"]
    for candidate in target_candidates:
        if candidate in df.columns: return candidate
    logger.warning("Standard target column name not found, falling back to the last column.")
    return df.columns[-1]

# SPATIAL IMPUTATION 
from sklearn.impute import KNNImputer
from dataclasses import dataclass

@dataclass
class SpatialImputationConfig:
    imputed_data_path: str = os.path.join('artifacts', 'imputed_features.csv')
    shapefile_path: str = os.path.join('notebook', 'india_districts.shp')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class SpatialImputer:
    def __init__(self):
        self.imputation_config = SpatialImputationConfig()
    def initiate_spatial_imputation(self, district_col_name='District', shapefile_district_col='DISTRICT'):
        logger.info("Starting Spatial Imputation (Stage A)")
        try:
            df = pd.read_csv(self.imputation_config.raw_data_path)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if 'Year' in numeric_cols: numeric_cols.remove('Year')
            target_col_name = _get_target_column(df)
            if target_col_name in numeric_cols: numeric_cols.remove(target_col_name)
            df_agg = df.groupby(district_col_name)[numeric_cols].mean().reset_index()
            logger.info(f"Aggregated {len(df_agg)} districts to their mean profiles.")
            gdf = gpd.read_file(self.imputation_config.shapefile_path)
            gdf[district_col_name] = gdf[shapefile_district_col].str.strip().str.title()
            gdf_merged = gdf.merge(df_agg, on=district_col_name, how='left')
            numeric_cols = df_agg.select_dtypes(include=['number']).columns.tolist() 
            features_to_impute = [district_col_name, 'geometry'] + numeric_cols
            gdf_impute = gdf_merged[[col for col in features_to_impute if col in gdf_merged.columns]]
            logger.info("Preparing for KNN Imputation")
            gdf_impute = gdf_impute.to_crs(epsg=3857) 
            gdf_impute['centroid_x'] = gdf_impute.geometry.centroid.x
            gdf_impute['centroid_y'] = gdf_impute.geometry.centroid.y
            impute_cols = ['centroid_x', 'centroid_y'] + numeric_cols
            imputer_data = gdf_impute[impute_cols].copy()
            logger.info("Running KNNImputer (k=5) based on spatial centroids...")
            imputer = KNNImputer(n_neighbors=5, weights="distance")
            imputed_array = imputer.fit_transform(imputer_data)
            df_imputed = pd.DataFrame(imputed_array, columns=impute_cols)
            final_df = pd.concat(
                [gdf_impute[[district_col_name]].reset_index(drop=True),
                 df_imputed.drop(['centroid_x', 'centroid_y'], axis=1).reset_index(drop=True)],
                axis=1
            )
            final_df.to_csv(self.imputation_config.imputed_data_path, index=False)
            logger.info(f"Spatial Imputation complete. Full dataset saved to {self.imputation_config.imputed_data_path}")
            return self.imputation_config.imputed_data_path
        except Exception as e:
            raise CustomException(e, sys)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC 
from imblearn.over_sampling import SMOTE 

def build_preprocessing_pipeline(df: pd.DataFrame, include_gnn=False):
    try:
        target_col = _get_target_column(df)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        if 'Year' in numeric_cols: numeric_cols.remove('Year')
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if target_col in categorical_cols: categorical_cols.remove(target_col)
        if 'District' in categorical_cols: categorical_cols.remove('District')
        gnn_cols = [col for col in df.columns if col.startswith('gnn_embed_')]
        if gnn_cols:
            numeric_cols = [col for col in numeric_cols if col not in gnn_cols]
        logger.info(f"Numeric columns: {numeric_cols}")
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"GNN columns: {gnn_cols}")
        categorical_features_indices = [df.columns.get_loc(c) for c in categorical_cols if c in df]
        logger.info(f"Categorical feature indices for SMOTE-NC: {categorical_features_indices}")
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers = []
        if numeric_cols: transformers.append(("num", numeric_pipeline, numeric_cols))
        if categorical_cols: transformers.append(("cat", categorical_pipeline, categorical_cols))
        if include_gnn and gnn_cols:
            gnn_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
            transformers.append(("gnn", gnn_pipeline, gnn_cols))
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        return preprocessor, numeric_cols, categorical_cols, categorical_features_indices
    except Exception as e:
        raise CustomException(e, sys)

def initiate_data_transformation_smote(train_input, test_input, artifacts_dir="artifacts"):
    try:
        logger.info("Initiating Data Transformation Component (Baseline SMOTE-NC)")
        train_df = _load_df(train_input)
        test_df = _load_df(test_input)
        target_col = _get_target_column(train_df)
        train_df.dropna(subset=[target_col], inplace=True)
        test_df.dropna(subset=[target_col], inplace=True)
        X_train_df = train_df.drop(columns=[target_col], errors='ignore')
        y_train_series = train_df[target_col]
        X_test_df = test_df.drop(columns=[target_col], errors='ignore')
        y_test_series = test_df[target_col]
        logger.info(f"Original training class distribution:\n{y_train_series.value_counts().to_string()}")
        if 'District' in X_train_df.columns:
            X_train_df_no_dist = X_train_df.drop(columns=['District'])
            X_test_df_no_dist = X_test_df.drop(columns=['District'])
            logger.info("Dropped 'District' column before SMOTE-NC and preprocessing.")
        else:
            X_train_df_no_dist = X_train_df
            X_test_df_no_dist = X_test_df
        preprocessor, numeric_cols, categorical_cols, categorical_features_indices = build_preprocessing_pipeline(X_train_df_no_dist, include_gnn=False)
        logger.info("Applying SMOTE-NC with a CONSERVATIVE strategy...")
        min_class_count = y_train_series.value_counts().min()
        k_neighbors_smote = max(1, min_class_count - 1)
        current_counts = y_train_series.value_counts().to_dict()
        sampling_strategy = {key: max(count, 1500) for key, count in current_counts.items() if count < 1500}
        for key, count in current_counts.items():
            if key not in sampling_strategy: sampling_strategy[key] = count
        logger.info(f"Setting SMOTE-NC k_neighbors to: {k_neighbors_smote}")
        logger.info(f"Using conservative sampling strategy: {sampling_strategy}")
        smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42, k_neighbors=k_neighbors_smote, sampling_strategy=sampling_strategy) 
        X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train_df_no_dist, y_train_series)
        logger.info(f"SMOTE-NC finished. New training shape: {X_train_resampled.shape}")
        logger.info("Applying preprocessing (imputer/encoder)...")
        X_train_processed = preprocessor.fit_transform(X_train_resampled)
        X_test_processed = preprocessor.transform(X_test_df_no_dist)
        logger.info("Applying StandardScaler.")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        logger.info("Encoding target variable.")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_resampled)
        y_test_encoded = label_encoder.transform(y_test_series)
        train_arr_scaled_aug = np.hstack([X_train_scaled, y_train_encoded.reshape(-1, 1)])
        test_arr_scaled = np.hstack([X_test_scaled, y_test_encoded.reshape(-1, 1)])
        preprocessor_path = os.path.join(artifacts_dir, "transformer_smote.pkl")
        scaler_path = os.path.join(artifacts_dir, "scaler_smote.pkl")
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        with open(preprocessor_path, "wb") as f: pickle.dump(preprocessor, f)
        with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
        with open(label_encoder_path, "wb") as f: pickle.dump(label_encoder, f)
        logger.info(f"AUGMENTED train_arr shape: {train_arr_scaled_aug.shape}, test_arr shape: {test_arr_scaled.shape}")
        return train_arr_scaled_aug, test_arr_scaled, X_train_df, y_train_series, X_test_df, y_test_series
    except Exception as e:
        raise CustomException(e, sys)

# MODEL TRAINER (RandomForest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def initiate_model_training(train_arr, test_arr, artifacts_dir="artifacts", model_name="model_rf.pkl"):
    try:
        logger.info(f"Initiating Model Training Component ({model_name})")
        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        logger.info(f"Training {model_name} with default parameters.")
        model.fit(X_train, y_train)
        logger.info("Evaluating model on test set.")
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        label_encoder_path = os.path.join(artifacts_dir, "label_encoder.pkl")
        target_names = None
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)
                target_names = label_encoder.classes_
        report = classification_report(y_test, preds, target_names=target_names, zero_division=0)
        model_path = os.path.join(artifacts_dir, model_name)
        with open(model_path, "wb") as f: pickle.dump(model, f)
        logger.info(f"[{model_name}] model saved to: {model_path}")
        logger.info(f"[{model_name}] Accuracy: {acc:.4f}, F1 (weighted): {f1_w:.4f}")
        logger.info(f"[{model_name}] Classification Report:\n{report}")
        result = { "model_path": model_path, "accuracy": acc, "f1_weighted": f1_w }
        logger.info(f"Model Training Component ({model_name}) Finished")
        return result
    except Exception as e:
        raise CustomException(e, sys)

import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import from_networkx
import libpysal
from libpysal.weights import Queen

@dataclass
class GraphFeaturesConfig:
    shapefile_path: str = os.path.join('notebook', 'india_districts.shp')
    imputed_features_path: str = os.path.join('artifacts', 'imputed_features.csv')
    embedding_output_path: str = os.path.join('artifacts', 'gnn_embeddings.csv')

class GraphSAGEModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, in_channels) 
    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        embedding = self.conv1(x, edge_index)
        embedding = F.relu(embedding)
        return embedding
    def decode(self, embedding, edge_index):
        x_reconstructed = self.conv2(embedding, edge_index)
        return x_reconstructed
    def forward(self, data):
        embedding = self.encode(data)
        x_reconstructed = self.decode(embedding, data.edge_index)
        return x_reconstructed, embedding

class GraphFeatureGenerator:
    def __init__(self, embedding_dim=64, epochs=50):
        self.config = GraphFeaturesConfig()
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"GNN (Stage C) will run on {self.device}")
    def _build_graph_from_shapefile(self, gdf):
        logger.info("Building spatial adjacency graph (Queen Contiguity) using libpysal...")
        try:
            w = Queen.from_dataframe(gdf, silence_warnings=True)
            g = w.to_networkx()
        except Exception as e:
            logger.error(f"Failed to build graph with libpysal: {e}. Falling back to simple touches.")
            adjacency_list = gdf.geometry.touches(gdf.geometry)
            g = nx.Graph()
            for i, neighbors in enumerate(adjacency_list):
                if not neighbors: 
                    g.add_node(i)
                    continue
                for neighbor_idx in np.where(neighbors)[0]:
                    if neighbor_idx > i: g.add_edge(i, neighbor_idx)
        logger.info(f"Graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
        if g.number_of_edges() == 0:
             logger.warning("Graph has 0 edges. The GNN will not learn spatial features.")
        return g
    def initiate_graph_feature_generation(self, shapefile_district_col='NAME_2'):
        try:
            logger.info("Starting Stage C: Graph Feature Generation")
            gdf = gpd.read_file(self.config.shapefile_path)
            gdf['District'] = gdf[shapefile_district_col].str.strip().str.title()
            gdf = gdf.set_index('District', drop=False) 
            df_features = pd.read_csv(self.config.imputed_features_path)
            df_features = df_features.set_index('District', drop=False)
            df_features = df_features.reindex(gdf.index)
            gdf_for_graph = gdf.reset_index(drop=True)
            G = self._build_graph_from_shapefile(gdf_for_graph)
            logger.info("Preparing data for PyTorch Geometric")
            node_features_df = df_features.drop(columns=['District'])
            node_features = node_features_df.values.astype(np.float32)
            if np.isnan(node_features).any():
                logger.warning("NaNs found in imputed features. Filling with 0 before GNN training.")
                node_features = np.nan_to_num(node_features)
            x = torch.tensor(node_features, dtype=torch.float)
            pyg_data = from_networkx(G)
            edge_index = pyg_data.edge_index
            data = Data(x=x, edge_index=edge_index).to(self.device)
            logger.info("Defining GNN model...")
            in_channels = data.num_node_features
            model = GraphSAGEModel(
                in_channels=in_channels,
                hidden_channels=self.embedding_dim
            ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            logger.info(f"Starting GNN training for {self.epochs} epochs...")
            model.train()
            for epoch in range(self.epochs):
                optimizer.zero_grad()
                x_reconstructed, embedding = model(data)
                loss = F.mse_loss(x_reconstructed, data.x) 
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    logger.info(f'GNN Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}')
            logger.info("Extracting final node embeddings...")
            model.eval()
            with torch.no_grad():
                _ , final_embeddings = model(data)
                final_embeddings = final_embeddings.cpu().numpy()
            logger.info(f"Saving embeddings to {self.config.embedding_output_path}")
            embed_cols = [f'gnn_embed_{i}' for i in range(self.embedding_dim)]
            df_embeddings = pd.DataFrame(final_embeddings, columns=embed_cols)
            df_embeddings['District'] = gdf.index.values
            df_embeddings = df_embeddings[['District'] + embed_cols]
            df_embeddings = df_embeddings.drop_duplicates(subset=['District'])
            logger.info(f"Dropped duplicates. Final embeddings shape: {df_embeddings.shape}")
            os.makedirs(os.path.dirname(self.config.embedding_output_path), exist_ok=True)
            df_embeddings.to_csv(self.config.embedding_output_path, index=False)
            logger.info("Stage C: Graph Feature Generation Complete")
            return self.config.embedding_output_path
        except Exception as e:
            raise CustomException(e, sys)

def initiate_final_rf_training(
    X_train_df, y_train_series, # Original training data
    X_test_df, y_test_series,   # Original test data
    gnn_embeddings_path,        # Path to GNN features
    artifacts_dir="artifacts"
):
    try:
        logger.info("Initiating FINAL Spatially-Aware RandomForest (Stage D)")
        
        # 1. Load GNN embeddings
        df_gnn = pd.read_csv(gnn_embeddings_path)
        df_gnn['District'] = df_gnn['District'].str.strip().str.title()
        df_gnn = df_gnn.drop_duplicates(subset=['District'])

        # 2. Merge GNN features into train and test dataframes
        logger.info("Merging GNN features into training and test sets...")
        X_train_spatial = pd.merge(X_train_df, df_gnn, on='District', how='left')
        X_test_spatial = pd.merge(X_test_df, df_gnn, on='District', how='left')
        
        # Failsafe for length mismatch
        if len(X_train_spatial) != len(y_train_series):
            logger.warning("Length mismatch after merge. Aligning indices.")
            X_train_spatial = X_train_spatial.set_index('District').reindex(X_train_df['District']).reset_index()
        if len(X_test_spatial) != len(y_test_series):
            logger.warning("Length mismatch after merge. Aligning indices.")
            X_test_spatial = X_test_spatial.set_index('District').reindex(X_test_df['District']).reset_index()

        gnn_cols = [col for col in X_train_spatial.columns if col.startswith('gnn_embed_')]
        X_train_spatial[gnn_cols] = X_train_spatial[gnn_cols].fillna(0)
        X_test_spatial[gnn_cols] = X_test_spatial[gnn_cols].fillna(0)
        
        X_train_spatial = X_train_spatial.drop(columns=['District'])
        X_test_spatial = X_test_spatial.drop(columns=['District'])
        
        logger.info(f"New training feature shape (with GNN): {X_train_spatial.shape}")
        
        # 3. Apply SMOTE-NC 
        logger.info("Applying SMOTE-NC to the new spatio-temporal features...")
        
        preprocessor_gnn, _, _, categorical_features_indices_gnn = build_preprocessing_pipeline(X_train_spatial, include_gnn=True)
        
        min_class_count = y_train_series.value_counts().min()
        k_neighbors_smote = max(1, min_class_count - 1)

        current_counts = y_train_series.value_counts().to_dict()
        sampling_strategy = {
            key: max(count, 1500) for key, count in current_counts.items() if count < 1500
        }
        for key, count in current_counts.items():
            if key not in sampling_strategy:
                sampling_strategy[key] = count
        
        logger.info(f"Setting SMOTE-NC k_neighbors to: {k_neighbors_smote}")
        smote_nc = SMOTENC(
            categorical_features=categorical_features_indices_gnn, 
            random_state=42, 
            k_neighbors=k_neighbors_smote,
            sampling_strategy=sampling_strategy
        )
        X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train_spatial, y_train_series)
        logger.info(f"SMOTE-NC finished. New training shape: {X_train_resampled.shape}")

        # 4. Preprocess and Scale the new data
        logger.info("Applying preprocessing and scaling to new spatio-temporal data...")
        
        X_train_processed = preprocessor_gnn.fit_transform(X_train_resampled)
        X_test_processed = preprocessor_gnn.transform(X_test_spatial)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_test_scaled = scaler.transform(X_test_processed)
        
        # 5. Encode Targets
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train_resampled)
        y_test_encoded = label_encoder.transform(y_test_series)
        
        # 6. Train the Final Model
        logger.info("Training FINAL Spatially-Aware RandomForest (Stage D)")
        final_train_arr = np.hstack([X_train_scaled, y_train_encoded.reshape(-1, 1)])
        final_test_arr = np.hstack([X_test_scaled, y_test_encoded.reshape(-1, 1)])
        
        final_preprocessor_path = os.path.join(artifacts_dir, "final_preprocessor.pkl")
        final_scaler_path = os.path.join(artifacts_dir, "final_scaler.pkl")
        with open(final_preprocessor_path, "wb") as f: pickle.dump(preprocessor_gnn, f)
        with open(final_scaler_path, "wb") as f: pickle.dump(scaler, f)
        logger.info(f"Final preprocessor and scaler saved for prediction.")

        final_model_results = initiate_model_training(
            final_train_arr,
            final_test_arr,
            artifacts_dir,
            model_name="model_final_spatial_rf.pkl" 
        )
        
        logger.info("Final Spatially-Aware RandomForest training complete.")
        return final_model_results

    except Exception as e:
        raise CustomException(e, sys)

# --- 9. MAIN PIPELINE ---
if __name__ == "__main__":
    logger.info("Main SMOTE-NC training pipeline started")
    try:
        artifacts_dir = "artifacts"
        train_data_path = os.path.join(artifacts_dir, "train.csv")
        test_data_path = os.path.join(artifacts_dir, "test.csv")
        raw_data_path = os.path.join('artifacts', 'data.csv') 

        # --- STEP 1: (Assume data exists) ---
        logger.info("STEP 1: Data Ingestion (Skipped, using existing artifacts)")
        
        # --- STEP 2: Stage A - Spatial Imputation ---
        logger.info("STEP 2: Starting Stage A: Spatial Imputation")
        SHAPEFILE_DISTRICT_COL = 'NAME_2' 
        spatial_imputer_obj = SpatialImputer()
        imputed_features_path = spatial_imputer_obj.initiate_spatial_imputation(
            district_col_name='District',
            shapefile_district_col=SHAPEFILE_DISTRICT_COL 
        )
        logger.info(f"Stage A complete. Full imputed feature set at {imputed_features_path}")
        
        # --- STEP 3: Data Transformation & Augmentation  ---
        logger.info("STEP 3: Running Data Transformation (with SMOTE-NC augmentation)")
        (
            train_arr_scaled_aug, test_arr_scaled, 
            X_train_df, y_train_series, 
            X_test_df, y_test_series
        ) = initiate_data_transformation_smote(
            train_input=train_data_path,
            test_input=test_data_path,
            artifacts_dir=artifacts_dir
        )
        logger.info("Data Transformation and Augmentation complete.")
        
        # --- STEP 4: Graph Feature Generation ---
        logger.info("STEP 4: Running Graph Feature Generation (Stage C)")
        gnn_obj = GraphFeatureGenerator(embedding_dim=64, epochs=50) 
        gnn_embedding_path = gnn_obj.initiate_graph_feature_generation(
            shapefile_district_col=SHAPEFILE_DISTRICT_COL
        )
        logger.info(f"Stage C complete. GNN Embeddings saved to {gnn_embedding_path}")

        # --- STEP 5: Model Trainer (Baseline RandomForest) ---
        logger.info("STEP 5: Running Baseline Model Trainer (on SMOTE data, NO GNN)")
        model_trainer_obj = initiate_model_training(
            train_arr=train_arr_scaled_aug, 
            test_arr=test_arr_scaled,   
            artifacts_dir=artifacts_dir,
            model_name="model_baseline_rf.pkl"
        )
        logger.info(f"Baseline Model Trainer complete. Accuracy: {model_trainer_obj['accuracy']:.4f}")
        
        # --- STEP 6: Final Model Trainer  ---
        logger.info("STEP 6: Running Final Spatially-Aware RandomForest Model (Stage D)")
        final_model_results = initiate_final_rf_training(
            X_train_df=X_train_df, y_train_series=y_train_series,
            X_test_df=X_test_df, y_test_series=y_test_series,
            gnn_embeddings_path=gnn_embedding_path,
            artifacts_dir=artifacts_dir
        )
        logger.info(f"Final Spatially-Aware Model complete. Accuracy: {final_model_results['accuracy']:.4f}")
        
        logger.info("Main SMOTE-NC training pipeline finished successfully")

    except Exception as e:
        logger.error(f"An error occurred in the main SMOTE-NC pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise e