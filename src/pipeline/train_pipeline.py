# # src/pipeline/train_pipeline.py

# import sys
# from src.exception import CustomException
# from src.logger import logging

# # --- Import ALL your components ---
# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation
# from src.components.model_trainer import ModelTrainer

# # --- Import our NEW Stage A component ---
# from src.components.spatial_imputation import SpatialImputer

# class TrainPipeline:
#     def __init__(self):
#         pass

#     def run_pipeline(self):
#         logging.info("Main training pipeline started")
#         try:
#             # --- STEP 1: Data Ingestion (Your original step) ---
#             logging.info("Running Data Ingestion")
#             ingest_obj = DataIngestion()
#             # These paths come from your data_ingestion.py
#             train_data_path, test_data_path, raw_data_path = ingest_obj.initiate_data_ingestion()
#             logging.info(f"Data Ingestion complete. Train/Test files at: {train_data_path}, {test_data_path}")

            
#             # --- STEP 2: Stage A (NEW) - Spatial Imputation ---
#             logging.info("Starting Stage A: Spatial Imputation")
            
#             # This is the correct column name you found: 'NAME_2'
#             SHAPEFILE_DISTRICT_COL = 'NAME_2' 
            
#             spatial_imputer_obj = SpatialImputer()
#             # This step creates 'artifacts/imputed_features.csv'
#             imputed_features_path = spatial_imputer_obj.initiate_spatial_imputation(
#                 district_col_name='District', # Assumes 'District' in your data.csv
#                 shapefile_district_col=SHAPEFILE_DISTRICT_COL 
#             )
#             logging.info(f"Stage A complete. Full imputed feature set at {imputed_features_path}")

            
#             # --- STEP 3: Data Transformation (Your original step) ---
#             logging.info("Running Data Transformation")
#             data_transformation_obj = DataTransformation()
#             # This step uses your original train.csv and test.csv
#             train_arr, test_arr, transformer_obj_path = data_transformation_obj.initiate_data_transformation(
#                 train_data_path, 
#                 test_data_path
#             )
#             logging.info("Data Transformation complete. train_arr and test_arr created.")

            
#             # --- STEP 4: Model Trainer (Your original step) ---
#             logging.info("Running Model Trainer (Original RandomForest)")
#             model_trainer_obj = ModelTrainer()
#             # This trains the model on your original, non-imputed data
#             model_trainer_obj.initiate_model_training(
#                 train_arr, 
#                 test_arr
#             )
#             logging.info("Original Model Trainer complete.")

            
#             # --- FUTURE STEPS WILL GO HERE ---
#             logging.info("--- Stages B, C, and D (GAN, GNN, LSTM) will be added here later ---")

            
#             logging.info("Main training pipeline finished successfully")

#         except Exception as e:
#             logging.error("An error occurred in the main training pipeline")
#             raise CustomException(e, sys)

# # --- This makes the file runnable from the command line ---
# if __name__ == "__main__":
#     pipeline = TrainPipeline()
#     pipeline.run_pipeline()






# src/pipeline/train_pipeline.py

import sys
from src.exception import CustomException
from src.logger import logging

# --- Import components ---
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.spatial_imputation import SpatialImputer


class TrainPipeline:
    def __init__(self, artifacts_dir="artifacts"):
        self.artifacts_dir = artifacts_dir

    def run_pipeline(self):
        logging.info("Main training pipeline started")
        try:
            # --- STEP 1: Data Ingestion ---
            logging.info("Running Data Ingestion")
            ingest_obj = DataIngestion(artifacts_dir=self.artifacts_dir)
            ingestion_result = ingest_obj.initiate_data_ingestion()
            logging.info(f"Data Ingestion complete. Training + test data processed.")

            # If you want the paths for any reason:
            train_data_path = f"{self.artifacts_dir}/train.csv"
            test_data_path = f"{self.artifacts_dir}/test.csv"

            # --- STEP 2: Spatial Imputation ---
            logging.info("Starting Stage A: Spatial Imputation")
            SHAPEFILE_DISTRICT_COL = 'NAME_2'  # shapefile column
            spatial_imputer_obj = SpatialImputer()
            imputed_features_path = spatial_imputer_obj.initiate_spatial_imputation(
                district_col_name='District', 
                shapefile_district_col=SHAPEFILE_DISTRICT_COL
            )
            logging.info(f"Stage A complete. Imputed features saved at {imputed_features_path}")

            # --- STEP 3: Data Transformation ---
            logging.info("Running Data Transformation")
            data_transform_obj = DataTransformation()
            train_arr, test_arr, transformer_obj_path = data_transform_obj.initiate_data_transformation(
                train_data_path,
                test_data_path,
                artifacts_dir=self.artifacts_dir
            )
            logging.info(f"Data Transformation complete. train_arr shape: {train_arr.shape}, test_arr shape: {test_arr.shape}")

            # --- STEP 4: Model Training ---
            logging.info("Running Model Trainer")
            model_trainer_obj = ModelTrainer()
            training_result = model_trainer_obj.initiate_model_training(
                train_arr,
                test_arr,
                artifacts_dir=self.artifacts_dir
            )
            logging.info(f"Model training complete. Model saved at {training_result['model_path']}")
            logging.info(f"Accuracy: {training_result['accuracy']:.4f}, F1 (weighted): {training_result['f1_weighted']:.4f}")

            logging.info("Main training pipeline finished successfully")

        except Exception as e:
            logging.error("An error occurred in the training pipeline")
            raise CustomException(e, sys)


# --- Make the pipeline runnable ---
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
